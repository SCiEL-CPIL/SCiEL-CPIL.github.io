"""HP lattice protein folding on the cubic lattice - beam search with lazy state copying."""
import time
import random

# EVOLVE-BLOCK-START

# Precompute neighbor offsets for 3D
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _energy_of_chain(sequence, chain):
    """Compute the energy (negative number of H-H contacts) of a complete chain."""
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            px, py, pz = p
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (px + dx, py + dy, pz + dz)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_fold(sequence, dimension, beam_width, rng, time_limit):
    """
    Beam search for HP protein folding.
    Each beam entry is a tuple: (energy, chain, occupied_set)
    We use lazy copying of the occupied set to reduce overhead.
    """
    start = (0, 0, 0)
    # Beam entry: (energy, chain_list, occupied_dict)
    # occupied_dict maps position -> index in chain
    beam = [(-0, [start], {start: 0})]
    
    n = len(sequence)
    
    for i in range(1, n):
        if not beam:
            break
        
        new_beam = []
        seen = set()
        
        for energy, chain, occupied in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                pos = (lx + dx, ly + dy, lz + dz)
                if pos in occupied:
                    continue
                
                # Compute new energy
                new_energy = energy
                if sequence[i] == 'H':
                    for odx, ody, odz in _NEIGHBOR_OFFSETS:
                        nb = (pos[0] + odx, pos[1] + ody, pos[2] + odz)
                        j = occupied.get(nb)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            new_energy -= 1
                
                new_chain = chain + [pos]
                new_occupied = dict(occupied)
                new_occupied[pos] = i
                
                # Use a hash to avoid duplicates in beam
                # We track by the last few positions to reduce collisions
                # Actually, let's just add to new_beam and deduplicate by energy+pos
                key = (new_energy, pos)
                if key not in seen:
                    seen.add(key)
                    new_beam.append((new_energy, new_chain, new_occupied))
        
        if not new_beam:
            break
        
        # Sort by energy (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
        # Time check
        if time.time() > time_limit:
            break
    
    if not beam:
        return None
    
    # Return the best chain
    best = min(beam, key=lambda x: x[0])
    return best[1]


def _greedy_fold(sequence, dimension, rng):
    """Fast greedy folding as a fallback or initial guess."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        lx, ly, lz = last
        options = []
        
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            pos = (lx + dx, ly + dy, lz + dz)
            if pos in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for odx, ody, odz in _NEIGHBOR_OFFSETS:
                    nb = (pos[0] + odx, pos[1] + ody, pos[2] + odz)
                    j = occupied.get(nb)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for odx, ody, odz in _NEIGHBOR_OFFSETS
                       if (pos[0] + odx, pos[1] + ody, pos[2] + odz) not in occupied)
            
            score = gain * 10 + free * 0.1 + rng.random()
            options.append((score, pos))
        
        if not options:
            return None
        
        best_pos = max(options)[1]
        occupied[best_pos] = i
        chain.append(best_pos)
    
    return chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0, 0, 0]]
    
    # Total time budget: 330s for all sequences, but we're called per sequence.
    # The evaluator calls fold once per sequence. We need to be careful with time.
    # Let's assume we get roughly 330/26 ~ 12s per sequence on average, but some are longer.
    # Use a conservative per-call time limit based on sequence length.
    start_time = time.time()
    
    # Adaptive time limit: shorter sequences get less time, longer get more
    # But we need to finish all 26 within 330s total.
    # Let's use a fixed fraction: aim for ~10s per call, but cap based on remaining.
    # Since we don't know remaining time, use a heuristic.
    if n <= 48:
        time_limit = start_time + 8.0
    elif n <= 64:
        time_limit = start_time + 10.0
    elif n <= 100:
        time_limit = start_time + 12.0
    else:
        time_limit = start_time + 15.0
    
    # Ensure we don't exceed 300s total (safety margin)
    absolute_limit = 290.0  # seconds from start of program
    # We don't track global time easily, so use a conservative per-call limit.
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (unfolded)
    
    # First, try a few greedy runs to get a baseline
    rng = random.Random(42 + n)
    for attempt in range(10):
        chain = _greedy_fold(sequence, dimension, rng)
        if chain is not None:
            e = _energy_of_chain(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
    
    # Now try beam search with increasing width
    # Beam widths: start small, increase if time permits
    beam_widths = [50, 100, 200, 400, 600]
    
    for bw in beam_widths:
        if time.time() > time_limit - 2.0:
            break
        
        chain = _beam_fold(sequence, dimension, bw, rng, time_limit)
        if chain is not None:
            e = _energy_of_chain(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain
    
    # If beam search didn't improve, try a few more greedy runs
    if time.time() < time_limit - 1.0:
        for attempt in range(20):
            chain = _greedy_fold(sequence, dimension, rng)
            if chain is not None:
                e = _energy_of_chain(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0, 0] for i in range(n)]
    
    return best_chain

# EVOLVE-BLOCK-END
