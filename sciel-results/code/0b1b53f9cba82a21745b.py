"""HP lattice protein folding on the cubic lattice.
Strategy: Beam search growth + local kink-move optimization with deadline guard.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_TOTAL_DEADLINE = 330.0
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_remaining():
    return _TOTAL_DEADLINE - _get_elapsed() - 10.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy_incremental(sequence, chain, occupied, i):
    """Compute energy gain when adding residue i at chain[i]."""
    if sequence[i] != 'H':
        return 0
    gain = 0
    for q in _neighbours(chain[i]):
        j = occupied.get(q)
        if j is not None and j < i - 1 and sequence[j] == 'H':
            gain += 1
    return gain


def _beam_grow(sequence, dimension, beam_width, rng, time_limit):
    """Beam search chain growth with incremental energy tracking."""
    start = (0,) * dimension
    # Each beam entry: (chain, occupied, energy)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        if _get_elapsed() > time_limit:
            break
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute energy gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Count free neighbors for diversity
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain * 10 + free + rng.random() * 0.5
                options.append((score, site, gain))
            
            if not options:
                continue
            
            # Keep top options from this beam
            options.sort(key=lambda x: -x[0])
            for score, site, gain in options[:max(1, beam_width // 2)]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy + gain
                new_beams.append((new_chain, new_occupied, new_energy))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    
    if not beams:
        return None
    
    # Return the best beam
    best = max(beams, key=lambda x: x[2])
    return best[0]


def _kink_moves(sequence, chain, occupied, max_iterations, time_limit):
    """Local optimization using kink moves (3-residue rotations)."""
    current_energy = 0
    where = {p: i for i, p in enumerate(chain)}
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    current_energy += 1
    current_energy = -current_energy
    
    best_chain = list(chain)
    best_energy = current_energy
    best_occupied = dict(occupied)
    
    for iteration in range(max_iterations):
        if _get_elapsed() > time_limit:
            break
        if iteration % 100 == 0 and _time_remaining() < 5:
            break
        
        # Pick a random internal residue to try kink move
        if len(chain) < 4:
            break
        i = rng_sample = random.randint(1, len(chain) - 2)
        
        # Try all 6 possible kink directions (rotate middle residue)
        center = chain[i]
        prev = chain[i-1]
        next_ = chain[i+1]
        
        # The kink move rotates the center residue around the axis defined by prev and next
        # Actually, a kink move changes one residue to a new position that maintains unit steps
        # Let's try moving chain[i] to all 6 neighbors of its current position
        for new_pos in _neighbours(center):
            if new_pos == prev or new_pos == next_:
                continue
            if new_pos in best_occupied:
                continue
            
            # Check if this creates a valid self-avoiding chain
            # We need to verify that the chain remains self-avoiding
            # Since we're only changing one position, we just need to check it doesn't collide
            # with any other part of the chain
            valid = True
            for j, p in enumerate(best_chain):
                if j != i and p == new_pos:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute energy change
            old_energy = best_energy
            new_chain = list(best_chain)
            new_chain[i] = new_pos
            new_occupied = dict(best_occupied)
            del new_occupied[center]
            new_occupied[new_pos] = i
            
            # Recompute energy locally
            energy_delta = 0
            if sequence[i] == 'H':
                # Remove old contacts
                for q in _neighbours(center):
                    j = best_occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        energy_delta -= 1
                # Add new contacts
                for q in _neighbours(new_pos):
                    j = new_occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        energy_delta += 1
            # Also check if neighbors of i were H and had contacts with i
            if sequence[i-1] == 'H':
                for q in _neighbours(prev):
                    j = new_occupied.get(q)
                    if j is not None and j == i and sequence[i] == 'H':
                        pass  # Already counted
            if sequence[i+1] == 'H':
                for q in _neighbours(next_):
                    j = new_occupied.get(q)
                    if j is not None and j == i and sequence[i] == 'H':
                        pass
            
            new_energy = old_energy + energy_delta
            
            if new_energy > best_energy:
                best_chain = new_chain
                best_occupied = new_occupied
                best_energy = new_energy
                break
        else:
            # No improvement, continue
            continue
    
    return best_chain, best_energy


def _compute_energy(sequence, chain):
    """Compute total energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def fold(sequence, dimension):
    """Best conformation using beam search growth + local optimization."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = len(sequence)
    time_limit = _time_remaining()
    
    if time_limit <= 0:
        # Return a straight line
        return [[i] + [0] * (dimension - 1) for i in range(n)]
    
    # Allocate time: 60% for beam search, 40% for local optimization
    beam_time = time_limit * 0.5
    opt_time = time_limit * 0.4
    
    best_chain = None
    best_energy = 1  # Best (most negative) energy
    
    # Multiple beam search restarts
    num_restarts = max(1, int(time_limit / 2))
    beam_width = max(10, min(50, n // 2))
    
    for restart in range(num_restarts):
        if _get_elapsed() > _start_time + time_limit * 0.6:
            break
        
        rng = random.Random(12345 + restart * 1000 + len(sequence))
        chain = _beam_grow(sequence, dimension, beam_width, rng, _start_time + time_limit * 0.6)
        
        if chain is None or len(chain) != n:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Local optimization with kink moves
    if _time_remaining() > 5 and len(best_chain) > 3:
        occupied = {p: i for i, p in enumerate(best_chain)}
        opt_time_remaining = _time_remaining() - 5
        max_kink_iters = max(100, int(opt_time_remaining * 100))
        optimized_chain, optimized_energy = _kink_moves(
            sequence, best_chain, occupied, max_kink_iters, _start_time + _time_remaining() - 5
        )
        if optimized_energy < best_energy:
            best_chain = optimized_chain
            best_energy = optimized_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
