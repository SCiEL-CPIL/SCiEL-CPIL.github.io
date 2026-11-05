"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

# Precompute neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _energy_correct(sequence, chain):
    """Compute energy: -number of H-H contacts on neighboring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=200, time_limit=None):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    if n == 2:
        return [(0, 0, 0), (1, 0, 0)]

    start_time = time.time()

    # Beam: list of (energy_so_far, chain_tuple, occupied_dict)
    beam = [(0, ((0, 0, 0), (1, 0, 0)), {(0, 0, 0): 0, (1, 0, 0): 1})]

    for i in range(2, n):
        # Time check
        if time_limit and (time.time() - start_time > time_limit):
            break
        if _time_left() < 15.0:
            break

        new_beam = []
        seen = set()
        residue = sequence[i]

        for energy_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                if residue == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                new_energy = energy_so_far - gain

                # Deduplicate: use the last position and a hash of the chain shape
                key = (site, chain_tuple[-3:] if len(chain_tuple) >= 3 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_energy, new_chain, new_occupied))

        if not new_beam:
            break

        # Sort by energy (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    # Return the best complete chain
    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def _local_search(sequence, chain, rng, max_iterations=5000, time_limit=10.0):
    """
    Local search using segment reversal (pivot moves) to improve energy.
    """
    n = len(chain)
    if n < 5:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, best_chain)

    start_time = time.time()

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 5.0:
            break

        # Try segment reversal
        seg_len = rng.randint(2, min(10, n - 2))
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Extract segment
        segment = best_chain[start_idx:end_idx + 1]
        reversed_segment = segment[::-1]

        # Check if reversal causes self-avoidance violations
        new_chain = best_chain[:]
        new_chain[start_idx:end_idx + 1] = reversed_segment

        # Quick check: verify no duplicates in the modified region
        valid = True

        # Check self-avoidance
        if len(set(new_chain)) != n:
            valid = False
        else:
            # Check unit steps around the modified region
            for k in range(max(0, start_idx - 1), min(n - 1, end_idx + 1)):
                diff = abs(new_chain[k][0] - new_chain[k+1][0]) + \
                       abs(new_chain[k][1] - new_chain[k+1][1]) + \
                       abs(new_chain[k][2] - new_chain[k+1][2])
                if diff != 1:
                    valid = False
                    break

        if not valid:
            continue

        new_energy = _energy_correct(sequence, new_chain)
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy

    return best_chain


def _backtrack_search(sequence, dimension, max_depth=None, time_limit=None):
    """
    Constraint-satisfaction backtracking search with conflict-directed pruning.
    Grows the chain step-by-step, but uses a more aggressive pruning strategy
    by tracking 'conflicts' (dead-ends) and avoiding similar configurations.
    This is a hybrid: it uses beam-like selection but with deeper exploration
    and conflict avoidance to escape local minima.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    if n == 2:
        return [(0, 0, 0), (1, 0, 0)]

    start_time = time.time()
    if max_depth is None:
        max_depth = n

    # We will use a DFS with a time limit and a 'seen' set of (last_pos, last_3, energy)
    # to avoid repeating similar dead-ends.
    best_chain = None
    best_energy = 0

    # State: (chain_tuple, occupied_dict, energy)
    # We start with a few initial directions
    initial_states = []
    for dx, dy, dz in _NEIGHBOR_OFFSETS[:3]:  # Only 3 directions to avoid symmetry
        init_chain = ((0, 0, 0), (dx, dy, dz))
        init_occ = {(0, 0, 0): 0, (dx, dy, dz): 1}
        initial_states.append((init_chain, init_occ, 0))

    seen_states = set()
    # To prevent infinite loops or too deep recursion without progress,
    # we limit the number of states we explore in the 'beam' of the DFS.
    # Actually, let's implement a simplified beam search with a 'conflict' history.
    # If a state leads to a dead end, we record its 'signature' and avoid similar ones.
    
    # Let's stick to a robust Beam Search but with a twist:
    # Instead of just keeping the top K by energy, we keep a diverse set
    # and use a 'temperature' to allow worse solutions to survive for exploration.
    
    beam_width = 500
    temperature = 0.1
    
    beam = initial_states[:]
    
    for i in range(2, n):
        if time_limit and (time.time() - start_time > time_limit):
            break
        if _time_left() < 15.0:
            break

        new_beam = []
        seen = set()
        residue = sequence[i]

        for chain_tuple, occupied, energy in beam:
            last = chain_tuple[-1]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                if residue == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                new_energy = energy - gain

                # Deduplicate
                key = (site, chain_tuple[-3:] if len(chain_tuple) >= 3 else chain_tuple)
                if key in seen:
                    continue
                seen.add(key)

                new_chain = chain_tuple + (site,)
                new_occupied = dict(occupied)
                new_occupied[site] = i

                new_beam.append((new_energy, new_chain, new_occupied))

        if not new_beam:
            break

        # Sort by energy
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width, but with some randomness for diversity
        # This helps escape local minima
        if len(new_beam) > beam_width:
            # Keep the best 80% and random 20%
            deterministic_count = int(beam_width * 0.8)
            random_count = beam_width - deterministic_count
            
            top_candidates = new_beam[:determinative_count] if deterministic_count > 0 else []
            # Actually, just taking top K is safer for correctness, 
            # but we can shuffle the tail a bit.
            # Let's stick to top K for now to ensure we don't degrade too much,
            # but we will run multiple instances with different seeds in the main fold function.
            
            beam = new_beam[:beam_width]
        else:
            beam = new_beam

    if not beam:
        return None

    best_energy, best_chain_tuple, _ = beam[0]
    return list(best_chain_tuple)


def fold(sequence, dimension):
    """
    Fold a protein sequence on the cubic lattice.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    if n == 2:
        return [[0, 0, 0], [1, 0, 0]]

    best_chain = None
    best_energy = 0

    # Time budget for this sequence
    seq_start = time.time()
    # Allocate time based on sequence length, but cap it
    time_budget = min(100.0, max(10.0, n * 0.5))
    
    # Phase 1: Beam search with different widths for diversity
    # Use multiple beam searches with varying widths
    beam_configs = [
        (150, 12345),
        (300, 54321),
        (200, 98765),
        (400, 11111),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.7:
            break
        if _time_left() < 20.0:
            break
            
        # Allocate remaining time for this beam search
        remaining = min(time_budget * 0.7 - (time.time() - seq_start), 20.0)
        if remaining <= 1.0:
            continue

        chain = _beam_search(sequence, dimension, beam_width=beam_width, time_limit=remaining)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chain
    if best_chain is not None and _time_left() > 15.0:
        rng = random.Random(42)
        # Allocate remaining time for local search
        ls_time = min(_time_left() - 15.0, time_budget * 0.3)
        if ls_time > 1.0:
            improved = _local_search(sequence, best_chain, rng, max_iterations=5000, time_limit=ls_time)
            improved_energy = _energy_correct(sequence, improved)
            if improved_energy < best_energy:
                best_chain = improved
                best_energy = improved_energy

    # Fallback: straight line if no valid chain found
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = _energy_correct(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
