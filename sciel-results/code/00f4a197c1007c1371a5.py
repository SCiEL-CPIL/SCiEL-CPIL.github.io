"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Return the 4 unit-step neighbours of a 2D point (tuple) as a list."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    temperature adds randomness to break ties / explore.
    Returns a list of tuples (the chain) or None if stuck.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count potential H-H contacts if we place residue i at site
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Count free neighbours (heuristic for avoiding dead-ends)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: gain dominates, then free space, then random noise
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        # Pick the best option
        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _greedy_grow_from(suffix_sequence, start_pos, full_sequence, occupied, rng, temperature=0.0):
    """Greedy growth starting from start_pos, avoiding occupied sites.
    full_sequence is the complete sequence for energy calculation.
    occupied maps position -> index in full_sequence.
    Returns a list of tuples (new positions only, excluding start_pos) or None.
    """
    chain = [start_pos]
    local_occupied = dict(occupied)
    # start_pos is already in occupied with its correct index
    
    for i, res in enumerate(suffix_sequence):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in local_occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = local_occupied.get(q)
                    if j is not None and j < i - 1 and full_sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in local_occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        local_occupied[best_site] = i
        chain.append(best_site)

    return chain[1:]  # exclude the start_pos which is already in prefix


def _local_improve(sequence, chain, rng, max_moves=5000):
    """
    Local search using single-residue moves and regrow moves.
    Returns (best_chain, best_energy).
    """
    n = len(chain)
    if n < 3:
        return list(chain), _compute_energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0

    for move_iter in range(max_moves):
        if time.time() > _DEADLINE - 10:
            break

        # Choose move type: 70% single-residue, 30% regrow
        if rng.random() < 0.7:
            # Single-residue move
            i = rng.randint(1, n - 2)
            p = chain[i]
            
            for np in _neighbours(p):
                if np in occupied and occupied[np] != i:
                    continue
                # Check connectivity: i-1 and i+1 must be adjacent to np
                dist1 = abs(np[0] - chain[i - 1][0]) + abs(np[1] - chain[i - 1][1])
                dist2 = abs(np[0] - chain[i + 1][0]) + abs(np[1] - chain[i + 1][1])
                if dist1 != 1 or dist2 != 1:
                    continue

                # Perform the move
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                new_energy = _compute_energy(sequence, chain)

                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    # Revert
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
                    no_improve_count += 1
                break
        else:
            # Regrow move: pick a split point and regrow the suffix
            split = rng.randint(2, max(2, n - 3))
            prefix = chain[:split]
            suffix_seq = sequence[split:]
            
            # Build occupied set for prefix
            prefix_occupied = {p: i for i, p in enumerate(prefix)}
            
            new_suffix = _greedy_grow_from(
                suffix_seq, prefix[-1], sequence, prefix_occupied, rng, temperature=1.0)
            
            if new_suffix is not None:
                new_chain = prefix + new_suffix
                new_energy = _compute_energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    chain = new_chain
                    current_energy = new_energy
                    occupied = {p: i for i, p in enumerate(chain)}
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        # Early termination if no improvement for many iterations
        if no_improve_count > 500:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of greedy growths with different random seeds
    and temperatures, plus local improvement. Keep the best within the time budget.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)

    # Phase 1: Multiple greedy restarts with varying temperatures
    # Use more restarts for shorter sequences, fewer for longer ones
    num_restart = min(3000, max(500, 15000 // n))
    
    for attempt in range(num_restart):
        if time.time() > _DEADLINE - 15:
            break

        temp = rng.uniform(0.0, 3.0)
        chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local improvement on the best found
    if best_chain is not None:
        improved_chain, improved_energy = _local_improve(
            sequence, best_chain, rng, max_moves=10000)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Phase 3: If we have time, do a second round of improvement on the best
    if best_chain is not None and time.time() < _DEADLINE - 30:
        improved_chain2, improved_energy2 = _local_improve(
            sequence, best_chain, rng, max_moves=5000)
        if improved_energy2 < best_energy:
            best_chain = improved_chain2
            best_energy = improved_energy2

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
