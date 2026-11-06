"""HP lattice protein folding: beam search with local refinement and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _energy_full(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=20):
    """
    Beam search for HP lattice folding.
    State: (chain, occupied_set, energy_so_far)
    We keep the best beam_width partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    # Initial state
    start = (0, 0)
    initial_occupied = {start}
    # Energy contribution of first residue: 0 (no contacts possible yet)
    initial_energy = 0

    # Beam: list of (energy, chain, occupied)
    # We want to keep the best (lowest energy) ones
    beam = [(initial_energy, [start], initial_occupied)]

    for i in range(1, n):
        if _remaining() < 5.0:
            break

        next_beam = []
        residue = sequence[i]

        for energy_so_far, chain, occupied in beam:
            current = chain[-1]
            
            for site in _neighbours(current):
                if site in occupied:
                    continue

                new_occupied = occupied | {site}
                new_chain = chain + [site]

                # Compute marginal energy gain for placing residue i at site
                gain = 0
                if residue == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            # Find the index of q in the chain
                            # Since we have chain, we can search
                            # But this is O(n) per neighbor. For beam search, n is small enough.
                            # Actually, we can maintain a pos_to_idx dict in the state for speed,
                            # but that increases memory. Let's try to be smart.
                            # For now, linear search in chain.
                            for j, p in enumerate(chain):
                                if p == q:
                                    if j < i - 1 and sequence[j] == 'H':
                                        gain += 1
                                    break

                new_energy = energy_so_far - gain

                next_beam.append((new_energy, new_chain, new_occupied))

        if not next_beam:
            break

        # Keep top beam_width
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    if not beam:
        # Fallback: simple growth
        return _grow_fallback(sequence, dimension)

    # Return the best full chain
    best_energy, best_chain, _ = beam[0]
    return best_chain


def _grow_fallback(sequence, dimension):
    """Simple greedy growth as fallback."""
    n = len(sequence)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    rng = random.Random(42)

    for i in range(1, n):
        current = chain[-1]
        options = []

        for site in _neighbours(current):
            if site in occupied:
                continue
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = free * 0.1 + rng.random()
            options.append((score, site))

        if not options:
            return chain

        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied.add(site)
        chain.append(site)

    return chain


def _simulated_anneal(sequence, chain, rng, max_steps=10000):
    """
    Simulated annealing with kink moves and segment moves.
    """
    n = len(chain)
    if n < 4:
        return chain

    current_chain = list(chain)
    current_energy = _energy_full(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    steps = max_steps
    if steps <= 0:
        return best_chain

    for step in range(steps):
        if _remaining() < 3.0:
            break

        # Linear cooling
        T = T_start + (T_end - T_start) * (step / steps)

        # Choose move type
        move_type = rng.random()

        if move_type < 0.5:
            # Kink move: move a single residue (not ends) to a neighbor
            if n < 3:
                continue
            idx = rng.randint(1, n - 2)
            current_pos = current_chain[idx]
            left = current_chain[idx - 1]
            right = current_chain[idx + 1]

            candidates = []
            for site in _neighbours(current_pos):
                if site == left or site == right:
                    continue
                # Check self-avoidance
                # We need to ensure site is not occupied by any other residue
                # Efficient check: remove current_pos, then check if site is in the set
                # For simplicity, we can check linearly for small n, or use a set
                # Let's use a set for efficiency
                occupied = set(current_chain)
                occupied.remove(current_pos)
                if site not in occupied:
                    candidates.append(site)

            if candidates:
                new_pos = rng.choice(candidates)
                new_chain = list(current_chain)
                new_chain[idx] = new_pos
                new_energy = _energy_full(sequence, new_chain)

                delta = new_energy - current_energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    current_chain = new_chain
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)

        else:
            # Segment move: relocate a small segment
            seg_len = rng.randint(2, min(5, n - 2))
            if seg_len >= n:
                continue
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1

            left_neighbor = current_chain[start_idx - 1]
            right_neighbor = current_chain[end_idx + 1]

            # Try to find a new path for the segment
            # This is complex, so we'll do a simple randomized walk
            occupied_by_others = set(current_chain)
            for i in range(start_idx, end_idx + 1):
                occupied_by_others.discard(current_chain[i])

            # Start from a neighbor of left_neighbor
            start_candidates = [p for p in _neighbours(left_neighbor) if p not in occupied_by_others]
            if not start_candidates:
                continue

            rng.shuffle(start_candidates)
            found = False
            for start_pos in start_candidates[:3]:  # Limit attempts
                path = [start_pos]
                visited = {start_pos}
                current = start_pos
                valid = True

                for _ in range(seg_len - 1):
                    neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied_by_others]
                    if not neighbors:
                        valid = False
                        break
                    next_pos = rng.choice(neighbors)
                    path.append(next_pos)
                    visited.add(next_pos)
                    current = next_pos

                if not valid:
                    continue

                # Check if we ended adjacent to right_neighbor
                if abs(current[0] - right_neighbor[0]) + abs(current[1] - right_neighbor[1]) == 1:
                    # Apply move
                    new_chain = list(current_chain)
                    for i, p in enumerate(path):
                        new_chain[start_idx + i] = p

                    new_energy = _energy_full(sequence, new_chain)
                    delta = new_energy - current_energy
                    if delta < 0 or rng.random() < math.exp(-delta / T):
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                    found = True
                    break

    return best_chain


def _polish(sequence, chain, rng):
    """
    Deterministic local polish: try all single-residue moves and accept improvements.
    """
    n = len(chain)
    if n < 3:
        return chain

    improved = True
    while improved:
        improved = False
        for idx in range(1, n - 1):
            if _remaining() < 2.0:
                break
            current_pos = chain[idx]
            left = chain[idx - 1]
            right = chain[idx + 1]
            best_new_pos = None
            best_new_energy = _energy_full(sequence, chain)

            occupied = set(chain)
            occupied.remove(current_pos)

            for site in _neighbours(current_pos):
                if site == left or site == right:
                    continue
                if site in occupied:
                    continue

                new_chain = list(chain)
                new_chain[idx] = site
                new_energy = _energy_full(sequence, new_chain)

                if new_energy < best_new_energy:
                    best_new_energy = new_energy
                    best_new_pos = site

            if best_new_pos is not None:
                chain[idx] = best_new_pos
                improved = True

    return chain


def fold(sequence, dimension):
    _remaining()  # Ensure time tracking is active

    n = len(sequence)
    if n == 0:
        return []

    if _remaining() < 1.0:
        return [[i, 0] for i in range(n)]

    rng = random.Random(12345 + len(sequence))

    # Phase 1: Beam search
    chain = _beam_search(sequence, dimension, beam_width=15)

    # If beam search failed or returned incomplete, use fallback
    if not chain or len(chain) != n:
        chain = _grow_fallback(sequence, dimension)

    # Phase 2: Simulated annealing refinement
    if _remaining() > 10.0:
        time_left = _remaining()
        # Estimate ~1ms per SA step
        max_steps = int(time_left * 1000 * 0.8)
        max_steps = min(max_steps, 15000)
        chain = _simulated_anneal(sequence, chain, rng, max_steps=max_steps)

    # Phase 3: Polish
    if _remaining() > 3.0:
        chain = _polish(sequence, chain, rng)

    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
