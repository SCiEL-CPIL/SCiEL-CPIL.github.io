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


def _beam_search(sequence, dimension, beam_width=200):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    # Beam: list of (energy_so_far, chain_tuple, occupied_set)
    # We use tuples for chains to make them hashable for deduplication
    beam = [(0, ((0, 0, 0),), {(0, 0, 0): 0})]

    for i in range(1, n):
        if _time_left() < 10.0:
            break

        new_beam = []
        seen = set()

        for energy_so_far, chain_tuple, occupied in beam:
            last = chain_tuple[-1]
            seq_i = sequence[i]

            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                site = (last[0] + dx, last[1] + dy, last[2] + dz)
                if site in occupied:
                    continue

                # Compute new contacts gained
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _NEIGHBOR_OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # New energy is more negative when gain is higher
                new_energy = energy_so_far - gain

                # Deduplicate: use the last position and a hash of the chain shape
                # For efficiency, we'll use a tuple of the last few positions as a proxy
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


def _local_search(sequence, dimension, chain, rng, max_iterations=500):
    """
    Local search using segment reversal to improve energy.
    """
    n = len(chain)
    if n < 5:
        return chain

    best_chain = list(chain)
    best_energy = _energy_correct(sequence, best_chain)
    occupied_set = set(best_chain)

    for iteration in range(max_iterations):
        if _time_left() < 5.0:
            break

        # Try segment reversal
        seg_len = rng.randint(2, min(8, n - 2))
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1

        # Extract segment
        segment = best_chain[start_idx:end_idx + 1]
        reversed_segment = segment[::-1]

        # Check if reversal causes self-avoidance violations
        # Only the boundaries and internal positions need checking
        new_chain = best_chain[:]
        new_chain[start_idx:end_idx + 1] = reversed_segment

        # Quick check: verify no duplicates in the modified region
        # and that unit steps are maintained
        valid = True
        # Check self-avoidance
        if len(set(new_chain)) != n:
            valid = False
        else:
            # Check unit steps around the modified region
            for k in range(max(0, start_idx - 1), min(n - 1, end_idx + 1)):
                diff = sum(abs(a - b) for a, b in zip(new_chain[k], new_chain[k + 1]))
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

    best_chain = None
    best_energy = 0

    # Time budget for this sequence
    seq_start = time.time()
    time_budget = min(120.0, max(15.0, n * 0.8))

    # Phase 1: Beam search with different widths
    # Use multiple beam searches with varying widths for diversity
    beam_configs = [
        (100, 12345),
        (200, 54321),
        (300, 98765),
        (150, 11111),
        (250, 22222),
    ]

    for beam_width, seed in beam_configs:
        if time.time() - seq_start > time_budget * 0.6:
            break
        if _time_left() < 15.0:
            break

        chain = _beam_search(sequence, dimension, beam_width=beam_width)
        if chain is None or len(chain) != n:
            continue

        energy = _energy_correct(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Local search on the best chain
    if best_chain is not None and _time_left() > 10.0:
        rng = random.Random(42)
        improved = _local_search(sequence, dimension, best_chain, rng, max_iterations=300)
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
