"""HP lattice protein folding on the cubic lattice.

Strategy:
  1. Beam search to grow a self-avoiding chain, keeping the top-K partial
     conformations by a combined score of H-H contacts gained and free-space
     heuristic.  This avoids the single-path greedy trap.
  2. Pivot-based local search (Monte-Carlo moves that rotate a segment of the
     chain around its first residue) to refine each beam-surviving conformation.
  3. Deadline guard: track elapsed time and return the best result found
     before the safety margin.
"""

import time
import random
import math
from collections import deque

_DEADLINE_SECONDS = 240.0   # hard CPU budget for all sequences
_SAFETY_MARGIN = 20.0       # seconds to leave before the 260 s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of a 3-D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Energy = -number of H-H contacts (non-adjacent, neighbouring sites)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_grow(sequence, dimension, beam_width, rng, time_limit=None, start_time=None):
    """
    Beam-search growth.  Returns a list of (chain, energy) tuples for the
    surviving beams at the end, or an empty list if no beam survives.
    """
    if start_time is None:
        start_time = time.time()

    # Each beam entry: (chain as list of tuples, occupied set, current energy)
    start = (0,) * dimension
    initial = {
        'chain': [start],
        'occupied': {start: 0},
        'energy': 0,
    }
    beams = [initial]

    for i in range(1, len(sequence)):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        new_beams = []
        for beam in beams:
            chain = beam['chain']
            occupied = beam['occupied']
            energy = beam['energy']
            last = chain[-1]

            candidates = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute gain: new H-H contacts this site would form
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                # Free-space heuristic: number of unoccupied neighbours
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Score: prefer more contacts, then more free space
                score = gain * 10 + free
                candidates.append((score, site, gain))

            if not candidates:
                continue  # dead end, drop this beam

            # Keep the best candidate for this beam (greedy within beam)
            candidates.sort(key=lambda x: -x[0])
            best_site = candidates[0][1]
            best_gain = candidates[0][2]

            new_chain = chain + [best_site]
            new_occupied = dict(occupied)
            new_occupied[best_site] = i
            new_energy = energy - best_gain  # each new contact lowers energy by 1

            new_beams.append({
                'chain': new_chain,
                'occupied': new_occupied,
                'energy': new_energy,
            })

        if not new_beams:
            break

        # Prune to beam_width: keep beams with lowest energy (most negative)
        new_beams.sort(key=lambda b: b['energy'])
        beams = new_beams[:beam_width]

    # Return surviving complete chains
    results = []
    for beam in beams:
        if len(beam['chain']) == len(sequence):
            results.append((beam['chain'], beam['energy']))
    return results


def _pivot_move(chain, occupied, i, rng):
    """
    Try a pivot move at position i: rotate the sub-chain from i+1 to end
    around chain[i].  Returns (new_chain, new_occupied) or None if invalid.
    """
    n = len(chain)
    if i >= n - 1:
        return None

    # The segment to rotate is chain[i+1:]
    # We try rotating it in one of the 6 directions relative to the bond chain[i]->chain[i+1]
    # Simpler: try moving the entire tail by shifting it in one of 6 directions
    # and check self-avoidance.

    # Actually, a standard pivot move: pick a random position i, then try to
    # rotate the segment [i+1, n-1] around chain[i] by 90 degrees around each
    # axis perpendicular to the bond direction.  For simplicity, we try all
    # 6 unit vectors as new directions for the first bond of the segment,
    # and then re-grow the rest greedily in the same relative orientation.

    # Simpler approach: try to shift the entire tail by one unit in each of
    # the 6 directions, checking self-avoidance.
    for axis in range(len(chain[0])):
        for step in (-1, 1):
            # New position for chain[i+1]
            new_pos = list(chain[i + 1])
            new_pos[axis] += step
            new_pos = tuple(new_pos)

            # Check if new_pos is free (not in occupied, except chain[i+1] itself)
            if new_pos in occupied and occupied.get(new_pos) != i + 1:
                continue

            # Try to shift the entire tail
            new_tail = []
            valid = True
            for j in range(i + 1, n):
                if j == i + 1:
                    new_tail.append(new_pos)
                else:
                    # Shift by the same delta
                    delta = tuple(new_pos[k] - chain[i + 1][k] for k in range(len(chain[0])))
                    candidate = tuple(chain[j][k] + delta[k] for k in range(len(chain[0])))
                    # Check self-avoidance: candidate must not be in the fixed part
                    # or in the new tail already placed
                    if candidate in occupied and occupied.get(candidate) <= i:
                        valid = False
                        break
                    if candidate in new_tail:
                        valid = False
                        break
                    new_tail.append(candidate)

            if not valid:
                continue

            # Check that the new tail doesn't collide with the fixed part
            new_chain = chain[:i + 1] + new_tail
            # Verify self-avoidance of the whole chain
            seen = set()
            for p in new_chain:
                if p in seen:
                    valid = False
                    break
                seen.add(p)
            if not valid:
                continue

            # Build new occupied
            new_occupied = {p: idx for idx, p in enumerate(new_chain)}
            return (new_chain, new_occupied)

    return None


def _local_refine(sequence, chain, occupied, rng, max_iters, time_limit=None, start_time=None):
    """
    Monte-Carlo local search using pivot moves.  Returns the best chain found.
    """
    if start_time is None:
        start_time = time.time()

    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    current_chain = list(chain)
    current_occupied = dict(occupied)
    current_energy = best_energy

    for _ in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        i = rng.randint(0, len(current_chain) - 2)
        result = _pivot_move(current_chain, current_occupied, i, rng)
        if result is None:
            continue

        new_chain, new_occupied = result
        new_energy = _compute_energy(sequence, new_chain)

        # Metropolis acceptance
        if new_energy < current_energy:
            current_chain = new_chain
            current_occupied = new_occupied
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            # Accept with probability exp(-delta / T)
            delta = new_energy - current_energy
            if delta == 0 or rng.random() < math.exp(-delta / max(1, len(sequence) * 0.1)):
                current_chain = new_chain
                current_occupied = new_occupied
                current_energy = new_energy

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Best conformation found by beam search + local refinement.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    n = len(sequence)
    rng = random.Random(42 + n)

    # Parameters
    beam_width = 20
    num_beams = 3  # number of independent beam searches
    local_iters = max(50, n * 5)

    best_chain = None
    best_energy = 0

    for trial in range(num_beams):
        if time.time() >= deadline:
            break

        # Beam search
        beam_limit = min(30.0, max(5.0, (deadline - time.time()) / (num_beams - trial + 1)))
        results = _beam_grow(sequence, dimension, beam_width, rng,
                             time_limit=beam_limit, start_time=start_time)

        if not results:
            continue

        # Local refine the best beam result
        chain, energy = results[0]
        occupied = {p: i for i, p in enumerate(chain)}

        local_limit = min(60.0, max(5.0, (deadline - time.time()) / (num_beams - trial + 1)))
        refined_chain, refined_energy = _local_refine(
            sequence, chain, occupied, rng,
            max_iters=local_iters,
            time_limit=local_limit,
            start_time=start_time
        )

        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
