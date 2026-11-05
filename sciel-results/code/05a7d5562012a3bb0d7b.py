"""HP lattice protein folding via beam search with time management."""
import time
import random

# Global deadline management
_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 320.0  # safety margin before 330s kill

def _time_left():
    return _WALL_CLOCK_LIMIT - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy: -number of H-H non-adjacent contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _beam_fold(sequence, dimension, beam_width, rng, time_budget):
    """
    Beam search: maintain K diverse partial conformations.
    Each beam entry is (chain, occupied_set, current_energy).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    # Initial beams: try all 6 directions from origin
    initial_beams = []
    for axis in range(dimension):
        for step in (-1, 1):
            pos = list(start)
            pos[axis] += step
            pos = tuple(pos)
            initial_beams.append((
                [start, pos],
                {start: 0, pos: 1},
                0  # no contacts possible with 2 residues
            ))

    # If sequence is length 1, just return start
    if n == 1:
        return [start]

    beams = initial_beams

    for i in range(2, n):
        if _time_left() < time_budget:
            break

        new_beams = []
        seen_positions = set()

        for chain, occupied, cur_energy in beams:
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute gain in energy (more negative = better)
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == "H":
                            gain -= 1  # each contact adds -1 to energy

                # Diversity score: prefer sites further from existing chain
                # to encourage exploration
                min_dist = float('inf')
                for p in chain:
                    d = sum(abs(a - b) for a, b in zip(site, p))
                    if d < min_dist:
                        min_dist = d
                diversity = min_dist * 0.01  # small bonus for diversity

                score = cur_energy + gain - diversity
                options.append((score, site, gain))

            if not options:
                continue

            # Sort by score (lower is better) and take top few
            options.sort(key=lambda x: x[0])
            for score, site, gain in options[:max(3, beam_width // 5)]:
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = cur_energy + gain
                # Use a hash to avoid duplicate beams
                key = tuple(new_chain)
                if key not in seen_positions:
                    seen_positions.add(key)
                    new_beams.append((new_chain, new_occupied, new_energy))

        if not new_beams:
            break

        # Keep only the best beam_width beams
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]

    # Pick the best complete chain
    if not beams:
        # Fallback: straight line
        return [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    best_chain = min(beams, key=lambda x: x[2])
    return best_chain[0]


def _local_refine(sequence, chain, dimension, rng, max_iter=500):
    """
    Local refinement via pivot moves and 2-opt-like reversals.
    Try to improve energy through local moves.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)

    for it in range(max_iter):
        if _time_left() < 5.0:
            break

        improved = False
        # Try pivot moves: rotate a subchain around a pivot point
        for pivot in range(1, n - 2):
            # Get the subchain after pivot
            sub = best_chain[pivot:]
            # Try rotating the subchain in 3D
            # Simple approach: try swapping two non-adjacent positions
            for a in range(pivot, n - 1):
                for b in range(a + 1, n):
                    # Check if swap is valid (no self-intersection)
                    new_chain = best_chain[:]
                    new_chain[a], new_chain[b] = new_chain[b], new_chain[a]

                    # Check self-avoidance
                    if len(set(new_chain)) != n:
                        continue

                    # Check unit steps
                    valid = True
                    for k in range(n - 1):
                        d = sum(abs(new_chain[k][d2] - new_chain[k+1][d2]) for d2 in range(dimension))
                        if d != 1:
                            valid = False
                            break
                    if not valid:
                        continue

                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # Try random perturbation: move one residue to a random neighbor
            pos = rng.randint(1, n - 2)
            old_pos = best_chain[pos]
            for new_pos in _neighbours(old_pos):
                if new_pos in best_chain:
                    continue
                new_chain = best_chain[:]
                new_chain[pos] = new_pos

                # Check self-avoidance
                if len(set(new_chain)) != n:
                    continue

                # Check unit steps
                valid = True
                for k in range(n - 1):
                    d = sum(abs(new_chain[k][d2] - new_chain[k+1][d2]) for d2 in range(dimension))
                    if d != 1:
                        valid = False
                        break
                if not valid:
                    continue

                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    best_chain = new_chain
                    best_energy = new_energy
                    improved = True
                    break

        if not improved and it > 50:
            break

    return best_chain


def fold(sequence, dimension):
    """
    Best conformation using beam search with local refinement.
    Returns list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]

    rng = random.Random(12345 + len(sequence))

    # Adaptive beam width based on sequence length
    if n <= 20:
        beam_width = 50
    elif n <= 50:
        beam_width = 30
    elif n <= 100:
        beam_width = 20
    else:
        beam_width = 15

    best_chain = None
    best_energy = 0  # We want most negative

    # Multiple beam search attempts with different randomizations
    max_attempts = 10
    for attempt in range(max_attempts):
        if _time_left() < 10.0:
            break

        # Vary beam width slightly for diversity
        bw = max(10, beam_width + rng.randint(-5, 5))

        chain = _beam_fold(sequence, dimension, bw, rng, time_budget=2.0)
        if chain is None or len(chain) != n:
            # Fallback
            chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

        energy = _energy(sequence, chain)

        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

        # Local refinement on the best chain found so far
        if _time_left() > 5.0:
            refined = _local_refine(sequence, best_chain, dimension, rng, max_iter=200)
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy

    # Final check: ensure self-avoidance and valid steps
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # Verify and fix if needed
    if len(set(best_chain)) != n:
        # Self-intersection detected, use straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
