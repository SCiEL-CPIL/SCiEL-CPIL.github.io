"""HP lattice protein folding: beam search + efficient local search + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy of a conformation efficiently."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _partial_score(sequence, chain, occupied):
    """Score a partial chain for beam search: contacts + free space heuristic."""
    i = len(chain) - 1
    score = 0
    if sequence[i] == "H":
        for q in _neighbours(chain[i]):
            j = occupied.get(q)
            if j is not None and j < i - 1 and sequence[j] == "H":
                score += 1
    # Add small bonus for free space to encourage compactness
    free = sum(1 for q in _neighbours(chain[i]) if q not in occupied)
    score += 0.05 * free
    return score


def _beam_search(sequence, dimension, beam_width=200, max_depth=None):
    """Beam search over self-avoiding walks with contact-aware scoring."""
    n = len(sequence)
    if max_depth is None:
        max_depth = n

    start = (0,) * dimension
    # Each beam entry: (negative_score, chain, occupied_dict)
    beam = [(-_partial_score(sequence, [start], {start: 0}), [start], {start: 0})]

    for depth in range(1, max_depth):
        if _remaining_time() < 1.0:
            break
        new_beam = []
        for neg_score, chain, occupied in beam:
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = depth
                score = _partial_score(sequence, new_chain, new_occupied)
                new_beam.append((-score, new_chain, new_occupied))

        if not new_beam:
            break

        # Keep top beam_width entries
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    if not beam:
        return None

    # Return the best complete chain
    best = min(beam, key=lambda x: _energy(sequence, x[1]))
    return best[1]


def _apply_reversal(chain, rng):
    """Apply a reversal move: reverse a segment of the chain."""
    n = len(chain)
    if n < 4:
        return None

    # Choose two positions i < j to reverse chain[i:j+1]
    i = rng.randint(0, n - 4)
    j = rng.randint(i + 2, n - 1)

    # The segment to reverse is chain[i+1:j+1]
    # After reversal, we need to check self-avoidance
    # The head is chain[:i+1], the tail is chain[j+1:]
    # The reversed segment is chain[j], chain[j-1], ..., chain[i+1]

    head = chain[:i+1]
    segment = chain[i+1:j+1]
    tail = chain[j+1:]

    # Reverse the segment
    reversed_segment = segment[::-1]

    # Check if the connection points are valid
    # head[-1] should connect to reversed_segment[0]
    # reversed_segment[-1] should connect to tail[0] (if tail exists)

    if abs(head[-1][0] - reversed_segment[0][0]) + abs(head[-1][1] - reversed_segment[0][1]) != 1:
        return None

    if tail and abs(reversed_segment[-1][0] - tail[0][0]) + abs(reversed_segment[-1][1] - tail[0][1]) != 1:
        return None

    # Check self-avoidance
    new_chain = head + reversed_segment + tail
    if len(set(new_chain)) != len(new_chain):
        return None

    return new_chain


def _apply_pivot(chain, rng):
    """Apply a pivot move: rotate a tail segment around a pivot point."""
    n = len(chain)
    if n < 4:
        return None

    pivot = rng.randint(1, n - 3)

    # The tail is chain[pivot+1:]
    # We rotate the tail by 90, 180, or 270 degrees around the pivot
    # In 2D: 90 deg: (x,y) -> (-y,x), 180: (x,y) -> (-x,-y), 270: (x,y) -> (y,-x)

    dx = chain[pivot+1][0] - chain[pivot][0]
    dy = chain[pivot+1][1] - chain[pivot][1]

    # Try 3 rotations
    rotations = [
        (-dy, dx),   # 90 deg
        (-dx, -dy),  # 180 deg
        (dy, -dx),   # 270 deg
    ]

    for rot_dx, rot_dy in rotations:
        # The new first step direction
        # Check if this direction is different from the original
        if rot_dx == dx and rot_dy == dy:
            continue

        # Build the new tail by applying the rotation to all points relative to pivot
        new_tail = []
        for k in range(len(chain) - pivot - 1):
            px = chain[pivot+k+1][0] - chain[pivot][0]
            py = chain[pivot+k+1][1] - chain[pivot][1]
            # Apply rotation
            if (rot_dx, rot_dy) == (-dy, dx):
                # 90 deg: (px,py) -> (-py,px)
                nx = -py
                ny = px
            elif (rot_dx, rot_dy) == (-dx, -dy):
                # 180 deg: (px,py) -> (-px,-py)
                nx = -px
                ny = -py
            else:
                # 270 deg: (px,py) -> (py,-px)
                nx = py
                ny = -px
            new_tail.append((chain[pivot][0] + nx, chain[pivot][1] + ny))

        new_chain = chain[:pivot+1] + new_tail

        # Check self-avoidance
        if len(set(new_chain)) != len(new_chain):
            continue

        return new_chain

    return None


def _local_move(chain, rng):
    """Perform a random local move: reversal or pivot."""
    if rng.random() < 0.5:
        result = _apply_reversal(chain, rng)
    else:
        result = _apply_pivot(chain, rng)

    if result is None:
        # Try the other move type
        if rng.random() < 0.5:
            result = _apply_pivot(chain, rng)
        else:
            result = _apply_reversal(chain, rng)

    return result


def _anneal(sequence, chain, max_steps, rng):
    """Simulated annealing with reversal and pivot moves."""
    n = len(chain)
    if n < 3:
        return chain

    current_energy = _energy(sequence, chain)
    best_chain = chain
    best_energy = current_energy

    T_start = 3.0
    T_end = 0.05
    cooling = (T_end / T_start) ** (1.0 / max_steps)

    for step in range(max_steps):
        if _remaining_time() < 0.5:
            break

        T = T_start * (cooling ** step)

        new_chain = _local_move(chain, rng)
        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta <= 0 or rng.random() < exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain

    return best_chain


def _local_search_polish(sequence, chain, max_iter=5000):
    """Greedy local search: only accept improving moves. More intensive than SA."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    rng = random.Random(42)

    for _ in range(max_iter):
        if _remaining_time() < 1.0:
            break
        improved = False
        # Try short reversals and pivots
        for _ in range(15):
            if rng.random() < 0.6:
                new_chain = _apply_reversal(chain, rng)
            else:
                new_chain = _apply_pivot(chain, rng)
            if new_chain is None:
                continue
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                improved = True
                break
        if not improved:
            break
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold using beam search + simulated annealing + local search polish."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    # Phase 1: Beam search to get good initial solutions
    best_chain = None
    best_energy = float('inf')

    # Try multiple beam widths
    for bw in [100, 300, 500]:
        if _remaining_time() < 2.0:
            break
        chain = _beam_search(sequence, dimension, beam_width=bw)
        if chain is not None:
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain

    if best_chain is None:
        # Fallback: simple greedy growth
        start = (0,) * dimension
        chain, occupied = [start], {start: 0}
        for i in range(1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                options.append(site)
            if not options:
                break
            site = options[rng.randint(0, len(options) - 1)]
            occupied[site] = i
            chain.append(site)
        best_chain = chain
        best_energy = _energy(sequence, chain)

    # Phase 2: Simulated annealing to improve
    if _remaining_time() > 3.0:
        time_budget = min(_remaining_time() - 3.0, 8.0)
        max_steps = int(time_budget * 1000)  # ~1000 steps per second
        best_chain = _anneal(sequence, best_chain, max_steps, rng)

    # Phase 3: Local search polish
    if _remaining_time() > 1.0:
        polish_budget = min(_remaining_time() - 1.0, 2.0)
        max_iter = int(polish_budget * 5000)
        best_chain, best_energy = _local_search_polish(sequence, best_chain, max_iter)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
