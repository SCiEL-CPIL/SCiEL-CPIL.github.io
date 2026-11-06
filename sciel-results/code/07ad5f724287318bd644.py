"""HP lattice protein folding: constraint-satisfaction growth with pivot/slit local search."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 195.0  # seconds, safety margin before 210s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute energy: minus number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    pos_map = {}
    for i, p in enumerate(chain):
        pos_map[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = pos_map.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _contact_map(sequence):
    """Build a set of (i, j) pairs where both are H and i < j, |i-j| >= 2."""
    n = len(sequence)
    h_indices = [i for i in range(n) if sequence[i] == 'H']
    pairs = set()
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if abs(i - j) >= 2:
                pairs.add((i, j))
    return pairs


def _grow_constrained(sequence, dimension, rng, contact_pairs):
    """
    Grow the chain with constraint satisfaction: prioritize placements that
    satisfy more potential H-H contacts from the contact map.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]

    # Precompute for each position i, which other positions j form a contact pair
    contacts_by_pos = defaultdict(list)
    for i, j in contact_pairs:
        contacts_by_pos[i].append(j)
        contacts_by_pos[j].append(i)

    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        best_score = -float('inf')
        best_site = None
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            score = 0.0
            if sequence[i] == 'H':
                # Count how many potential contacts this placement satisfies
                for j in contacts_by_pos[i]:
                    if j < i:  # only consider already-placed residues
                        jpos = chain[j]
                        # Check if site is adjacent to jpos
                        if abs(site[0] - jpos[0]) + abs(site[1] - jpos[1]) == 1:
                            score += 10.0  # high reward for satisfying a contact
            # Small bonus for free space
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free + rng.random() * 0.5
            options.append((score, site))

        if not options:
            return None

        # Pick the best option with some randomness among top choices
        options.sort(key=lambda x: -x[0])
        top_k = min(3, len(options))
        if rng.random() < 0.7:
            best_site = options[0][1]
        else:
            best_site = rng.choice([opt[1] for opt in options[:top_k]])

        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _pivot_move_fast(chain, pivot, new_dir, pos_map):
    """
    Try a pivot move: rotate the segment after pivot by new_dir.
    Returns (new_chain, new_pos_map) if valid, else (None, None).
    """
    n = len(chain)
    if pivot < 1 or pivot >= n - 1:
        return None, None

    # The segment to rotate is chain[pivot+1:]
    segment = chain[pivot+1:]
    if not segment:
        return None, None

    # Current direction of the first bond in the segment
    prev = chain[pivot]
    first = chain[pivot+1]
    cur_dir = (first[0] - prev[0], first[1] - prev[1])

    # Determine rotation: we want to change cur_dir to new_dir
    # In 2D, there are 3 possible directions other than the reverse of cur_dir
    # We'll just try to rotate the segment so that the first bond points in new_dir

    # Compute rotation vector
    # If cur_dir == (1,0), new_dir could be (0,1) or (0,-1) (90 degree rotations)
    # We need to apply the same rotation to all subsequent bonds

    # Find the rotation that maps cur_dir to new_dir
    # In 2D, this is a 90-degree rotation (clockwise or counter-clockwise)
    if cur_dir == (1, 0):
        if new_dir == (0, 1):
            rot = lambda p: (p[1], -p[0])  # 90 deg clockwise
        elif new_dir == (0, -1):
            rot = lambda p: (-p[1], p[0])  # 90 deg counter-clockwise
        else:
            return None, None
    elif cur_dir == (-1, 0):
        if new_dir == (0, 1):
            rot = lambda p: (-p[1], p[0])
        elif new_dir == (0, -1):
            rot = lambda p: (p[1], -p[0])
        else:
            return None, None
    elif cur_dir == (0, 1):
        if new_dir == (1, 0):
            rot = lambda p: (p[1], -p[0])
        elif new_dir == (-1, 0):
            rot = lambda p: (-p[1], p[0])
        else:
            return None, None
    elif cur_dir == (0, -1):
        if new_dir == (1, 0):
            rot = lambda p: (-p[1], p[0])
        elif new_dir == (-1, 0):
            rot = lambda p: (p[1], -p[0])
        else:
            return None, None
    else:
        return None, None

    # Apply rotation to the segment
    # The segment is a chain of bonds; we need to rotate each bond
    new_segment = [prev]
    current = prev
    for k in range(len(segment) - 1):
        bond = (segment[k+1][0] - segment[k][0], segment[k+1][1] - segment[k][1])
        rotated_bond = rot(bond)
        new_pos = (current[0] + rotated_bond[0], current[1] + rotated_bond[1])
        # Check if new_pos is already occupied (excluding the segment itself)
        if new_pos in pos_map:
            idx = pos_map[new_pos]
            if idx <= pivot:  # if it's in the fixed part, it's a collision
                return None, None
        new_segment.append(new_pos)
        current = new_pos

    # Check self-avoidance within the new segment
    new_segment_set = set(new_segment)
    if len(new_segment_set) != len(new_segment):
        return None, None

    # Build new chain
    new_chain = chain[:pivot+1] + new_segment[1:]

    # Verify no collisions with the fixed part
    new_pos_map = {}
    for i, p in enumerate(new_chain):
        if p in new_pos_map:
            return None, None
        new_pos_map[p] = i

    return new_chain, new_pos_map


def _slit_move(chain, pos, new_dir, pos_map):
    """
    Try a slit move: move residue at pos to a new position in the direction new_dir,
    shifting the segment from pos to the end.
    """
    n = len(chain)
    if pos < 1 or pos >= n - 1:
        return None, None

    prev = chain[pos - 1]
    # Check that new_dir is valid (adjacent to prev)
    if abs(new_dir[0]) + abs(new_dir[1]) != 1:
        return None, None

    new_pos = (prev[0] + new_dir[0], prev[1] + new_dir[1])
    if new_pos == chain[pos]:
        return None, None

    # Check if new_pos is occupied by a residue before pos
    if new_pos in pos_map:
        if pos_map[new_pos] < pos:
            return None, None

    # The segment from pos to n-1 needs to be shifted
    # The displacement is (new_pos - chain[pos])
    disp = (new_pos[0] - chain[pos][0], new_pos[1] - chain[pos][1])

    # Check if the entire shifted segment avoids collisions
    new_chain = list(chain)
    new_chain[pos] = new_pos
    for k in range(pos + 1, n):
        new_chain[k] = (chain[k][0] + disp[0], chain[k][1] + disp[1])

    # Verify self-avoidance
    new_pos_map = {}
    for i, p in enumerate(new_chain):
        if p in new_pos_map:
            return None, None
        new_pos_map[p] = i

    return new_chain, new_pos_map


def _local_search(sequence, chain, max_steps=200):
    """
    Local search using pivot and slit moves to escape local minima.
    """
    n = len(chain)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    pos_map = {}
    for i, p in enumerate(best_chain):
        pos_map[p] = i

    steps = 0
    while steps < max_steps and _time_left() > 5.0:
        improved = False

        # Try pivot moves
        for pivot in range(1, n - 1):
            if _time_left() < 5.0:
                break
            prev = best_chain[pivot]
            for new_dir in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                # Skip the direction that goes back to the previous residue
                if pivot > 0:
                    back_dir = (best_chain[pivot-1][0] - prev[0], best_chain[pivot-1][1] - prev[1])
                    if new_dir == back_dir:
                        continue
                new_chain, new_pos_map = _pivot_move_fast(best_chain, pivot, new_dir, pos_map)
                if new_chain is not None:
                    e = _energy(sequence, new_chain)
                    if e < best_energy:
                        best_chain = new_chain
                        best_energy = e
                        pos_map = new_pos_map
                        improved = True
                        steps += 1
                        break
            if improved:
                break

        if improved:
            continue

        # Try slit moves
        for pos in range(1, n - 1):
            if _time_left() < 5.0:
                break
            prev = best_chain[pos - 1]
            for new_dir in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_chain, new_pos_map = _slit_move(best_chain, pos, new_dir, pos_map)
                if new_chain is not None:
                    e = _energy(sequence, new_chain)
                    if e < best_energy:
                        best_chain = new_chain
                        best_energy = e
                        pos_map = new_pos_map
                        improved = True
                        steps += 1
                        break
            if improved:
                break

        if not improved:
            break

    return best_chain


def _polish(sequence, chain, max_steps=100):
    """
    Final polish: try simple single-residue moves to squeeze out last contacts.
    """
    n = len(chain)
    if n < 3:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    pos_map = {}
    for i, p in enumerate(best_chain):
        pos_map[p] = i

    steps = 0
    while steps < max_steps and _time_left() > 5.0:
        improved = False
        for pos in range(1, n - 1):
            if _time_left() < 5.0:
                break
            prev = best_chain[pos - 1]
            nxt = best_chain[pos + 1]
            for site in _neighbours(best_chain[pos]):
                if site == prev or site == nxt:
                    continue
                if site in pos_map:
                    continue
                new_chain = list(best_chain)
                new_chain[pos] = site
                # Quick validity check
                valid = True
                for i, p in enumerate(new_chain):
                    if i > 0:
                        if abs(p[0] - new_chain[i-1][0]) + abs(p[1] - new_chain[i-1][1]) != 1:
                            valid = False
                            break
                if not valid:
                    continue
                e = _energy(sequence, new_chain)
                if e < best_energy:
                    best_chain = new_chain
                    best_energy = e
                    pos_map = {}
                    for i, p in enumerate(best_chain):
                        pos_map[p] = i
                    improved = True
                    steps += 1
                    break
            if improved:
                break
        if not improved:
            break

    return best_chain


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    Uses constraint-satisfaction growth, pivot/slit local search, and polish.
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]

    rng = random.Random(42 + len(sequence))
    contact_pairs = _contact_map(sequence)

    best_chain = None
    best_energy = 0

    # Strategy 1: Multiple constraint-satisfaction growths
    num_grow = 80
    for _ in range(num_grow):
        if _time_left() < 15.0:
            break
        chain = _grow_constrained(sequence, dimension, rng, contact_pairs)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = chain

    # Strategy 2: Local search with pivot/slit moves on best chain
    if best_chain is not None:
        if _time_left() > 20.0:
            polished = _local_search(sequence, best_chain, max_steps=150)
            e = _energy(sequence, polished)
            if e < best_energy:
                best_energy = e
                best_chain = polished

    # Strategy 3: Final polish
    if best_chain is not None:
        if _time_left() > 10.0:
            polished = _polish(sequence, best_chain, max_steps=80)
            e = _energy(sequence, polished)
            if e < best_energy:
                best_energy = e
                best_chain = polished

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
