"""
HP lattice protein folding on the square lattice.
Strategy: Contact-map guided growth with pivot/slither local search and
simulated annealing. Combines constraint-aware growth (from inspiration)
with efficient incremental energy tracking and multiple restarts.
"""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin before 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Return the 4 neighbours of a 2D lattice point."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None or len(chain) != len(sequence):
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for nb in _neighbours(p):
                j = where.get(nb)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _validate_chain(sequence, chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n != len(sequence):
        return False
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            d = abs(p[0] - chain[i-1][0]) + abs(p[1] - chain[i-1][1])
            if d != 1:
                return False
    return True


def _contact_pairs(sequence):
    """Pre-compute weighted H-H contact pairs for guidance."""
    n = len(sequence)
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    pairs = []
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            dist = j - i
            if dist > 1:
                weight = max(0, 1.0 - dist / (n * 0.5))
                pairs.append((i, j, weight))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def _grow_guided(sequence, rng, contact_pairs, bias=1.0, temperature=0.5):
    """
    Grow chain guided by contact map, compactness, and free space.
    Returns list of (x,y) tuples or None if stuck.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    satisfied_pairs = set()

    # Precompute for speed
    seq_h = [c == 'H' for c in sequence]

    for i in range(1, n):
        options = []
        last = chain[-1]

        for site in _neighbours(last):
            if site in occupied:
                continue

            score = 0.0

            if seq_h[i]:
                # Count new H-H contacts formed
                gain = 0
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq_h[j]:
                        gain += 1
                        pair_key = (min(i, j), max(i, j))
                        if pair_key not in satisfied_pairs:
                            satisfied_pairs.add(pair_key)
                            # Bonus for high-priority pairs
                            for (pi, pj, w) in contact_pairs[:30]:
                                if (pi, pj) == pair_key:
                                    score += w * 3.0
                                    break
                score += bias * gain

            # Compactness: prefer sites closer to centroid
            if i > 5:
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = abs(site[0] - cx) + abs(site[1] - cy)
                score -= 0.25 * dist

            # Free space to avoid dead ends
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.1 * free

            # Random noise for diversity
            score += rng.random() * temperature

            options.append((score, site))

        if not options:
            return None

        # Select with softmax-like probability
        options.sort(key=lambda x: -x[0])
        top_score = options[0][0]
        k = min(4, len(options))
        probs = []
        total = 0.0
        for sc, _ in options[:k]:
            p = math.exp((sc - top_score) / max(temperature, 0.1))
            probs.append(p)
            total += p
        if total > 0:
            probs = [p / total for p in probs]
            r = rng.random()
            cum = 0.0
            chosen = k - 1
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen = idx
                    break
            site = options[chosen][1]
        else:
            site = options[0][1]

        occupied[site] = i
        chain.append(site)

    return chain


def _grow_random(sequence, rng, bias=0.5):
    """Simpler random growth for diversity."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    seq_h = [c == 'H' for c in sequence]

    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            score = 0.0
            if seq_h[i]:
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and seq_h[occupied[q]])
                score += bias * gain
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            score += rng.random()
            options.append((score, site))

        if not options:
            return None

        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)

    return chain


def _local_search_sa(sequence, chain, rng, max_iter=200, T_start=3.0, T_end=0.1):
    """
    Simulated annealing local search with pivot, slither, and segment moves.
    Uses incremental energy tracking for speed.
    """
    if chain is None:
        return None, 0

    n = len(chain)
    current = list(chain)
    pos_map = {p: i for i, p in enumerate(current)}
    seq_h = [c == 'H' for c in sequence]

    # Compute initial energy and contact set
    current_energy = _energy(sequence, current)
    best_chain = list(current)
    best_energy = current_energy

    # Track contacts as a set of frozenset({i,j})
    contact_set = set()
    for i, p in enumerate(current):
        if seq_h[i]:
            for nb in _neighbours(p):
                j = pos_map.get(nb)
                if j is not None and j > i + 1 and seq_h[j]:
                    contact_set.add((i, j))

    cooling = (T_start / T_end) ** (1.0 / max(max_iter, 1))
    T = T_start

    for it in range(max_iter):
        if _time_left() < 0.5:
            break
        T = T_start / (cooling ** it)

        move_type = rng.random()
        best_delta = 0
        best_move = None

        if move_type < 0.4:
            # Pivot move: rotate tail around a pivot
            if n >= 4:
                pivot_idx = rng.randint(1, n - 2)
                pivot = current[pivot_idx]
                seg_start = pivot_idx + 1
                if seg_start < n:
                    segment = current[seg_start:]
                    fixed_set = set(current[:seg_start])

                    for angle in (90, -90, 180):
                        rotated = []
                        valid = True
                        for p in segment:
                            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                            if angle == 90:
                                ndx, ndy = -dy, dx
                            elif angle == -90:
                                ndx, ndy = dy, -dx
                            else:
                                ndx, ndy = -dx, -dy
                            new_p = (pivot[0] + ndx, pivot[1] + ndy)
                            if new_p in fixed_set:
                                valid = False
                                break
                            rotated.append(new_p)

                        if not valid:
                            continue

                        # Check unit steps at junction
                        if seg_start > 0:
                            if abs(rotated[0][0] - current[seg_start - 1][0]) + \
                               abs(rotated[0][1] - current[seg_start - 1][1]) != 1:
                                continue

                        # Compute energy delta
                        delta = 0
                        old_positions = {current[seg_start + k]: seg_start + k for k in range(len(segment))}
                        new_positions = {rotated[k]: seg_start + k for k in range(len(segment))}

                        # Remove contacts involving moved residues
                        for k in range(len(segment)):
                            idx = seg_start + k
                            if seq_h[idx]:
                                old_p = current[idx]
                                for nb in _neighbours(old_p):
                                    j = pos_map.get(nb)
                                    if j is not None and j < idx and seq_h[j] and (j, idx) in contact_set:
                                        if not (seg_start <= j < seg_start + len(segment)):
                                            delta += 1

                        # Add new contacts
                        for k in range(len(segment)):
                            idx = seg_start + k
                            if seq_h[idx]:
                                new_p = rotated[k]
                                for nb in _neighbours(new_p):
                                    j = pos_map.get(nb)
                                    if j is not None and j < idx and seq_h[j]:
                                        if not (seg_start <= j < seg_start + len(segment)):
                                            if (j, idx) not in contact_set:
                                                delta -= 1
                                    # Check if j is in the moved segment
                                    if j is not None and seg_start <= j < seg_start + len(segment) and seq_h[j]:
                                        if abs(j - idx) > 1:
                                            if (min(j, idx), max(j, idx)) not in contact_set:
                                                delta -= 1

                        if delta < best_delta:
                            best_delta = delta
                            best_move = ('pivot', pivot_idx, seg_start, rotated)

        elif move_type < 0.6:
            # Slither move: move the end
            if n >= 3:
                last = current[-1]
                prev = current[-2]
                for site in _neighbours(prev):
                    if site == current[-3] or site == last:
                        continue
                    if site in pos_map and pos_map[site] != n - 1:
                        continue

                    delta = 0
                    if seq_h[n - 1]:
                        for nb in _neighbours(last):
                            j = pos_map.get(nb)
                            if j is not None and j < n - 1 and seq_h[j] and (j, n - 1) in contact_set:
                                delta += 1
                        for nb in _neighbours(site):
                            j = pos_map.get(nb)
                            if j is not None and j < n - 1 and seq_h[j]:
                                if (j, n - 1) not in contact_set:
                                    delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('slither', site)

        else:
            # Segment shift: move a short segment perpendicular
            seg_len = rng.choice([2, 3, 4])
            if n > seg_len + 2:
                i = rng.randint(1, n - seg_len - 2)
                d1 = (current[i][0] - current[i-1][0], current[i][1] - current[i-1][1])
                perp_dirs = [(-d1[1], d1[0]), (d1[1], -d1[0])]

                for px, py in perp_dirs:
                    new_seg = []
                    valid = True
                    for k in range(seg_len):
                        new_p = (current[i + k][0] + px, current[i + k][1] + py)
                        new_seg.append(new_p)

                    for new_p in new_seg:
                        j = pos_map.get(new_p)
                        if j is not None and not (i <= j < i + seg_len):
                            valid = False
                            break

                    if not valid:
                        continue

                    if abs(new_seg[0][0] - current[i-1][0]) + abs(new_seg[0][1] - current[i-1][1]) != 1:
                        continue
                    if abs(new_seg[-1][0] - current[i + seg_len][0]) + \
                       abs(new_seg[-1][1] - current[i + seg_len][1]) != 1:
                        continue
                    for k in range(seg_len - 1):
                        if abs(new_seg[k+1][0] - new_seg[k][0]) + \
                           abs(new_seg[k+1][1] - new_seg[k][1]) != 1:
                            valid = False
                            break
                    if not valid:
                        continue

                    delta = 0
                    for k in range(seg_len):
                        idx = i + k
                        if seq_h[idx]:
                            old_p = current[idx]
                            new_p = new_seg[k]
                            for nb in _neighbours(old_p):
                                j = pos_map.get(nb)
                                if j is not None and j != idx and seq_h[j] and (min(j,idx), max(j,idx)) in contact_set:
                                    if not (i <= j < i + seg_len):
                                        delta += 1
                            for nb in _neighbours(new_p):
                                j = pos_map.get(nb)
                                if j is not None and j != idx and seq_h[j]:
                                    if not (i <= j < i + seg_len):
                                        if (min(j, idx), max(j, idx)) not in contact_set:
                                            delta -= 1

                    if delta < best_delta:
                        best_delta = delta
                        best_move = ('segment', i, seg_len, new_seg)

        # Accept/reject
        if best_move is not None:
            accept = False
            if best_delta <= 0:
                accept = True
            else:
                if T > 0.01:
                    accept = rng.random() < math.exp(-best_delta / T)
                else:
                    accept = False

            if accept:
                mtype = best_move[0]
                if mtype == 'pivot':
                    _, pivot_idx, seg_start, rotated = best_move
                    for k in range(seg_start, n):
                        old_p = current[k]
                        del pos_map[old_p]
                    for k, new_p in enumerate(rotated):
                        current[seg_start + k] = new_p
                        pos_map[new_p] = seg_start + k

                elif mtype == 'slither':
                    _, site = best_move
                    old_p = current[-1]
                    del pos_map[old_p]
                    current[-1] = site
                    pos_map[site] = n - 1

                elif mtype == 'segment':
                    _, i, seg_len, new_seg = best_move
                    for k in range(seg_len):
                        old_p = current[i + k]
                        del pos_map[old_p]
                    for k, new_p in enumerate(new_seg):
                        current[i + k] = new_p
                        pos_map[new_p] = i + k

                # Update contact set
                contact_set = set()
                for idx, p in enumerate(current):
                    if seq_h[idx]:
                        for nb in _neighbours(p):
                            j = pos_map.get(nb)
                            if j is not None and j > idx + 1 and seq_h[j]:
                                contact_set.add((idx, j))

                current_energy += best_delta

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current)

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses contact-map guided growth with multiple restarts and simulated annealing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(42 + len(sequence))
    contact_pairs = _contact_pairs(sequence)

    best_chain = None
    best_energy = 0  # unfolded has energy 0

    # Time budget allocation
    if n <= 25:
        max_restarts = 600
        ls_iter = 120
    elif n <= 40:
        max_restarts = 350
        ls_iter = 90
    elif n <= 60:
        max_restarts = 180
        ls_iter = 70
    else:
        max_restarts = 80
        ls_iter = 50

    temps = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    for restart in range(max_restarts):
        if _time_left() < 1.0:
            break

        # Alternate between guided and random growth
        if restart % 3 == 0:
            temp = temps[restart % len(temps)]
            chain = _grow_guided(sequence, rng, contact_pairs, bias=1.0, temperature=temp)
        elif restart % 3 == 1:
            temp = temps[restart % len(temps)] + 0.5
            chain = _grow_guided(sequence, rng, contact_pairs, bias=0.5, temperature=temp)
        else:
            chain = _grow_random(sequence, rng, bias=0.3 + rng.random() * 0.7)

        if chain is None:
            continue

        if not _validate_chain(sequence, chain):
            continue

        energy = _energy(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)

        # Local search improvement
        if _time_left() < 2.0:
            break

        improved_chain, improved_energy = _local_search_sa(
            sequence, chain, rng, max_iter=ls_iter
        )

        if improved_chain is not None and _validate_chain(sequence, improved_chain):
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = list(improved_chain)

    # Final polish: one more SA pass on the best chain with more iterations
    if best_chain is not None and _time_left() > 3.0:
        polished_chain, polished_energy = _local_search_sa(
            sequence, best_chain, rng, max_iter=150, T_start=2.0, T_end=0.05
        )
        if polished_chain is not None and _validate_chain(sequence, polished_chain):
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = list(polished_chain)

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
