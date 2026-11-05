"""
HP Lattice Protein Folding: Beam Search + Fast Local Search (segment-reversal + pivot).

Strategy:
1. Beam search with symmetry-breaking dedup for diverse, high-quality initial folds.
2. Fast local search using incremental energy deltas with two move types:
   - Segment reversals (short segments, O(1) boundary checks)
   - Pivot rotations (tail rotation about a bond)
3. Multi-start with greedy growth for additional diversity.
4. Adaptive time allocation based on chain length.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _energy_full(sequence, chain):
    """Compute energy: -number of H-H contacts on neighbouring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=200, max_time=3.0):
    """
    Beam search for HP protein folding.
    Uses symmetry-breaking dedup (last-4 positions) for diversity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[(0, 0, 0)]]

    t_start = time.time()
    start = (0, 0, 0)
    # Each beam entry: (score, chain, occupied_set, pos_to_idx, energy)
    beam = [(0.0, [start], {start}, {start: 0}, 0)]
    
    # Symmetry breaking: track last-4 position tuples to avoid equivalent folds
    seen_keys = set()
    seen_keys.add(tuple([start] * 1))  # just the start

    for i in range(1, n):
        if _time_left() < 5.0 or (time.time() - t_start) > max_time:
            break

        candidates = []
        seq_i = sequence[i]

        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last

            for dx, dy, dz in _OFFSETS:
                site = (lx + dx, ly + dy, lz + dz)
                if site in occ:
                    continue

                # Symmetry-breaking dedup: use last 4 positions
                if i >= 3:
                    key = tuple(chain[-3:]) + (site,)
                    if key in seen_keys:
                        continue
                else:
                    key = tuple(chain[-1:]) + (site,)
                    if key in seen_keys:
                        continue

                # Incremental energy gain from new H-H contacts
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = pos_idx.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # Free neighbours (room to grow)
                free = 0
                for dx2, dy2, dz2 in _OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    if q not in occ:
                        free += 1

                new_energy = energy - gain
                # Score: energy (lower is better) + small free-space bonus
                new_score = new_energy - 0.03 * free + random.random() * 0.005

                new_chain = chain + [site]
                new_occ = set(occ)
                new_occ.add(site)
                new_pos = dict(pos_idx)
                new_pos[site] = i

                candidates.append((new_score, new_chain, new_occ, new_pos, new_energy))
                seen_keys.add(key)

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]
        # Reset seen_keys for next iteration (only dedup within same step)
        seen_keys = set()
        for _, ch, _, _, _ in beam:
            if len(ch) >= 4:
                seen_keys.add(tuple(ch[-4:]))
            elif len(ch) >= 1:
                seen_keys.add(tuple(ch[-len(ch):]))

    if not beam:
        return []

    # Return all complete folds from the beam, sorted by energy
    results = []
    for score, chain, occ, pos_idx, energy in beam:
        if len(chain) == n:
            results.append((energy, chain))

    results.sort(key=lambda x: x[0])
    return [chain for _, chain in results[:beam_width]]


def _initial_grow(seq, dim, rng, bias=1.0):
    """Randomized greedy growth with H-contact bias."""
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    h_set = {i for i, s in enumerate(seq) if s == 'H'}

    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours_fast(last):
            if site in occupied:
                continue
            score = 0.0
            if i in h_set:
                for q in _neighbours_fast(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and seq[j] == 'H':
                        score += 3.0
            free = sum(1 for q in _neighbours_fast(site) if q not in occupied)
            score += free * 0.05
            score += rng.random() * bias
            candidates.append((score, site))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        if rng.random() < 0.7:
            site = candidates[0][1]
        else:
            k = min(3, len(candidates))
            site = candidates[rng.randint(0, k - 1)][1]

        chain.append(site)
        occupied[site] = i

    return chain


def _neighbours_fast(p):
    x, y, z = p
    return ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1))


def _local_search(sequence, dimension, chain, rng, time_limit, aggressive=False):
    """
    Fast local search with segment reversals and pivot rotations.
    Uses incremental energy computation.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]
    pos_idx = {p: i for i, p in enumerate(chain)}
    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    if aggressive:
        T = max(2.0, -current_energy * 0.8)
        T_min = 0.003
        alpha = 0.9998
        max_seg_base = min(n - 2, 20)
    else:
        T = max(1.5, -current_energy * 0.5)
        T_min = 0.004
        alpha = 0.9997
        max_seg_base = min(n - 2, 14)

    t_start = time.time()
    t_end = t_start + time_limit

    # Precompute rotation helpers
    def rot_x(p):
        x, y, z = p
        return (x, -z, y)
    def rot_x2(p):
        x, y, z = p
        return (x, -y, -z)
    def rot_y(p):
        x, y, z = p
        return (z, y, -x)
    def rot_y2(p):
        x, y, z = p
        return (-x, y, -z)
    def rot_z(p):
        x, y, z = p
        return (y, -x, z)
    def rot_z2(p):
        x, y, z = p
        return (-x, -y, z)

    rotations = [rot_x, rot_x2, rot_y, rot_y2, rot_z, rot_z2]

    no_improve = 0
    max_no_improve = 8000 if aggressive else 15000

    while time.time() < t_end and _time_left() > 3.0:
        if no_improve > max_no_improve:
            break

        if rng.random() < 0.5:
            # ---- Segment reversal ----
            max_seg = max_seg_base
            if max_seg < 2:
                no_improve += 1
                continue
            r = rng.random()
            if r < 0.5:
                seg_len = rng.randint(2, min(4, max_seg))
            elif r < 0.8:
                seg_len = rng.randint(4, min(8, max_seg))
            else:
                seg_len = rng.randint(8, max_seg)
            seg_len = min(seg_len, max_seg)
            if seg_len < 2:
                no_improve += 1
                continue

            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            # Save old positions
            old_chain = chain[s:e+1][:]
            old_pos = {}
            for k in range(s, e+1):
                old_pos[chain[k]] = k

            # Reverse the segment
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                chain[a], chain[b] = chain[b], chain[a]
                pos_idx[chain[a]] = a
                pos_idx[chain[b]] = b

            # Check boundary unit-steps
            valid = True
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False

            # Check self-avoidance in moved region
            if valid:
                seen = set()
                for k in range(s, e+1):
                    p = chain[k]
                    if p in seen:
                        valid = False
                        break
                    seen.add(p)
                    # Check against non-moved positions
                    j = pos_idx.get(p)
                    if j is not None and j < s or (j is not None and j > e):
                        # This position is occupied by a non-moved residue
                        # But wait, pos_idx already updated. Need to check properly.
                        pass

            # More careful self-avoidance check: build set of non-moved positions
            if valid:
                moved_set = set(chain[s:e+1])
                for k in range(s, e+1):
                    p = chain[k]
                    # Check if p is in the set of positions held by non-moved residues
                    # We need to check against original non-moved positions
                    # Actually, since we only moved s..e, check if any two positions in s..e are equal
                    # or if any position in s..e equals a position outside s..e
                    for dx, dy, dz in _OFFSETS:
                        pass  # skip, do simpler check below

            # Simple self-avoidance: check all positions in moved segment are unique
            # and don't collide with fixed parts
            if valid:
                fixed_positions = set()
                for k in range(n):
                    if k < s or k > e:
                        fixed_positions.add(chain[k])
                moved_positions = chain[s:e+1]
                if len(set(moved_positions)) != len(moved_positions):
                    valid = False
                else:
                    for p in moved_positions:
                        if p in fixed_positions:
                            valid = False
                            break

            if valid:
                # Compute energy delta
                delta = 0
                for k in range(s, e+1):
                    if sequence[k] == 'H':
                        p = chain[k]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(k - j) > 1 and sequence[j] == 'H':
                                delta -= 1
                new_energy = current_energy + delta

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = chain[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    for k in range(s, e+1):
                        chain[k] = old_chain[k - s]
                        pos_idx[chain[k]] = k
                    no_improve += 1
            else:
                # Revert
                for k in range(s, e+1):
                    chain[k] = old_chain[k - s]
                    pos_idx[chain[k]] = k
                no_improve += 1

        else:
            # ---- Pivot rotation ----
            pivot_idx = rng.randint(1, n - 1)
            if pivot_idx >= n - 1:
                no_improve += 1
                continue

            rot = rotations[rng.randint(0, 5)]
            center = chain[pivot_idx - 1]

            # Save old tail
            old_tail = chain[pivot_idx:]
            old_tail_pos = {p: k for k, p in enumerate(old_tail)}

            # Apply rotation to tail
            new_tail = []
            valid = True
            for k in range(pivot_idx, n):
                v = (chain[k][0] - center[0],
                     chain[k][1] - center[1],
                     chain[k][2] - center[2])
                rv = rot(v)
                np = (center[0] + rv[0], center[1] + rv[1], center[2] + rv[2])
                new_tail.append(np)

            # Check self-avoidance of new tail
            tail_set = set(new_tail)
            if len(tail_set) != len(new_tail):
                valid = False

            # Check collision with head
            if valid:
                head_positions = set(chain[:pivot_idx])
                for p in new_tail:
                    if p in head_positions:
                        valid = False
                        break

            # Check boundary unit step
            if valid:
                p1 = chain[pivot_idx - 1]
                p2 = new_tail[0]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False

            if valid:
                # Update chain and pos_idx
                for k in range(pivot_idx, n):
                    pos_idx.pop(chain[k], None)
                    chain[k] = new_tail[k - pivot_idx]
                    pos_idx[chain[k]] = k

                # Compute energy delta
                delta = 0
                for k in range(pivot_idx, n):
                    if sequence[k] == 'H':
                        p = chain[k]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(k - j) > 1 and sequence[j] == 'H':
                                delta -= 1
                new_energy = current_energy + delta

                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = chain[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    for k in range(pivot_idx, n):
                        pos_idx.pop(chain[k], None)
                        chain[k] = old_tail[k - pivot_idx]
                        pos_idx[chain[k]] = k
                    no_improve += 1
            else:
                no_improve += 1

        # Cool
        T = max(T_min, T * alpha)

    return best_chain


def fold(sequence, dimension):
    """Fold HP chain on cubic lattice. Returns list of lattice points."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    rng = random.Random(42 + n)
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']

    best_chain = None
    best_energy = float('inf')

    # Allocate time based on chain length
    if n <= 48:
        total_time = 4.0
    elif n <= 80:
        total_time = 5.5
    elif n <= 120:
        total_time = 7.0
    else:
        total_time = 9.0

    max_total = _time_left() * 0.95
    total_time = min(total_time, max_total)

    t_start = time.time()

    # Phase 1: Beam search for good initial solutions
    beam_width = 200 if n <= 60 else 100
    beam_time = min(total_time * 0.35, 2.5)
    beam_results = _beam_search(sequence, dimension, beam_width=beam_width, max_time=beam_time)

    for chain in beam_results:
        e = _energy_full(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)

    # Phase 2: SA refinement on top beam results
    remaining = total_time - (time.time() - t_start)
    if remaining > 0.5 and beam_results:
        for chain in beam_results[:4]:
            if _time_left() < 5.0:
                break
            sa_time = min(remaining * 0.25, 2.5 if n <= 48 else 3.5)
            refined = _local_search(sequence, dimension, chain, rng, sa_time)
            new_e = _energy_full(sequence, refined)
            if new_e < best_energy:
                best_energy = new_e
                best_chain = list(refined)
            remaining = total_time - (time.time() - t_start)
            if remaining < 0.5:
                break

    # Phase 3: Multi-start with greedy growth + SA
    remaining = total_time - (time.time() - t_start)
    while remaining > 1.0 and _time_left() > 5.0:
        bias = 0.5 + rng.random() * 1.5
        chain = _initial_grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        e = _energy_full(sequence, chain)

        if e < best_energy:
            best_energy = e
            best_chain = list(chain)

        remaining = total_time - (time.time() - t_start)
        if remaining < 0.5:
            break

        sa_time = min(remaining * 0.35, 2.5 if n <= 48 else 3.5)
        chain = _local_search(sequence, chain, dimension, rng, sa_time)

        e = _energy_full(sequence, chain)

        if e < best_energy:
            best_energy = e
            best_chain = list(chain)

        remaining = total_time - (time.time() - t_start)

    # Phase 4: Final aggressive polish
    if best_chain is not None:
        remaining = total_time - (time.time() - t_start)
        if remaining > 1.0 and _time_left() > 5.0:
            polish_time = min(remaining * 0.6, 4.0)
            polished = _local_search(sequence, best_chain, dimension, rng, polish_time, aggressive=True)
            e = _energy_full(sequence, polished)
            if e < best_energy:
                best_energy = e
                best_chain = list(polished)

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
