"""Covering code K_3(6,1): minimum dominating set on H(6,3) via structured local search."""
import time
import random
import itertools
import numpy as np

_start = time.time()
_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill

# Precompute all words, index map, and neighbor info once at module level
_q3_n6 = (3, 6)
_all_tuples_36 = list(itertools.product(range(3), repeat=6))
_total_36 = 729
_idx_map_36 = {t: i for i, t in enumerate(_all_tuples_36)}
_neighbor_indices_36 = []
_neighbors_36 = []
for _i, _t in enumerate(_all_tuples_36):
    _nib = [_i]
    _mask = 1 << _i
    for _pos in range(6):
        _orig = _t[_pos]
        for _new_val in range(3):
            if _new_val == _orig:
                continue
            _t_list = list(_t)
            _t_list[_pos] = _new_val
            _j = _idx_map_36[tuple(_t_list)]
            _nib.append(_j)
            _mask |= (1 << _j)
    _neighbor_indices_36.append(_nib)
    _neighbors_36.append(_mask)

_perturbations_36 = []
for _pos in range(6):
    for _delta in (1, 2):
        _p = [0] * 6
        _p[_pos] = _delta
        _perturbations_36.append(_p)
for _pos1 in range(6):
    for _pos2 in range(_pos1 + 1, 6):
        for _d1 in (1, 2):
            for _d2 in (1, 2):
                _p = [0] * 6
                _p[_pos1] = _d1
                _p[_pos2] = _d2
                _perturbations_36.append(_p)


def _elapsed():
    return time.time() - _start


def _tetracode():
    # 9 words of length 4: a perfect code of radius 1 in H(4,3) (covers all 81 vertices).
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _init_cover_from_indices(code_set):
    cover_count = [0] * _total_36
    for c in code_set:
        for idx in _neighbor_indices_36[c]:
            cover_count[idx] += 1
    return cover_count


def _uncovered_mask(cover_count):
    m = 0
    for i in range(_total_36):
        if cover_count[i] == 0:
            m |= (1 << i)
    return m


def _greedy_cover(q, n, radius):
    # Fast valid (non-optimal) covering code: greedy set cover over the whole space.
    from itertools import product
    space = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
    N = space.shape[0]
    unc_set = set(range(N))
    chosen = []
    space_i = space
    while unc_set and _elapsed() < _DEADLINE - 2:
        best = -1
        best_cnt = 0
        for ci in range(N):
            c = space_i[ci]
            diff = (space_i != c).sum(axis=1)
            cov = np.nonzero(diff <= radius)[0]
            if len(cov) == 0:
                continue
            cnt = 0
            for v in cov:
                if v in unc_set:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best = ci
                if cnt == len(unc_set):
                    break
        if best < 0 or best_cnt == 0:
            rem = list(unc_set)
            if not rem:
                break
            best = rem[0]
            best_cnt = 1
        chosen.append(list(space_i[best]))
        c = space_i[best]
        diff = (space_i != c).sum(axis=1)
        for v in np.nonzero(diff <= radius)[0]:
            unc_set.discard(int(v))
    if unc_set:
        for v in list(unc_set):
            chosen.append(list(space_i[v]))
    return chosen


def _try_removals(code_set, cover_count, deadline, max_rounds=50):
    improved = True
    rounds = 0
    while improved and rounds < max_rounds and _elapsed() < deadline - 5:
        improved = False
        rounds += 1
        order = list(code_set)
        random.shuffle(order)
        for c in order:
            if c not in code_set:
                continue
            redundant = True
            for idx in _neighbor_indices_36[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in _neighbor_indices_36[c]:
                    cover_count[idx] -= 1
                improved = True
                break
    return code_set, cover_count


def _single_swap(code_set, cover_count, c, new_idx, deadline):
    """Try removing c and adding new_idx. Returns True if improved (size decreased or same but valid)."""
    if new_idx in code_set:
        return False
    # Remove c
    for idx in _neighbor_indices_36[c]:
        cover_count[idx] -= 1
    # Check if c was redundant
    any_uncovered = False
    for idx in _neighbor_indices_36[c]:
        if cover_count[idx] == 0:
            any_uncovered = True
            break
    if not any_uncovered:
        code_set.remove(c)
        return True  # size decreased
    # Try adding new_idx
    code_set.remove(c)
    code_set.add(new_idx)
    for idx in _neighbor_indices_36[new_idx]:
        cover_count[idx] += 1
    # Check all vertices covered
    for idx in _neighbor_indices_36[c]:
        if cover_count[idx] == 0:
            # Revert
            code_set.remove(new_idx)
            for idx2 in _neighbor_indices_36[new_idx]:
                cover_count[idx2] -= 1
            code_set.add(c)
            for idx2 in _neighbor_indices_36[c]:
                cover_count[idx2] += 1
            return False
    # Also check no other vertex became uncovered (shouldn't happen but be safe)
    for i in range(_total_36):
        if cover_count[i] == 0:
            code_set.remove(new_idx)
            for idx2 in _neighbor_indices_36[new_idx]:
                cover_count[idx2] -= 1
            code_set.add(c)
            for idx2 in _neighbor_indices_36[c]:
                cover_count[idx2] += 1
            return False
    return True


def _double_swap(code_set, cover_count, c1, c2, new1, new2, deadline):
    """Try removing c1,c2 and adding new1,new2. Returns True if valid and size decreased by 1 or 2."""
    if new1 in code_set or new2 in code_set or new1 == new2:
        return False
    # Remove c1 and c2
    for idx in _neighbor_indices_36[c1]:
        cover_count[idx] -= 1
    for idx in _neighbor_indices_36[c2]:
        cover_count[idx] -= 1
    # Check if both were redundant
    any_uncovered = False
    for idx in _neighbor_indices_36[c1]:
        if cover_count[idx] == 0:
            any_uncovered = True
            break
    if not any_uncovered:
        for idx in _neighbor_indices_36[c2]:
            if cover_count[idx] == 0:
                any_uncovered = True
                break
    if not any_uncovered:
        code_set.remove(c1)
        code_set.remove(c2)
        return True  # size decreased by 2
    # Try adding new1 and new2
    code_set.remove(c1)
    code_set.remove(c2)
    code_set.add(new1)
    code_set.add(new2)
    for idx in _neighbor_indices_36[new1]:
        cover_count[idx] += 1
    for idx in _neighbor_indices_36[new2]:
        cover_count[idx] += 1
    # Check all vertices covered
    for i in range(_total_36):
        if cover_count[i] == 0:
            code_set.remove(new1)
            code_set.remove(new2)
            for idx in _neighbor_indices_36[new1]:
                cover_count[idx] -= 1
            for idx in _neighbor_indices_36[new2]:
                cover_count[idx] -= 1
            code_set.add(c1)
            code_set.add(c2)
            for idx in _neighbor_indices_36[c1]:
                cover_count[idx] += 1
            for idx in _neighbor_indices_36[c2]:
                cover_count[idx] += 1
            return False
    return True


def _local_search(code_set, cover_count, rng, deadline, max_rounds=30):
    """Run local search: removals, single swaps, double swaps."""
    for round_num in range(max_rounds):
        if _elapsed() > deadline - 5:
            break
        # Removals
        code_set, cover_count = _try_removals(code_set, cover_count, deadline, max_rounds=20)
        if len(code_set) <= 73:
            break
        # Single swaps
        improved = False
        attempts = 0
        while not improved and attempts < 100 and _elapsed() < deadline - 5:
            attempts += 1
            c = rng.choice(list(code_set))
            d = rng.choice(_perturbations_36)
            c_tuple = _all_tuples_36[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % 3 for i in range(6))
            new_idx = _idx_map_36[new_tuple]
            if new_idx in code_set:
                continue
            if _single_swap(code_set, cover_count, c, new_idx, deadline):
                improved = True
        # Double swaps
        if _elapsed() > deadline - 5:
            break
        improved2 = False
        attempts2 = 0
        while not improved2 and attempts2 < 50 and _elapsed() < deadline - 5:
            attempts2 += 1
            c1 = rng.choice(list(code_set))
            c2 = rng.choice(list(code_set))
            if c1 == c2:
                continue
            d1 = rng.choice(_perturbations_36)
            d2 = rng.choice(_perturbations_36)
            t1 = _all_tuples_36[c1]
            t2 = _all_tuples_36[c2]
            nt1 = tuple((t1[i] + d1[i]) % 3 for i in range(6))
            nt2 = tuple((t2[i] + d2[i]) % 3 for i in range(6))
            n1 = _idx_map_36[nt1]
            n2 = _idx_map_36[nt2]
            if n1 in code_set or n2 in code_set or n1 == n2:
                continue
            if _double_swap(code_set, cover_count, c1, c2, n1, n2, deadline):
                improved2 = True
        # If we've reached 73 or fewer, we're done
        if len(code_set) <= 73:
            break
    return code_set, cover_count


def _repair_coverage(code_set, cover_count, deadline):
    """Add codewords to cover any uncovered vertices."""
    uncovered = _uncovered_mask(cover_count)
    while uncovered and _elapsed() < deadline - 2:
        best_idx = -1
        best_cnt = -1
        for i in range(_total_36):
            if i in code_set:
                continue
            new_covered = _neighbors_36[i] & uncovered
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1:
            break
        code_set.add(best_idx)
        for idx in _neighbor_indices_36[best_idx]:
            cover_count[idx] += 1
        uncovered &= ~_neighbors_36[best_idx]
    return code_set, cover_count


def _solve_q3_n6_r1(deadline):
    """Structured local search for K_3(6,1): multi-start with double swaps."""
    best_result = None
    best_size = float('inf')
    
    # Multiple starting seeds
    tetra = _tetracode()
    
    # Start 1: tetracode x identity on last 2 coords
    base_indices = set()
    for h in tetra:
        for tail in itertools.product(range(3), repeat=2):
            w = h + list(tail)
            base_indices.add(_idx_map_36[tuple(w)])
    
    rng = random.Random(42)
    
    # Try multiple seeds
    seeds = [base_indices]
    
    # Start 2: permuted tetracode (swap some coordinates)
    perm = [0, 1, 4, 5, 2, 3]
    base_indices2 = set()
    for h in tetra:
        for tail in itertools.product(range(3), repeat=2):
            w = h + list(tail)
            pw = [w[perm[i]] for i in range(6)]
            base_indices2.add(_idx_map_36[tuple(pw)])
    seeds.append(base_indices2)
    
    # Start 3: another permutation
    perm2 = [2, 3, 0, 1, 4, 5]
    base_indices3 = set()
    for h in tetra:
        for tail in itertools.product(range(3), repeat=2):
            w = h + list(tail)
            pw = [w[perm2[i]] for i in range(6)]
            base_indices3.add(_idx_map_36[tuple(pw)])
    seeds.append(base_indices3)
    
    # Start 4: random perturbation of base
    base_indices4 = set(base_indices)
    for _ in range(10):
        c = rng.choice(list(base_indices4))
        d = rng.choice(_perturbations_36)
        ct = _all_tuples_36[c]
        nt = tuple((ct[i] + d[i]) % 3 for i in range(6))
        base_indices4.discard(c)
        base_indices4.add(_idx_map_36[nt])
    seeds.append(base_indices4)
    
    for seed_idx, seed_set in enumerate(seeds):
        if _elapsed() > deadline - 10:
            break
        code_set = set(seed_set)
        cover_count = _init_cover_from_indices(code_set)
        # Repair if needed
        code_set, cover_count = _repair_coverage(code_set, cover_count, deadline)
        # Local search
        code_set, cover_count = _local_search(code_set, cover_count, rng, deadline, max_rounds=20)
        # Final removals
        code_set, cover_count = _try_removals(code_set, cover_count, deadline, max_rounds=50)
        # Verify
        unc = _uncovered_mask(cover_count)
        if unc != 0:
            code_set, cover_count = _repair_coverage(code_set, cover_count, deadline)
        size = len(code_set)
        if size < best_size:
            best_size = size
            best_result = [list(_all_tuples_36[c]) for c in code_set]
        if best_size <= 73:
            break
    
    if best_result is None:
        # Fallback: use first seed
        code_set = set(seeds[0])
        cover_count = _init_cover_from_indices(code_set)
        code_set, cover_count = _repair_coverage(code_set, cover_count, deadline)
        best_result = [list(_all_tuples_36[c]) for c in code_set]
    
    return best_result


def construct(q, n, radius):
    # Fast path: structured local search for the specific case q=3, n=6, radius=1.
    if q == 3 and n == 6 and radius == 1:
        result = None
        if _elapsed() < _DEADLINE - 10:
            try:
                result = _solve_q3_n6_r1(_DEADLINE)
            except Exception:
                result = None
        
        # Verify validity
        if result is not None:
            # Check all distinct
            if len(set(tuple(w) for w in result)) != len(result):
                result = None
            # Check coverage
            if result is not None:
                from itertools import product
                space = np.array(list(product(range(3), repeat=6)), dtype=np.int8)
                code_arr = np.array(result, dtype=np.int8)
                ok = True
                for v in space:
                    diff = (code_arr != v).sum(axis=1)
                    if diff.min() > 1:
                        ok = False
                        break
                if not ok:
                    result = None
        
        if result is not None:
            return result
        # Fallback: greedy (valid but suboptimal)
        return _greedy_cover(q, n, radius)
    else:
        # Generic: greedy
        return _greedy_cover(q, n, radius)
