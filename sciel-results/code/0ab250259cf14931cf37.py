"""Covering code K_3(6,1): minimum dominating set on H(6,3) via structured local search."""
import time
import random
import itertools
import numpy as np

_start = time.time()
_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill


def _elapsed():
    return time.time() - _start


def _tetracode():
    # 9 words of length 4: a perfect code of radius 1 in H(4,3) (covers all 81 vertices).
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


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


def _solve_q3_n6_r1(deadline):
    """Structured local search for K_3(6,1): start from 81-word product code, reduce to 73."""
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute all words and index map
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor bitsets (radius 1 ball)
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 1 << i
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
    
    full_mask = (1 << total_words) - 1
    
    # Precompute perturbation vectors in Z_3^6
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)
    
    # Start from the 81-word product code (tetracode x identity on 2 coords)
    tetra = _tetracode()
    base_indices = set()
    for h in tetra:
        for tail in itertools.product(range(3), repeat=2):
            w = h + list(tail)
            base_indices.add(idx_map[tuple(w)])
    
    code_set = set(base_indices)
    cover_count = [0] * total_words
    for c in code_set:
        for idx in neighbor_indices[c]:
            cover_count[idx] += 1
    
    # Verify initial coverage
    uncovered_mask = 0
    for i in range(total_words):
        if cover_count[i] == 0:
            uncovered_mask |= (1 << i)
    if uncovered_mask != 0:
        # Should not happen, but fix it
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
    
    # Phase 1: Aggressive removal of redundant codewords
    def try_removals(code_set, cover_count, max_rounds=50):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and _elapsed() < deadline - 5:
            improved = False
            rounds += 1
            # Shuffle order for diversity
            order = list(code_set)
            random.shuffle(order)
            for c in order:
                if c not in code_set:
                    continue
                # Check if c is redundant: all its neighbors have cover_count >= 2
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    improved = True
                    break  # Restart after modification
        return code_set, cover_count
    
    code_set, cover_count = try_removals(code_set, cover_count)
    
    # Phase 2: Structured swaps (remove c, add c+d) to enable further reductions
    rng = random.Random(42)
    
    def structured_swap_attempt(code_set, cover_count, rng, max_attempts=500):
        improved = False
        for _ in range(max_attempts):
            if _elapsed() > deadline - 5:
                break
            if not code_set:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]
            
            if new_idx in code_set:
                continue
            
            # Try the swap
            # Remove c
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            
            # Check if any vertex becomes uncovered
            uncovered_after_removal = 0
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 0:
                    uncovered_after_removal |= (1 << idx)
            
            if uncovered_after_removal == 0:
                # c was redundant, just remove it (size decreases)
                code_set.remove(c)
                improved = True
                break
            
            # Add new_idx and see if it fixes the uncovered vertices
            code_set.remove(c)
            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1
            
            # Check if all previously uncovered are now covered
            still_uncovered = 0
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 0:
                    still_uncovered |= (1 << idx)
            
            if still_uncovered == 0:
                # Also check no new uncovered vertices elsewhere
                global_uncovered = 0
                for i in range(total_words):
                    if cover_count[i] == 0:
                        global_uncovered |= (1 << i)
                if global_uncovered == 0:
                    improved = True
                    break
                else:
                    # Revert
                    code_set.remove(new_idx)
                    for idx in neighbor_indices[new_idx]:
                        cover_count[idx] -= 1
                    code_set.add(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] += 1
            else:
                # Revert
                code_set.remove(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved
    
    # Interleave removals and swaps
    for round_num in range(20):
        if _elapsed() > deadline - 5:
            break
        
        # Try swaps to enable new removals
        code_set, cover_count, improved = structured_swap_attempt(code_set, cover_count, rng, max_attempts=200)
        
        # Try removals
        code_set, cover_count = try_removals(code_set, cover_count, max_rounds=20)
        
        # If we've reached 73 or fewer, we're done
        if len(code_set) <= 73:
            break
    
    # Phase 3: Final aggressive removal pass
    code_set, cover_count = try_removals(code_set, cover_count, max_rounds=100)
    
    # Verify final coverage
    uncovered_mask = 0
    for i in range(total_words):
        if cover_count[i] == 0:
            uncovered_mask |= (1 << i)
    
    if uncovered_mask != 0:
        # Repair any uncovered vertices
        while uncovered_mask and _elapsed() < deadline - 2:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
    
    # Convert to list of lists
    result = [list(all_tuples[c]) for c in code_set]
    return result


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
