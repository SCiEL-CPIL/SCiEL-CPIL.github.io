import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls using positional weights
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse: for each word, which codewords cover it
    inv_lists = [[] for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv_lists[w].append(c)
    inv_arrays = [np.array(lst, dtype=np.int32) for lst in inv_lists]
    
    best_code = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline - 15.0:
        code, size = _run_dual_greedy_prune(balls, inv_arrays, total, seed, deadline - 8.0)
        if size < best_size:
            best_size = size
            best_code = code
            if size <= 490:
                break
        seed += 1
    
    if best_code is None:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
    
    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    return result


def _run_dual_greedy_prune(balls, inv_arrays, total, seed, sub_deadline):
    rng = np.random.default_rng(seed)
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    
    # Dual prices: words with fewer covering candidates are "harder"
    num_coverers = np.array([len(inv_arrays[w]) for w in range(total)], dtype=np.float64)
    dual_prices = 1.0 / np.maximum(num_coverers, 1.0)
    
    active = np.ones(total, dtype=bool)
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    
    # Dual-guided greedy phase
    while uncovered_count > 0 and time.time() < sub_deadline - 20.0:
        # Score each active codeword: sum of dual prices of uncovered words in its ball
        scores = np.zeros(total, dtype=np.float64)
        uncovered_words = np.where(uncovered)[0]
        
        for w in uncovered_words:
            price = dual_prices[w]
            if price > 0:
                covering = inv_arrays[w]
                scores[covering] += price
        
        scores[~active] = -np.inf
        max_score = np.max(scores)
        
        if max_score <= 0:
            break
        
        # Pick from top candidates with randomness for diversity
        top_mask = np.isclose(scores, max_score, rtol=0, atol=1e-9)
        top_candidates = np.where(top_mask & active)[0]
        if len(top_candidates) == 0:
            break
        
        best_idx = top_candidates[rng.integers(len(top_candidates))]
        selected.append(best_idx)
        active[best_idx] = False
        
        # Update coverage
        ball = balls[best_idx]
        for w in ball:
            if cover_count[w] == 0:
                uncovered[w] = False
                uncovered_count -= 1
            cover_count[w] += 1
        
        # Update dual prices for remaining uncovered words
        if uncovered_count > 0 and uncovered_count < total:
            # Recompute for uncovered words only (sample for speed)
            remaining_uncov = np.where(uncovered)[0]
            for w in remaining_uncov[:min(len(remaining_uncov), 500)]:
                cnt = 0
                for c in inv_arrays[w]:
                    if active[c]:
                        cnt += 1
                dual_prices[w] = 1.0 / max(cnt, 1)
    
    # Aggressive pruning: remove redundant codewords
    for _pass in range(100):
        if time.time() > sub_deadline - 5.0:
            break
        if not selected:
            break
        sel_arr = np.array(selected, dtype=np.int32)
        sel_balls = balls[sel_arr]
        min_counts = np.min(cover_count[sel_balls], axis=1)
        redundant_mask = min_counts > 1
        if not np.any(redundant_mask):
            break
        redundant_indices = sel_arr[redundant_mask]
        cover_count[balls[redundant_indices]] -= 1
        selected = sel_arr[~redundant_mask].tolist()
    
    # Two-exchange local search: try removing pairs and adding one
    improved = True
    while improved and time.time() < sub_deadline - 2.0:
        improved = False
        if len(selected) < 2:
            break
        sel_arr = np.array(selected, dtype=np.int32)
        m = len(sel_arr)
        
        for _attempt in range(300):
            if time.time() > sub_deadline - 2.0:
                break
            i = rng.integers(m)
            j = rng.integers(m)
            if i == j:
                continue
            
            ball_i = balls[sel_arr[i]]
            ball_j = balls[sel_arr[j]]
            
            temp_cover = cover_count.copy()
            temp_cover[ball_i] -= 1
            temp_cover[ball_j] -= 1
            uncovered_after = np.where(temp_cover == 0)[0]
            
            if len(uncovered_after) == 0:
                cover_count[ball_i] -= 1
                cover_count[ball_j] -= 1
                selected = [sel_arr[k] for k in range(m) if k != i and k != j]
                improved = True
                break
            elif len(uncovered_after) <= 5:
                first_uncov = uncovered_after[0]
                candidates = balls[first_uncov]
                found = False
                for k in candidates:
                    ball_k = balls[k]
                    all_covered = True
                    for uw in uncovered_after:
                        if uw not in ball_k:
                            all_covered = False
                            break
                    if all_covered:
                        cover_count[ball_i] -= 1
                        cover_count[ball_j] -= 1
                        cover_count[ball_k] += 1
                        new_sel = [sel_arr[t] for t in range(m) if t != i and t != j]
                        if k not in new_sel:
                            new_sel.append(k)
                        selected = new_sel
                        improved = True
                        found = True
                        break
                if found:
                    break
    
    return selected, len(selected)
