import math
import time
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 135  # 135s safety margin before 150s limit

    def elapsed():
        return time.time() - START

    def time_ok(limit_frac=1.0):
        return elapsed() < DEADLINE - 5 * limit_frac

    def compute_score_fast(u):
        """Compute score using numpy for speed on larger sets."""
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if n == 0:
            return -1.0
        if 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        
        # Sumset: all a+b for a,b in u
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        
        # Diffset: all a-b for a,b in u
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def compute_score_slow(u):
        """Fallback for very small sets or when numpy is slow."""
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def compute_score(u):
        if len(u) <= 8:
            return compute_score_slow(u)
        else:
            return compute_score_fast(u)

    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known optimal/near-optimal Golomb rulers
    golomb_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],       # n=7 optimal
        [0, 1, 4, 10, 12, 17, 25, 34],   # n=8 optimal
        [0, 1, 4, 10, 12, 17, 25, 34, 44], # n=9 (near-optimal)
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55], # n=10 (near-optimal)
    ]
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Evaluate all known Golomb rulers
    for u in golomb_candidates:
        if not time_ok():
            break
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
    
    # Local search: try to improve by adding/removing/adjusting elements
    def local_search(start_u, time_budget=30):
        """Hill-climbing local search on the score function."""
        start_t = time.time()
        current_u = sorted(set(start_u))
        current_score = compute_score(current_u)
        best_u = list(current_u)
        best_s = current_score
        
        # Try different neighborhoods
        while time.time() - start_t < time_budget and time_ok():
            improved = False
            
            # Neighborhood 1: add a new element
            max_u = max(current_u)
            search_hi = max_u + len(current_u) * len(current_u) + 10
            found_add = False
            for new_val in range(max_u + 1, search_hi + 1):
                candidate = sorted(current_u + [new_val])
                s = compute_score(candidate)
                if s > current_score + 1e-12:
                    current_u = candidate
                    current_score = s
                    if s > best_s:
                        best_s = s
                        best_u = list(candidate)
                    improved = True
                    found_add = True
                    break
            
            # Neighborhood 2: remove an element (if len > 2)
            if len(current_u) > 2:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue  # Must keep 0
                    candidate = [x for j, x in enumerate(current_u) if j != i]
                    if len(candidate) >= 2:
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
            
            # Neighborhood 3: shift a non-zero element by small delta
            if not improved:
                for i in range(len(current_u)):
                    if current_u[i] == 0:
                        continue
                    for delta in [-2, -1, 1, 2]:
                        new_val = current_u[i] + delta
                        if new_val < 1:
                            continue
                        candidate = list(current_u)
                        candidate[i] = new_val
                        candidate = sorted(set(candidate))
                        if 0 not in candidate or len(candidate) != len(current_u):
                            continue
                        s = compute_score(candidate)
                        if s > current_score + 1e-12:
                            current_u = candidate
                            current_score = s
                            if s > best_s:
                                best_s = s
                                best_u = list(candidate)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_u, best_s
    
    # Try local search from the best Golomb ruler found
    if time_ok(2.0):
        ls_u, ls_s = local_search(best_set, time_budget=25)
        if ls_s > best_score:
            best_score = ls_s
            best_set = ls_u
    
    # Try local search from other starting points
    if time_ok(3.0):
        for u in golomb_candidates[:5]:
            if not time_ok(3.0):
                break
            ls_u, ls_s = local_search(u, time_budget=10)
            if ls_s > best_score:
                best_score = ls_s
                best_set = ls_u
    
    # Try a greedy construction approach
    if time_ok(4.0):
        def greedy_build(n_target, time_budget=15):
            """Build a Golomb-like ruler greedily, optimizing score."""
            start_t = time.time()
            u = [0, 1]
            while len(u) < n_target and time.time() - start_t < time_budget and time_ok():
                max_u = max(u)
                best_new = None
                best_new_score = compute_score(u)
                search_hi = max_u + len(u) * len(u) + 5
                for new_val in range(max_u + 1, search_hi + 1):
                    candidate = sorted(u + [new_val])
                    if not is_golomb(candidate):
                        continue
                    s = compute_score(candidate)
                    if s > best_new_score:
                        best_new_score = s
                        best_new = candidate
                    if time.time() - start_t > time_budget:
                        break
                if best_new is not None:
                    u = best_new
                else:
                    # Try non-Golomb additions too
                    for new_val in range(max_u + 1, search_hi + 1):
                        candidate = sorted(u + [new_val])
                        s = compute_score(candidate)
                        if s > best_new_score:
                            best_new_score = s
                            best_new = candidate
                        if time.time() - start_t > time_budget:
                            break
                    if best_new is not None:
                        u = best_new
                    else:
                        break
            return u, compute_score(u)
        
        for n_target in [6, 8, 10, 12]:
            if not time_ok(4.0):
                break
            g_u, g_s = greedy_build(n_target, time_budget=8)
            if g_s > best_score:
                best_score = g_s
                best_set = g_u

    # Final: ensure we return a valid set containing 0
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    
    return best_set
