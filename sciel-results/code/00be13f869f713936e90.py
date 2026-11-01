import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Base values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Generate all words as index arrays
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    # Precompute ball arrays as numpy
    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]
    
    # Precompute coverers: coverers[p] is the set of codewords c such that p is in ball(c)
    # This is the inverse of the ball mapping.
    coverers = [set() for _ in range(total)]
    for c in range(total):
        for p in ball_indices[c]:
            coverers[p].add(c)

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        remaining = np.ones(total, dtype=bool)
        code = []
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            remaining[ball_arrays[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            if np.all(cnt[ball_arrays[c]] >= 2):
                code_set.remove(c)
                cnt[ball_arrays[c]] -= 1
                changed = True
        return changed

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1
    best_cnt = None

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5", "random6", "random7"]

    for s in strategies:
        if time.time() > deadline - 40:
            break
        if s == "natural":
            order = list(range(total))
        else:
            order = list(range(total))
            random.shuffle(order)
        
        code = greedy_cover(order)
        code_set = set(code)
        
        # Build counts
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        
        # Delete redundant
        while delete_redundant(code_set, cnt):
            pass
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Local Search with Efficient Swap Checks using Coverers ===
    
    def compute_unique_points_fast(code_set, cnt):
        """
        Fast version: only compute for codewords that might have unique points.
        """
        unique = {}
        for c in code_set:
            ball = ball_arrays[c]
            if np.any(cnt[ball] == 1):
                mask = (cnt[ball] == 1)
                unique[c] = ball[mask]
            else:
                unique[c] = np.array([], dtype=np.int32)
        return unique

    def find_candidates_for_unique(U, current_code):
        """
        Find codewords that cover all points in U, excluding those in current_code.
        U is a numpy array of point indices.
        """
        if len(U) == 0:
            return set()
        
        # Start with coverers of the first point
        cand_set = set(coverers[int(U[0])])
        
        # Intersect with coverers of other points
        for p in U[1:]:
            p_idx = int(p)
            cand_set.intersection_update(coverers[p_idx])
            if not cand_set:
                break
        
        # Remove current codewords
        cand_set.difference_update(current_code)
        
        return cand_set

    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 500
    
    while time.time() < deadline - 5.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Compute unique points
        unique_points = compute_unique_points_fast(current_code, cnt)
        
        # 2. Try Deletions
        for c in list(current_code):
            if len(unique_points[c]) == 0:
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                # After deletion, unique points for others might change, so break and recompute
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 3. Try 2-for-1 Swaps
        # We need to find a pair (c1, c2) and a candidate c_add such that
        # c_add covers all unique points of c1 and c2.
        code_list = list(current_code)
        random.shuffle(code_list)
        found_2for1 = False
        
        # To speed up, we can limit the number of pairs checked per iteration
        # but the intersection method is fast.
        # We'll iterate through all pairs in the shuffled list.
        for i in range(len(code_list)):
            if time.time() > deadline - 5.0:
                break
            c1 = code_list[i]
            u1 = unique_points[c1]
            
            for j in range(i + 1, len(code_list)):
                c2 = code_list[j]
                u2 = unique_points[c2]
                
                # Combine unique points
                # If both are empty, they are redundant, but deletion should have caught them.
                # If one is empty, it's effectively a 1-for-1 case, but 2-for-1 logic handles it if we just use u1.
                # Actually, if u1 is empty, c1 is redundant. Deletion should have removed it.
                # So we assume u1 and u2 are non-empty or one is empty.
                
                # Efficient union:
                if len(u1) == 0:
                    U = u2
                elif len(u2) == 0:
                    U = u1
                else:
                    # Union of two small arrays
                    U = np.union1d(u1, u2)
                
                if len(U) == 0:
                    # Both redundant? Should have been deleted.
                    continue
                
                cand_set = find_candidates_for_unique(U, current_code)
                
                if cand_set:
                    # Pick a candidate. 
                    # Heuristic: Pick one that covers the most total points (greedy) to help future steps?
                    # Or just pick the first one.
                    # Let's pick the one that maximizes the count of points it covers that are currently covered by c1 or c2?
                    # Simpler: just pick random or first.
                    c_add = next(iter(cand_set))
                    
                    # Apply swap
                    current_code.remove(c1)
                    current_code.remove(c2)
                    current_code.add(c_add)
                    cnt[ball_arrays[c1]] -= 1
                    cnt[ball_arrays[c2]] -= 1
                    cnt[ball_arrays[c_add]] += 1
                    
                    improved = True
                    found_2for1 = True
                    break
            if found_2for1:
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 4. Try 1-for-1 Swaps
        # Find c_remove and c_add such that c_add covers all unique points of c_remove.
        found_1for1 = False
        for c_remove in code_list:
            if time.time() > deadline - 5.0:
                break
            u_rem = unique_points[c_remove]
            
            if len(u_rem) == 0:
                # Should have been deleted
                continue
            
            cand_set = find_candidates_for_unique(u_rem, current_code)
            
            if cand_set:
                c_add = next(iter(cand_set))
                
                # Apply swap
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_arrays[c_remove]] -= 1
                cnt[ball_arrays[c_add]] += 1
                
                improved = True
                found_1for1 = True
                break
                
        if improved:
            no_improve_streak = 0
            continue

        if not improved:
            no_improve_streak += 1

    # Final: multiple delete passes
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        changed = delete_redundant(current_code, cnt)
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_list = list(current_code)
        if not changed:
            break
    
    # Convert back to list of lists
    result = []
    for idx in best_code_list:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
    
    return result
