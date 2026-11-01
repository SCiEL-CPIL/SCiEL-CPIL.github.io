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

    # Precompute ball_mask: ball_mask[i, j] is True if j is in ball of i
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

    # Precompute candidate_adds: for each codeword c, list of all words that could replace c
    # A word w is a valid add candidate if it can cover all unique points of c
    # We'll compute this dynamically during local search

    def greedy_cover(order=None):
        """Greedy set cover: repeatedly pick the codeword covering most uncovered points."""
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                new_cov = np.count_nonzero(remaining & ball_mask[c])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_mask[best_c]] = True
            remaining[ball_mask[best_c]] = False
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

    strategies = ["natural", "random1", "random2", "random3", "random4", "random5"]

    for s in strategies:
        if time.time() > deadline - 30:
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

    # === Systematic Local Search with Swap Moves ===
    
    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 200
    
    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False
        
        # Step 1: Try deletions
        for c in list(current_code):
            if np.all(cnt[ball_arrays[c]] >= 2):
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                break
                
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # Step 2: Systematic 1-for-1 swaps
        # For each codeword c, find a replacement w such that unique_points(c) ⊆ ball(w)
        code_list = list(current_code)
        swap_found = False
        
        for c in code_list:
            if time.time() > deadline - 10.0:
                break
            if c not in current_code:
                continue
            
            # Get unique points of c
            ball_c = ball_arrays[c]
            unique_mask = (cnt[ball_c] == 1)
            unique_points = ball_c[unique_mask]
            
            if len(unique_points) == 0:
                continue  # Should have been caught by deletion
            
            # Find candidate replacements: words w not in current_code
            # such that all unique_points are in ball(w)
            # Efficient check: for each unique point u, w must be in ball(u)
            # Intersect all balls of unique points, then exclude current_code
            
            # Start with all words, intersect with ball of each unique point
            candidates = np.ones(total, dtype=bool)
            for u in unique_points:
                candidates &= ball_mask[u]
            
            # Exclude current codewords
            for c2 in current_code:
                candidates[c2] = False
            
            # Pick the first valid candidate
            valid_candidates = np.where(candidates)[0]
            if len(valid_candidates) > 0:
                # Choose the one that maximizes overlap with current code's coverage
                # (to minimize disruption) - or just pick randomly
                w = random.choice(valid_candidates)
                
                # Apply swap
                current_code.remove(c)
                current_code.add(w)
                cnt[ball_arrays[c]] -= 1
                cnt[ball_arrays[w]] += 1
                improved = True
                swap_found = True
                break
        
        if improved:
            no_improve_streak = 0
            continue

        # Step 3: Try 2-for-1 swaps (remove 2, add 1)
        if len(current_code) >= 2:
            found_2for1 = False
            code_list2 = list(current_code)
            random.shuffle(code_list2)
            
            for i in range(min(20, len(code_list2))):
                if time.time() > deadline - 10.0:
                    break
                c1 = code_list2[i]
                if c1 not in current_code:
                    continue
                
                # Get unique points of c1
                ball_c1 = ball_arrays[c1]
                unique_mask1 = (cnt[ball_c1] == 1)
                unique_points1 = ball_c1[unique_mask1]
                
                for j in range(min(10, len(code_list2))):
                    if j == i:
                        continue
                    c2 = code_list2[j]
                    if c2 not in current_code or c2 == c1:
                        continue
                    
                    # Get unique points of c2 (after hypothetically removing c1)
                    # Actually, we need to check if unique_points(c1) ∪ unique_points(c2) ⊆ ball(w)
                    # where unique points are computed with both c1 and c2 present
                    
                    ball_c2 = ball_arrays[c2]
                    unique_mask2 = (cnt[ball_c2] == 1)
                    unique_points2 = ball_c2[unique_mask2]
                    
                    # Combine unique points from both
                    all_unique = np.concatenate([unique_points1, unique_points2])
                    all_unique = np.unique(all_unique)
                    
                    if len(all_unique) == 0:
                        # Both can be deleted
                        current_code.remove(c1)
                        current_code.remove(c2)
                        cnt[ball_arrays[c1]] -= 1
                        cnt[ball_arrays[c2]] -= 1
                        improved = True
                        found_2for1 = True
                        break
                    
                    # Find w such that all all_unique ⊆ ball(w)
                    candidates = np.ones(total, dtype=bool)
                    for u in all_unique:
                        candidates &= ball_mask[u]
                    
                    # Exclude current codewords
                    for c3 in current_code:
                        candidates[c3] = False
                    
                    valid_candidates = np.where(candidates)[0]
                    if len(valid_candidates) > 0:
                        w = random.choice(valid_candidates)
                        
                        # Apply 2-for-1
                        current_code.remove(c1)
                        current_code.remove(c2)
                        current_code.add(w)
                        cnt[ball_arrays[c1]] -= 1
                        cnt[ball_arrays[c2]] -= 1
                        cnt[ball_arrays[w]] += 1
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

        if not improved:
            no_improve_streak += 1

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code_list = list(current_code)

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
