import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # Place values for index conversion
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

    # Precompute ball arrays as numpy for faster operations
    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

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
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_arrays[best_c]] = True
            remaining[ball_arrays[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        """Remove redundant codewords where all points in ball have count >= 2."""
        changed = False
        for c in list(code_set):
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                code_set.remove(c)
                cnt[ball] -= 1
                changed = True
        return changed

    def try_swap(code_set, cnt, c_remove, c_add):
        """Try swapping c_remove for c_add. Returns True if valid."""
        if c_add in code_set or c_add == c_remove:
            return False
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        # Check: after removing c_remove, all points still covered?
        # Points in ball_r must have cnt >= 2 OR be in ball_a
        # Points in ball_a but not ball_r will get +1, so they're fine
        # Points in ball_r but not ball_a need cnt >= 2
        overlap_mask = np.zeros(total, dtype=bool)
        overlap_mask[ball_r] = True
        overlap_mask[ball_a] = False  # ball_a overrides: these points get +1
        # Actually simpler: check points in ball_r that are NOT in ball_a
        in_r_not_a = ball_r[np.isin(ball_r, ball_a, assume_unique=False, invert=True)]
        if len(in_r_not_a) > 0 and np.any(cnt[in_r_not_a] < 2):
            return False
        # Valid: apply swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_r] -= 1
        cnt[ball_a] += 1
        return True

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1

    strategies = [
        "natural",
        "random1",
        "random2",
        "random3",
    ]

    for s in strategies:
        if time.time() > deadline - 10:
            break
        if s == "natural":
            order = list(range(total))
        elif s == "random1":
            order = list(range(total))
            random.shuffle(order)
        elif s == "random2":
            order = list(range(total))
            random.shuffle(order)
        elif s == "random3":
            # Weighted: prefer words that are "central" (high degree in overlap graph)
            # Approximate by random
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
        # Fallback
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)

    # === Local Search Phase ===
    current_code = set(best_code)
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_arrays[c]] += 1

    best_size = len(current_code)
    best_code = list(current_code)

    # Precompute: for each pair, overlap size (for faster swap decisions)
    # Actually, let's just do random swaps and check

    # Local search loop
    no_improve_streak = 0
    max_streak = 500

    while time.time() < deadline - 5.0 and no_improve_streak < max_streak:
        improved = False

        # 1. Delete phase
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            improved = True

        # 2. Two-for-one move: remove two codewords, add one
        #    This can reduce code size by 1
        if len(current_code) >= 2:
            code_list = list(current_code)
            two_for_one_attempts = 0
            max_two_for_one = 50
            while two_for_one_attempts < max_two_for_one and time.time() < deadline - 5.0:
                two_for_one_attempts += 1
                # Pick two codewords to remove (prefer ones with low unique coverage)
                # Score each codeword by how many points it uniquely covers
                c1_idx = random.randint(0, len(code_list) - 1)
                c2_idx = random.randint(0, len(code_list) - 1)
                if c1_idx == c2_idx:
                    continue
                c1 = code_list[c1_idx]
                c2 = code_list[c2_idx]
                ball1 = ball_arrays[c1]
                ball2 = ball_arrays[c2]
                
                # Union of ball1 and ball2
                union = np.union1d(ball1, ball2)
                # Points in union that are NOT in either ball individually need cnt >= 2
                # Actually: after removing both, points in union must still be covered by other codewords
                # i.e., cnt[point] - (1 if point in ball1) - (1 if point in ball2) >= 1
                # Equivalently: for points in union, cnt >= 2 if in both, cnt >= 2 if in one
                # Wait: if point is in ball1 only, after removing c1 it has cnt-1, need cnt-1 >= 1, so cnt >= 2
                # If point is in ball2 only, same: cnt >= 2
                # If point is in both, after removing both it has cnt-2, need cnt-2 >= 1, so cnt >= 3
                
                in_both = np.intersect1d(ball1, ball2)
                in_one_only = np.setdiff1d(union, in_both)
                
                valid = True
                if len(in_both) > 0 and np.any(cnt[in_both] < 3):
                    valid = False
                elif len(in_one_only) > 0 and np.any(cnt[in_one_only] < 2):
                    valid = False
                
                if not valid:
                    continue
                
                # Find a codeword c3 whose ball covers all points in union that would become uncovered
                # After removing c1 and c2, points in union with cnt==1 (in_one_only) or cnt==2 (in_both) 
                # become uncovered. We need c3 to cover those.
                # Points that become uncovered:
                uncovered_after_removal = np.zeros(total, dtype=bool)
                if len(in_both) > 0:
                    mask_both_uncovered = cnt[in_both] == 2  # was 2, now 0
                    uncovered_after_removal[in_both[mask_both_uncovered]] = True
                if len(in_one_only) > 0:
                    mask_one_uncovered = cnt[in_one_only] == 1  # was 1, now 0
                    uncovered_after_removal[in_one_only[mask_one_uncovered]] = True
                
                n_uncovered = np.count_nonzero(uncovered_after_removal)
                
                if n_uncovered == 0:
                    # Just remove both, no need to add
                    current_code.remove(c1)
                    current_code.remove(c2)
                    cnt[ball1] -= 1
                    cnt[ball2] -= 1
                    improved = True
                    while delete_redundant(current_code, cnt):
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        improved = True
                    break
                
                # Need to find c3 that covers all uncovered points
                # Sample some candidates
                found_c3 = False
                for _ in range(100):
                    c3 = random.randint(0, total - 1)
                    if c3 == c1 or c3 == c2 or c3 in current_code:
                        continue
                    ball3 = ball_arrays[c3]
                    # Check if ball3 covers all uncovered points
                    if np.all(uncovered_after_removal[ball3]):
                        # Apply: remove c1, c2, add c3
                        current_code.remove(c1)
                        current_code.remove(c2)
                        current_code.add(c3)
                        cnt[ball1] -= 1
                        cnt[ball2] -= 1
                        cnt[ball3] += 1
                        improved = True
                        found_c3 = True
                        break
                
                if found_c3:
                    while delete_redundant(current_code, cnt):
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        improved = True
                    break

        if not improved:
            no_improve_streak += 1
            # Try swaps as a last resort before giving up
            swap_attempts = 0
            max_swap_attempts = 100
            while swap_attempts < max_swap_attempts and time.time() < deadline - 5.0:
                swap_attempts += 1
                if len(current_code) == 0:
                    break

                c_remove = random.choice(list(current_code))
                c_add = random.randint(0, total - 1)

                if c_add == c_remove or c_add in current_code:
                    continue

                # Check validity of swap
                ball_r = ball_arrays[c_remove]
                ball_a = ball_arrays[c_add]
                # Points in ball_r not in ball_a must have cnt >= 2
                mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                points_to_check = ball_r[~mask_in_a]
                if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                    continue

                # Valid swap
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_r] -= 1
                cnt[ball_a] += 1
                improved = True

                # After swap, try to delete
                while delete_redundant(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    improved = True
                break  # Only try one swap per iteration to allow two-for-one to work
        
        else:
            no_improve_streak = 0

    # Final: one more delete pass
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
