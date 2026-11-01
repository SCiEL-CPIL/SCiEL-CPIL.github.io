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

    # === Initial solution: multiple greedy strategies ===
    best_code = None
    best_size = total + 1

    strategies = [
        "natural",
        "random1",
        "random2",
        "random3",
        "random4",
        "random5",
    ]

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
        # Fallback
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # === Simulated Annealing Phase ===
    current_code = set(best_code)
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_arrays[c]] += 1

    best_size = len(current_code)
    best_code = list(current_code)
    best_cnt = cnt.copy()

    # Simulated annealing parameters
    initial_temp = 0.5
    cooling_rate = 0.9995
    min_temp = 0.01
    temp = initial_temp

    # Precompute overlap for faster swap validation
    overlap = np.zeros((total, total), dtype=np.int8)
    balls_containing = [[] for _ in range(total)]
    for i in range(total):
        for p in ball_indices[i]:
            balls_containing[p].append(i)
    
    for p in range(total):
        bc = balls_containing[p]
        for a in range(len(bc)):
            for b in range(a+1, len(bc)):
                i = bc[a]
                j = bc[b]
                overlap[i, j] += 1
                overlap[j, i] += 1

    def is_valid_swap(c_remove, c_add, code_set, cnt):
        """Check if swapping c_remove for c_add maintains coverage."""
        if c_add in code_set or c_add == c_remove:
            return False
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        in_r_not_a = ball_r[np.isin(ball_r, ball_a, assume_unique=False, invert=True)]
        if len(in_r_not_a) > 0 and np.any(cnt[in_r_not_a] < 2):
            return False
        return True

    def apply_swap(c_remove, c_add, code_set, cnt):
        """Apply a swap."""
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt[ball_r] -= 1
        cnt[ball_a] += 1

    def undo_swap(c_remove, c_add, code_set, cnt):
        """Undo a swap."""
        ball_r = ball_arrays[c_remove]
        ball_a = ball_arrays[c_add]
        code_set.remove(c_add)
        code_set.add(c_remove)
        cnt[ball_r] += 1
        cnt[ball_a] -= 1

    # Simulated annealing loop
    no_improve_streak = 0
    max_streak = 2000

    while time.time() < deadline - 10.0 and no_improve_streak < max_streak:
        improved = False

        # 1. Delete phase
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                best_cnt = cnt.copy()
            improved = True

        if not improved:
            no_improve_streak += 1
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
            continue
        else:
            no_improve_streak = 0

        # 2. Informed 1-for-1 swap phase with simulated annealing
        swap_attempts = 0
        max_swap_attempts = 100
        while swap_attempts < max_swap_attempts and time.time() < deadline - 10.0:
            swap_attempts += 1
            if len(current_code) == 0:
                break

            c_remove = random.choice(list(current_code))
            if c_remove not in current_code:
                continue
            c_add = random.randint(0, total - 1)

            if c_add == c_remove or c_add in current_code:
                continue

            ov = overlap[c_remove, c_add]
            if ov >= ball_size:
                # ball_r fully contained in ball_a: always valid
                apply_swap(c_remove, c_add, current_code, cnt)
                improved = True
            else:
                # Need to check non-overlapping points
                ball_r = ball_arrays[c_remove]
                ball_a = ball_arrays[c_add]
                mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                points_to_check = ball_r[~mask_in_a]
                if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                    # Invalid swap - skip
                    continue
                else:
                    apply_swap(c_remove, c_add, current_code, cnt)
                    improved = True

            # After swap, try to delete
            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    best_cnt = cnt.copy()
                improved = True

        # 3. 2-for-1 swap: try replacing two codewords with one
        two_for_one_attempts = 0
        max_two_for_one = 50
        while two_for_one_attempts < max_two_for_one and time.time() < deadline - 10.0:
            two_for_one_attempts += 1
            if len(current_code) < 2:
                break

            # Pick two codewords to remove
            c1, c2 = random.sample(list(current_code), 2)
            if c1 not in current_code or c2 not in current_code:
                continue

            ball1 = ball_arrays[c1]
            ball2 = ball_arrays[c2]
            
            # Try a few random c_add candidates
            for _ in range(3):
                c_add = random.randint(0, total - 1)
                if c_add in current_code:
                    continue
                ball_a = ball_arrays[c_add]
                
                # Check: after removing c1 and c2, adding c_add, all points still covered
                mask1_in_a = np.isin(ball1, ball_a, assume_unique=False)
                mask2_in_a = np.isin(ball2, ball_a, assume_unique=False)
                
                b1_not_a = ball1[~mask1_in_a]
                b2_not_a = ball2[~mask2_in_a]
                
                valid = True
                if len(b1_not_a) > 0:
                    in_b2 = np.isin(b1_not_a, ball2, assume_unique=False)
                    if np.any(in_b2):
                        if np.any(cnt[b1_not_a[in_b2]] < 3):
                            valid = False
                    if valid and np.any(~in_b2):
                        if np.any(cnt[b1_not_a[~in_b2]] < 2):
                            valid = False
                
                if valid and len(b2_not_a) > 0:
                    in_b1 = np.isin(b2_not_a, ball1, assume_unique=False)
                    if np.any(in_b1):
                        if np.any(cnt[b2_not_a[in_b1]] < 3):
                            valid = False
                    if valid and np.any(~in_b1):
                        if np.any(cnt[b2_not_a[~in_b1]] < 2):
                            valid = False
                
                if not valid:
                    continue
                
                # Valid 2-for-1 swap
                current_code.remove(c1)
                current_code.remove(c2)
                current_code.add(c_add)
                cnt[ball1] -= 1
                cnt[ball2] -= 1
                cnt[ball_a] += 1
                improved = True
                break  # Break inner loop, try another pair
                
                # After 2-for-1, try to delete more
                while delete_redundant(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        best_cnt = cnt.copy()
                    improved = True

        if not improved:
            no_improve_streak += 1
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp
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
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)

    return result
