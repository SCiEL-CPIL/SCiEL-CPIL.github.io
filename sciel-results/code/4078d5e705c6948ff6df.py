import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

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
    
    # Precompute sorted balls for faster set operations? 
    # np.isin is O(N log M). For small N, it's okay.
    # But we can precompute a "membership" matrix? 2187x2187 bool is 4.7MB. Feasible.
    # Let's create a bitset or boolean matrix for balls.
    # ball_mask[i, j] is True if j is in ball of i.
    # This allows O(1) lookup for "is p in ball c?".
    # And O(N) for intersection checks if we use numpy logical ops.
    
    # Memory: 2187 * 2187 * 1 byte (bool) ~ 4.7 MB. Very safe.
    ball_mask = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mask[i, ball_arrays[i]] = True

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
                # Count how many new points this codeword covers
                # remaining & ball_mask[c]
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
            # Check if any point in ball[c] has count 1
            # If min(cnt[ball[c]]) >= 2, it's redundant
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

    # === Simulated Annealing Local Search ===
    
    current_code = set(best_code)
    cnt = best_cnt.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    alpha = 0.999 # Cooling rate per step
    
    T = T_start
    steps = 0
    max_steps = 100000
    
    while time.time() < deadline - 10.0 and steps < max_steps and T > T_end:
        steps += 1
        
        # 1. Try Deletions (always accept if valid)
        improved = False
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c in code_list:
            if np.all(cnt[ball_arrays[c]] >= 2):
                # Redundant
                current_code.remove(c)
                cnt[ball_arrays[c]] -= 1
                improved = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                break
        
        if improved:
            T = max(T * alpha, T_end)
            continue
            
        # 2. Try 2-for-1 Swaps (always accept if valid)
        found_2for1 = False
        for _ in range(20): # Limit attempts per step to keep time predictable
            if time.time() > deadline - 10.0:
                break
            if len(current_code) < 2:
                break
            
            c1, c2 = random.sample(list(current_code), 2)
            c_add = random.randint(0, total - 1)
            if c_add in current_code:
                continue
            
            # Check validity: unique points of c1 and c2 must be covered by c_add
            # Unique points of c1: p in ball[c1] where cnt[p] == 1
            # Since c1 and c2 are distinct, their unique points are disjoint.
            
            # Optimization: Precompute unique points for all codewords in current_code?
            # Or just compute on the fly for the pair?
            # On-the-fly is safer for memory but might be slower.
            # Let's compute unique points for c1 and c2.
            
            u1 = ball_arrays[c1][(cnt[ball_arrays[c1]] == 1)]
            u2 = ball_arrays[c2][(cnt[ball_arrays[c2]] == 1)]
            
            if len(u1) == 0 and len(u2) == 0:
                valid = True
            elif len(u1) == 0:
                valid = np.all(ball_mask[c_add, u2])
            elif len(u2) == 0:
                valid = np.all(ball_mask[c_add, u1])
            else:
                # Check if union is covered
                # Since u1 and u2 are disjoint, we can check both
                valid = np.all(ball_mask[c_add, u1]) and np.all(ball_mask[c_add, u2])
                
            if valid:
                current_code.remove(c1)
                current_code.remove(c2)
                current_code.add(c_add)
                cnt[ball_arrays[c1]] -= 1
                cnt[ball_arrays[c2]] -= 1
                cnt[ball_arrays[c_add]] += 1
                improved = True
                found_2for1 = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_list = list(current_code)
                break
                
        if improved:
            T = max(T * alpha, T_end)
            continue

        # 3. Simulated Annealing Moves (1-for-1 swaps)
        # We accept moves that increase size (worse) with probability exp(-delta/T)
        # We accept moves that decrease size (better) always.
        # We accept moves that keep size same (neutral) always? Or with some prob?
        # Usually, we want to explore the neutral space to find better configurations.
        
        c_remove = random.choice(list(current_code))
        c_add = random.randint(0, total - 1)
        
        if c_add in current_code or c_add == c_remove:
            T = max(T * alpha, T_end)
            continue
            
        # Check validity of swap
        # Valid if unique points of c_remove are covered by c_add
        u = ball_arrays[c_remove][(cnt[ball_arrays[c_remove]] == 1)]
        
        if len(u) == 0:
            valid = True
        else:
            valid = np.all(ball_mask[c_add, u])
            
        if valid:
            # Apply swap
            current_code.remove(c_remove)
            current_code.add(c_add)
            cnt[ball_arrays[c_remove]] -= 1
            cnt[ball_arrays[c_add]] += 1
            
            # This move is always "valid" in terms of coverage.
            # It doesn't change the size.
            # We accept it to explore the state space.
            
        else:
            # Invalid move: coverage would be lost.
            # In standard SA for covering codes, we might allow temporary violations?
            # But the problem requires a valid solution at the end.
            # Allowing invalid states requires tracking "damage" and penalizing it.
            # Let's stick to valid moves only for now (Metropolis on the valid state space).
            # This is essentially a random walk on the graph of valid configurations.
            # To escape local minima, we rely on the 2-for-1 moves which are rare.
            # We might need to add "add and remove" moves that temporarily increase size?
            # Or just accept that this is a hard problem.
            
            # Alternative: Add a move that adds a codeword (size +1) and then tries to remove two?
            # That's complex.
            
            # Let's just cool down and continue.
            pass

        T = max(T * alpha, T_end)

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
