import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 150s limit (evaluator kills at 150s)
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute ball indices for each word
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        ball_indices[i].append(i)
        temp = i
        for coord in range(n):
            orig = temp % q
            temp //= q
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Build ball membership bitsets (Python integers as bitsets)
    # ball_bitset[c] has bit w set if w is in ball(c)
    ball_bitset = [0] * total
    for c in range(total):
        b = 0
        for w in ball_indices[c]:
            b |= (1 << w)
        ball_bitset[c] = b
    
    # Build inverse: for each word w, list of codewords c such that w in ball(c)
    # i.e., coverers[w] = {c : w in ball(c)} = {c : c in ball(w)} since ball is symmetric
    coverers = [[] for _ in range(total)]
    for c in range(total):
        for w in ball_indices[c]:
            coverers[w].append(c)
    
    # Convert coverers to numpy arrays
    coverers_np = [np.array(cv, dtype=np.int32) for cv in coverers]
    
    def greedy_build(seed):
        rng = random.Random(seed)
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not covered.all():
            best_c = -1
            max_new = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = int(np.sum(~covered[ball]))
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Multiple greedy restarts
    best_initial = None
    best_initial_size = float('inf')
    
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() > deadline - 50.0:
            break
        init_code = greedy_build(seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)
    
    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize coverage count
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
    
    def run_delete_phase(code_set, cnt_arr):
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_indices_np[c]
                if cnt_arr[ball].min() >= 2:
                    code_set.remove(c)
                    cnt_arr[ball] -= 1
                    removed += 1
                    changed = True
        return removed
    
    def get_unique_words(c, cnt_arr):
        ball = ball_indices_np[c]
        return ball[cnt_arr[ball] == 1]
    
    # Local search with fast swap evaluation
    no_improve_counter = 0
    max_no_improve = 15
    rng = random.Random(12345)
    
    while time.time() < deadline - 5.0:
        # Delete phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # Swap phase - try multiple swaps
        swaps_done = 0
        max_swaps_per_round = 200
        while swaps_done < max_swaps_per_round and time.time() < deadline - 5.0:
            swaps_done += 1
            if len(current_code) == 0:
                break
            
            code_list = list(current_code)
            
            # Compute unique word counts
            unique_counts = np.zeros(len(code_list), dtype=np.int32)
            for i, c in enumerate(code_list):
                ball = ball_indices_np[c]
                unique_counts[i] = int(np.sum(cnt[ball] == 1))
            
            # Weighted random: prefer codewords with fewer unique words
            weights = 1.0 / (unique_counts + 1)
            total_w = weights.sum()
            r = rng.random() * total_w
            cumulative = 0.0
            chosen_idx = len(code_list) - 1
            for i in range(len(code_list)):
                cumulative += weights[i]
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            c = code_list[chosen_idx]
            ball_c = ball_indices_np[c]
            unique_to_c = ball_c[cnt[ball_c] == 1]
            
            if len(unique_to_c) == 0:
                # Any non-code word works
                c_prime = rng.randint(0, total - 1)
                attempts = 0
                while c_prime in current_code and attempts < 100:
                    c_prime = rng.randint(0, total - 1)
                    attempts += 1
                if c_prime in current_code:
                    continue
            else:
                # Need c_prime covering all unique words
                # Intersect balls
                valid_set = set(ball_indices_np[unique_to_c[0]].tolist())
                for u in unique_to_c[1:]:
                    valid_set &= set(ball_indices_np[u].tolist())
                    if not valid_set:
                        break
                
                valid_list = [v for v in valid_set if v not in current_code]
                if not valid_list:
                    no_improve_counter += 1
                    continue
                c_prime = rng.choice(valid_list)
            
            # Perform swap
            current_code.remove(c)
            current_code.add(c_prime)
            cnt[ball_indices_np[c]] -= 1
            cnt[ball_indices_np[c_prime]] += 1
            
            # Delete phase after swap
            removed = run_delete_phase(current_code, cnt)
            if removed > 0:
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
        
        # Restart if stuck
        if no_improve_counter > max_no_improve and time.time() < deadline - 20.0:
            seed = int(time.time() * 1000) % 100000
            init_code = greedy_build(seed)
            if len(init_code) <= len(current_code) + 3:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
        
        if best_size <= 160:
            break
    
    # Final polish: exhaustive deletion
    final_code = set(best_code)
    final_cnt = np.zeros(total, dtype=np.int32)
    for c in final_code:
        final_cnt[ball_indices_np[c]] += 1
    
    while time.time() < deadline - 1.0:
        found = False
        for c in list(final_code):
            ball = ball_indices_np[c]
            if final_cnt[ball].min() >= 2:
                final_code.remove(c)
                final_cnt[ball] -= 1
                found = True
                break
        if not found:
            break
    
    # Convert to result
    result = []
    for idx in final_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
