import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values
    base = np.zeros(n, dtype=np.int32)
    base[n-1] = 1
    for i in range(n-2, -1, -1):
        base[i] = base[i+1] * q
        
    # Precompute all words
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n-1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute balls: for each word, list of indices within Hamming distance radius
    # For radius=1, ball = {word} U {word with one coord changed}
    ball_indices = np.zeros((total, ball_size), dtype=np.int32)
    for i in range(total):
        w = words[i]
        idx = 0
        ball_indices[i, idx] = i
        idx += 1
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i, idx] = new_idx
                idx += 1
                
    # Convert to list of arrays for faster indexing
    ball_indices_list = [ball_indices[i] for i in range(total)]
    
    # Greedy construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                ball = ball_indices_list[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_list[best_c]] = True
        return code
    
    # Initial greedy
    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(best_code)
    
    # Initialize coverage counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_list[c]] += 1
        
    # Helper: check if a swap is valid
    # After swap (remove c_remove, add c_add), we need cnt'[x] > 0 for all x
    # cnt'[x] = cnt[x] - 1_{x in ball_rem} + 1_{x in ball_add}
    # Only problematic: x in ball_rem \ ball_add, need cnt[x] >= 2
    def is_valid_swap(c_remove, c_add, cnt_arr):
        ball_rem = ball_indices_list[c_remove]
        ball_add = ball_indices_list[c_add]
        
        # For efficiency, we only need to check points in ball_rem that are NOT in ball_add
        # Since ball size is small (1+7*2=15), we can do this quickly
        ball_add_set = set(ball_add.tolist())
        for x in ball_rem:
            if x not in ball_add_set and cnt_arr[x] < 2:
                return False
        return True
    
    def perform_swap(c_remove, c_add, code_set, cnt_arr):
        ball_rem = ball_indices_list[c_remove]
        ball_add = ball_indices_list[c_add]
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        changed = False
        for c in list(code_set):
            ball = ball_indices_list[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    # Simulated Annealing
    T_start = 5.0
    T_end = 0.01
    cooling = 0.999
    
    T = T_start
    sa_step = 0
    
    while time.time() < deadline:
        # Periodic delete phase
        if sa_step % 100 == 0:
            while run_delete_phase(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    if best_size <= 160:
                        break
                        
            if best_size <= 160:
                break
                
        sa_step += 1
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap: remove one, add one
            if len(current_code) == 0:
                continue
            c_remove = random.choice(list(current_code))
            c_add = random.randint(0, total - 1)
            
            if c_add == c_remove or c_add in current_code:
                continue
            
            if is_valid_swap(c_remove, c_add, cnt):
                perform_swap(c_remove, c_add, current_code, cnt)
                # Run delete after swap
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        if best_size <= 160:
                            break
                            
        elif move_type < 0.7:
            # Delete: try to remove a codeword (temporarily allowing uncovered)
            if len(current_code) == 0:
                continue
            c_remove = random.choice(list(current_code))
            ball_rem = ball_indices_list[c_remove]
            
            # Calculate energy delta: size decreases by 1, but some points may become uncovered
            # Number of points that become uncovered = count of x in ball_rem where cnt[x] == 1
            newly_uncovered = np.sum(cnt[ball_rem] == 1)
            
            # Energy = size + lambda * uncovered
            # Delta E = -1 + lambda * newly_uncovered
            # Accept with probability exp(-Delta E / T)
            lam = 2.0
            delta_e = -1.0 + lam * newly_uncovered
            
            if delta_e <= 0 or random.random() < np.exp(-delta_e / T):
                # Perform delete (may create invalid state)
                current_code.remove(c_remove)
                cnt[ball_rem] -= 1
                
                # Try to fix by adding a codeword that covers many uncovered points
                if not np.all(cnt > 0):
                    uncovered_mask = (cnt == 0)
                    best_add = -1
                    max_fix = -1
                    for c in range(total):
                        ball = ball_indices_list[c]
                        fix = np.sum(uncovered_mask[ball])
                        if fix > max_fix:
                            max_fix = fix
                            best_add = c
                            if max_fix == ball_size:
                                break
                    if best_add >= 0 and best_add not in current_code:
                        current_code.add(best_add)
                        cnt[ball_indices_list[best_add]] += 1
                        
                # Update best if valid
                if np.all(cnt > 0):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        if best_size <= 160:
                            break
                            
        else:
            # Add: add a random codeword (increases size, may help escape)
            c_add = random.randint(0, total - 1)
            if c_add not in current_code:
                current_code.add(c_add)
                cnt[ball_indices_list[c_add]] += 1
                # Then try to delete
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        if best_size <= 160:
                            break
        
        # Cool down
        T = max(T_end, T * cooling)
        
        # Reset temperature if stuck
        if sa_step % 500 == 0:
            T = T_start * 0.5
        
        if best_size <= 160:
            break

    # Final delete pass
    final_code = set(best_code)
    final_cnt = np.zeros(total, dtype=np.int32)
    for c in final_code:
        final_cnt[ball_indices_list[c]] += 1
    while run_delete_phase(final_code, final_cnt):
        pass
    
    # Convert to list of words
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
