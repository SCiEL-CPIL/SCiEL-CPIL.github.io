import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
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
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute ball membership matrix: ball_mem[i, j] = True if j is in ball(i)
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True
        
    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            # Sample a subset of candidates to speed up, or use all if small
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Local Search Helpers
    
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy to allow modification
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
            
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Use precomputed ball membership matrix for fast intersection check
        in_add = ball_mem[c_add, ball_rem]
        only_in_rem = ~in_add
        
        # Check if any element only in ball_rem has count < 2
        if np.any(cnt_arr[ball_rem[only_in_rem]] < 2):
            return False
        
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True

    def local_search(code_set, cnt_arr, time_limit):
        """
        Perform local search (delete + swap) starting from code_set.
        Returns the best code_set and cnt_arr found.
        """
        best_code = list(code_set)
        best_size = len(code_set)
        no_improve_counter = 0
        max_no_improve = 10 # Lower for faster cycles in GA context
        
        while time.time() < time_limit:
            # 1. Delete Phase
            while run_delete_phase(code_set, cnt_arr):
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                
            # 2. Swap Phase
            swaps_attempted = 0
            max_swaps = 200 # Limit swaps per iteration to keep it fast
            while swaps_attempted < max_swaps and time.time() < time_limit:
                swaps_attempted += 1
                if len(code_set) == 0:
                    break
                    
                # Pick random codeword to remove
                c = random.choice(list(code_set))
                # Pick candidate to add
                c_prime = random.randint(0, total - 1)
                
                if try_swap(c, c_prime, code_set, cnt_arr):
                    # After a valid swap, run delete phase
                    improved = False
                    while run_delete_phase(code_set, cnt_arr):
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)
                            improved = True
                            no_improve_counter = 0
                        elif len(code_set) < len(code_set) + 1:
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
                        
                    if best_size <= 160:
                        return best_code, best_size
                else:
                    no_improve_counter += 1
                    
            if no_improve_counter > max_no_improve:
                break
                
        return best_code, best_size

    # --- Population-Based Local Search (Genetic Approach) ---
    
    POP_SIZE = 8
    population = []
    
    # Initialize Population
    for i in range(POP_SIZE):
        if time.time() > deadline - 20:
            break
        init_code = greedy_build(seed=i)
        code_set = set(init_code)
        cnt_arr = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt_arr[ball_indices_np[c]] += 1
        
        # Initial cleanup
        while run_delete_phase(code_set, cnt_arr):
            pass
            
        # Quick local search to polish initial state
        best_code, best_size = local_search(code_set, cnt_arr, deadline - 10)
        
        # Rebuild cnt_arr for the best_code
        cnt_arr_best = np.zeros(total, dtype=np.int32)
        for c in best_code:
            cnt_arr_best[ball_indices_np[c]] += 1
            
        population.append({
            'code': set(best_code),
            'cnt': cnt_arr_best,
            'size': len(best_code)
        })
        
    # Track global best
    global_best_code = min(population, key=lambda x: x['size'])['code']
    global_best_size = min(population, key=lambda x: x['size'])['size']
    
    # Evolution Loop
    generation = 0
    while time.time() < deadline - 15:
        generation += 1
        
        # Selection: Tournament selection of 2 parents
        p1_idx = random.randint(0, len(population) - 1)
        p2_idx = random.randint(0, len(population) - 1)
        if p1_idx == p2_idx:
            p2_idx = (p2_idx + 1) % len(population)
        
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]
        
        # Crossover: Greedy recombination
        # Union of codewords from both parents
        union_codes = list(parent1['code'] | parent2['code'])
        random.shuffle(union_codes)
        
        new_code_set = set()
        new_cnt = np.zeros(total, dtype=np.int32)
        covered = np.zeros(total, dtype=bool)
        
        # Greedily add codewords from the union that cover new words
        for c in union_codes:
            ball = ball_indices_np[c]
            if np.any(~covered[ball]):
                new_code_set.add(c)
                new_cnt[ball] += 1
                covered[ball] = True
                
        # Ensure full coverage (in case union was insufficient, add greedy picks)
        while not np.all(covered):
            # Find best candidate from entire space
            best_c = -1
            max_new = -1
            for c in range(total):
                if c in new_code_set:
                    continue
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
            if best_c == -1 or max_new == 0:
                break
            new_code_set.add(best_c)
            new_cnt[ball_indices_np[best_c]] += 1
            covered[ball_indices_np[best_c]] = True
            
        # Repair: Delete redundancies
        while run_delete_phase(new_code_set, new_cnt):
            pass
            
        # Local Search Refinement
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        search_time = min(5.0, time_left * 0.5) # Use at most 5s or half remaining
        
        best_code, best_size = local_search(new_code_set, new_cnt, time.time() + search_time)
        
        # Update global best
        if best_size < global_best_size:
            global_best_size = best_size
            global_best_code = set(best_code)
            # Rebuild cnt for global best if needed for future, but we store sets
            # We need cnt for the population entry
            best_cnt = np.zeros(total, dtype=np.int32)
            for c in best_code:
                best_cnt[ball_indices_np[c]] += 1
            new_entry = {'code': set(best_code), 'cnt': best_cnt, 'size': best_size}
        else:
            # Rebuild cnt for the entry we are about to add/replace
            best_cnt = np.zeros(total, dtype=np.int32)
            for c in best_code:
                best_cnt[ball_indices_np[c]] += 1
            new_entry = {'code': set(best_code), 'cnt': best_cnt, 'size': best_size}
            
        # Update Population
        # Replace the worst member if the new one is better
        worst_idx = max(range(len(population)), key=lambda i: population[i]['size'])
        if new_entry['size'] < population[worst_idx]['size']:
            population[worst_idx] = new_entry
        elif random.random() < 0.1: # Small chance to replace even if not better (diversity)
            population[worst_idx] = new_entry
            
    # Final Result Conversion
    result = []
    for idx in global_best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
