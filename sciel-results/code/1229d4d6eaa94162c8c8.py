import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    N = q ** n
    deadline = time.time() + 110.0
    
    # Precompute balls
    def idx_to_digits(idx):
        return [int(idx // (q ** i) % q) for i in range(n)]
    
    def digits_to_idx(digits):
        idx = 0
        for i, d in enumerate(digits):
            idx += d * (q ** i)
        return idx

    balls = []
    for idx in range(N):
        digits = idx_to_digits(idx)
        ball = [idx]
        for pos in range(n):
            for delta in range(1, q):
                new_digits = digits[:]
                new_digits[pos] = (digits[pos] + delta) % q
                ball.append(digits_to_idx(new_digits))
        balls.append(ball)
    
    # Membership matrix
    mem = np.zeros((N, N), dtype=bool)
    for c in range(N):
        mem[c, balls[c]] = True

    def greedy_set_cover():
        """
        Greedy set cover to find an initial code.
        """
        uncovered = np.ones(N, dtype=bool)
        code = []
        
        # Precompute coverage counts for speed? 
        # We need to find c that maximizes sum(uncovered & mem[c])
        
        while np.any(uncovered):
            # Vectorized calculation of how many new words each c covers
            # mem is (N, N). uncovered is (N,).
            # We want row-wise sum of (mem * uncovered)
            # But we only care about uncovered.
            # new_covered = mem @ uncovered (boolean to int)
            # This is O(N^2) per iteration.
            
            # Optimization: 
            # new_covered = np.sum(mem * uncovered[None, :], axis=1)
            # This creates a large intermediate array.
            # Better: 
            # new_covered = mem.dot(uncovered.astype(int))
            
            new_covered = mem.dot(uncovered.astype(int))
            
            # Find the best candidate
            # If there are ties, pick random or first
            max_cov = np.max(new_covered)
            if max_cov == 0:
                break # Should not happen if uncovered is not empty and mem is valid
                
            candidates = np.where(new_covered == max_cov)[0]
            c = random.choice(candidates)
            
            code.append(c)
            uncovered[balls[c]] = False
            
            if time.time() > deadline - 10:
                break
                
        return code

    def greedy_reduce(codewords_list, cov_array):
        """Greedy pass to remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords_list)):
                c = codewords_list[i]
                if np.min(cov_array[balls[c]]) > 1:
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    changed = True
                    break
        return codewords_list, cov_array

    def simulated_annealing(codewords_list, cov_array, T_start, cooling_rate, min_T, time_budget):
        """Run simulated annealing on the given state."""
        T = T_start
        best_size = len(codewords_list)
        best_codewords = codewords_list[:]
        best_cov = cov_array.copy()
        
        tabu_list = {} # c -> time it was last modified
        tabu_tenure = 100
        
        while time.time() < time_budget and T > min_T:
            # 1. Try to Delete a random codeword
            if len(codewords_list) > 0:
                i = random.randint(0, len(codewords_list) - 1)
                c = codewords_list[i]
                
                # Check tabu? Deletion is usually good if valid.
                if np.min(cov_array[balls[c]]) > 1:
                    cov_array[balls[c]] -= 1
                    codewords_list.pop(i)
                    if len(codewords_list) < best_size:
                        best_size = len(codewords_list)
                        best_codewords = codewords_list[:]
                        best_cov = cov_array.copy()
                    T *= cooling_rate
                    continue
                    
            # 2. Try a Swap
            if len(codewords_list) == 0:
                break
                
            i = random.randint(0, len(codewords_list) - 1)
            c_old = codewords_list[i]
            
            # Generate c_new
            # Bias towards words that cover many uncovered words if we remove c_old?
            # Or just random.
            c_new = random.randint(0, N - 1)
            
            if c_new == c_old:
                T *= cooling_rate
                continue
                
            # Check Tabu
            if c_new in tabu_list and tabu_list[c_new] > len(codewords_list) - tabu_tenure:
                T *= cooling_rate
                continue

            # Tentative removal of c_old
            cov_array[balls[c_old]] -= 1
            
            # Find uncovered words
            missing = np.where(cov_array < 1)[0]
            
            if len(missing) == 0:
                # c_old was redundant
                codewords_list.pop(i)
                if len(codewords_list) < best_size:
                    best_size = len(codewords_list)
                    best_codewords = codewords_list[:]
                    best_cov = cov_array.copy()
                T *= cooling_rate
                continue
            else:
                # c_old is essential. Check if c_new covers all missing words.
                if mem[c_new, missing].all():
                    # Accept swap
                    cov_array[balls[c_new]] += 1
                    codewords_list[i] = c_new
                    tabu_list[c_new] = len(codewords_list)
                    T *= cooling_rate
                else:
                    # Reject swap, revert
                    cov_array[balls[c_old]] += 1
                    T *= cooling_rate
        
        return best_codewords, best_cov, best_size

    # Run Greedy Set Cover
    initial_code = greedy_set_cover()
    
    # Calculate cov for initial code
    cov = np.zeros(N, dtype=np.int32)
    for c in initial_code:
        cov[balls[c]] += 1
        
    # Greedy Reduce
    initial_code, cov = greedy_reduce(initial_code, cov)
    
    all_best_codewords = [initial_code]
    all_best_sizes = [len(initial_code)]
    
    # Run SA
    sa_configs = [
        (2.0, 0.995, 1e-4, 20),
        (1.0, 0.998, 1e-5, 20),
        (5.0, 0.99, 1e-3, 20),
    ]
    
    for T_start, cooling_rate, min_T, time_budget in sa_configs:
        if time.time() >= deadline - 5:
            break
        
        cw = initial_code[:]
        cv = cov.copy()
        
        best_cw, best_cv, best_sz = simulated_annealing(cw, cv, T_start, cooling_rate, min_T, time.time() + time_budget)
        
        all_best_codewords.append(best_cw)
        all_best_sizes.append(best_sz)
        
    # Also try from Tetracode sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tet_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            tet_code.append(digits_to_idx(word))
            
    tet_cov = np.zeros(N, dtype=np.int32)
    for c in tet_code:
        tet_cov[balls[c]] += 1
    tet_code, tet_cov = greedy_reduce(tet_code, tet_cov)
    
    all_best_codewords.append(tet_code)
    all_best_sizes.append(len(tet_code))
    
    if time.time() < deadline - 10:
        best_cw, best_cv, best_sz = simulated_annealing(tet_code[:], tet_cov.copy(), 2.0, 0.995, 1e-4, time.time() + 15)
        all_best_codewords.append(best_cw)
        all_best_sizes.append(best_sz)

    # Find best
    best_idx = all_best_sizes.index(min(all_best_sizes))
    best_result = all_best_codewords[best_idx]
    
    # Final greedy reduce
    final_cov = np.zeros(N, dtype=np.int32)
    for c in best_result:
        final_cov[balls[c]] += 1
    best_result, final_cov = greedy_reduce(best_result, final_cov)
    
    result = [idx_to_digits(c) for c in best_result]
    return result
