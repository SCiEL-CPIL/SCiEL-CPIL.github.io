"""Combined approach: optimized greedy + multi-strategy local search for 3-AP-free sets."""
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute reciprocal weights for fast comparison
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    def can_add(x, S_set, S_arr):
        """Check if x can be added to S without forming a 3-term AP.
        Uses both set for membership and array for iteration."""
        # Case 1: x is the largest term: a, b, x where 2b = a + x
        for a in S_arr:
            if a >= x:
                continue
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S_set:
                    return False
        # Case 2: x is the smallest term: x, b, c where 2b = x + c
        for c in S_arr:
            if c <= x:
                continue
            s = x + c
            if s % 2 == 0:
                b = s // 2
                if b in S_set:
                    return False
        # Case 3: x is the middle term: a, x, c where 2x = a + c
        for a in S_arr:
            if a >= x:
                continue
            c = 2 * x - a
            if 1 <= c <= N and c in S_set:
                return False
        return True

    def get_float_sum(S):
        """Compute floating-point reciprocal sum."""
        return sum(recip[x] for x in S)

    def get_exact_sum(S):
        """Compute exact reciprocal sum using Fraction."""
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total

    def greedy_fill(forbidden, S_start):
        """Fill S by adding elements in ascending order if they don't create a 3-AP."""
        S = set(S_start)
        S_arr = sorted(S)
        for x in all_nums:
            if x in forbidden or x in S:
                continue
            if can_add(x, S, S_arr):
                S.add(x)
                S_arr.append(x)
                S_arr.sort()
        return S

    def greedy_fill_descending(forbidden, S_start):
        """Fill S by adding elements in descending order."""
        S = set(S_start)
        S_arr = sorted(S)
        for x in reversed(all_nums):
            if x in forbidden or x in S:
                continue
            if can_add(x, S, S_arr):
                S.add(x)
                S_arr.append(x)
                S_arr.sort()
        return S

    def local_search(S_init, max_time):
        """Run local search starting from S_init, return best set found.
        Uses Simulated Annealing to escape local optima."""
        S = set(S_init)
        S_arr = sorted(S)
        sum_float = get_float_sum(S)
        best_S = set(S)
        best_sum_float = sum_float
        
        # Simulated Annealing parameters
        T_start = 0.01
        T_end = 1e-6
        cooling = 0.95
        current_T = T_start
        start_ls = time.time()
        
        all_nums_list = list(range(1, N + 1))
        
        while (time.time() - start_ls) < max_time and current_T > T_end:
            current_T *= cooling
            
            move_type = random.random()
            S_candidate = set(S)
            S_cand_arr = sorted(S_candidate)
            valid_move = False
            
            if move_type < 0.3:
                # Add a small random number not in S
                candidates = [x for x in all_nums_list if x not in S and x < 50]
                if not candidates:
                    candidates = [x for x in all_nums_list if x not in S]
                if candidates:
                    y = random.choice(candidates)
                    if can_add(y, S_candidate, S_cand_arr):
                        S_candidate.add(y)
                        valid_move = True
            elif move_type < 0.6:
                # Remove a large random number from S
                if S:
                    candidates = [x for x in S if x > 100]
                    if not candidates:
                        candidates = list(S)
                    x = random.choice(candidates)
                    S_candidate.remove(x)
                    valid_move = True
            else:
                # Swap: Remove x from S, add y not in S
                if S:
                    x_candidates = [x for x in S if x > 50]
                    if not x_candidates:
                        x_candidates = list(S)
                    x = random.choice(x_candidates)
                    
                    y_candidates = [y for y in all_nums_list if y not in S and y < x]
                    if not y_candidates:
                        y_candidates = [y for y in all_nums_list if y not in S]
                    
                    if y_candidates:
                        y = random.choice(y_candidates)
                        S_candidate.remove(x)
                        S_cand_arr = [v for v in S_cand_arr if v != x]
                        if can_add(y, S_candidate, S_cand_arr):
                            S_candidate.add(y)
                            valid_move = True
                        else:
                            valid_move = False
                    else:
                        valid_move = False
            
            if not valid_move:
                continue
                
            sum_new = get_float_sum(S_candidate)
            delta = sum_new - sum_float
            
            if delta > 0:
                S = S_candidate
                sum_float = sum_new
                if sum_new > best_sum_float:
                    best_sum_float = sum_new
                    best_S = set(S)
            else:
                if random.random() < math.exp(delta / current_T):
                    S = S_candidate
                    sum_float = sum_new
        
        # Final polish: Greedy add
        S_final = set(best_S)
        S_final_arr = sorted(S_final)
        for y in range(1, N + 1):
            if y in S_final:
                continue
            if can_add(y, S_final, S_final_arr):
                S_final.add(y)
                S_final_arr.append(y)
                S_final_arr.sort()
        
        final_sum = get_float_sum(S_final)
        if final_sum > best_sum_float:
            best_S = S_final
            best_sum_float = final_sum
            
        return best_S, best_sum_float

    # Step 1: Multiple initializations
    initializations = []
    
    # Greedy ascending
    S_asc = greedy_fill(set(), set())
    initializations.append(S_asc)
    
    # Greedy descending
    S_desc = greedy_fill_descending(set(), set())
    initializations.append(S_desc)
    
    # Randomized greedy: skip some small numbers to allow different structures
    for trial in range(5):
        if (time.time() - start_time) > deadline - 10:
            break
        random.seed(trial * 12345 + 67890)
        S_rand = set()
        S_rand_arr = []
        for x in all_nums:
            # Randomly skip with small probability to explore different paths
            if random.random() < 0.05 and x > 10:
                continue
            if can_add(x, S_rand, S_rand_arr):
                S_rand.add(x)
                S_rand_arr.append(x)
                S_rand_arr.sort()
        initializations.append(S_rand)
    
    # Step 2: Run local search from each initialization
    best_S = None
    best_sum_float = 0.0
    
    time_per_ls = max(5.0, (deadline - 10 - (time.time() - start_time)) / max(1, len(initializations)))
    
    for i, S_init in enumerate(initializations):
        if (time.time() - start_time) > deadline - 5:
            break
        
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 3:
            break
        
        S_ls, sum_ls = local_search(S_init, min(time_per_ls, remaining_time))
        
        if sum_ls > best_sum_float:
            best_sum_float = sum_ls
            best_S = S_ls
    
    # Step 3: Final polish on the best solution
    if best_S is None:
        best_S = S_asc
        best_sum_float = get_float_sum(best_S)
    else:
        # Run one more local search pass on the best
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 5:
            S_final, sum_final = local_search(best_S, remaining)
            if sum_final > best_sum_float:
                best_S = S_final
                best_sum_float = sum_final
    
    # Step 4: Validate - ensure no 3-AP exists
    S_final = set(best_S)
    for x in S_final:
        for y in S_final:
            if x >= y:
                continue
            # Check if x, (x+y)/2, y forms a 3-AP
            s = x + y
            if s % 2 == 0:
                mid = s // 2
                if mid in S_final and mid != x and mid != y:
                    # Found a 3-AP, remove the middle element
                    S_final.remove(mid)
                    break
    
    # Return the result
    return list(S_final)
