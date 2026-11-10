import time
from math import factorial
from functools import lru_cache
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    total_exp = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}
    
    # We only care about 2, 3, 5 for the multipliers
    primes_235 = [2, 3, 5]
    exps_235 = [10, 5, 2]
    
    # Generate all divisors of 2^10 * 3^5 * 5^2
    # Represented as (value, exp_tuple)
    divs_235 = []
    def gen_divs(idx, curr_exp, curr_val):
        if idx == 3:
            divs_235.append((curr_val, tuple(curr_exp)))
            return
        for e in range(exps_235[idx] + 1):
            curr_exp.append(e)
            gen_divs(idx + 1, curr_exp, curr_val * (primes_235[idx] ** e))
            curr_exp.pop()
            
    gen_divs(0, [], 1)
    
    # Map value to exp tuple for quick lookup? Not strictly needed if we iterate.
    # But we need to subtract exponents.
    # Let's store as list of (val, e2, e3, e5)
    
    best_max_factor = N
    best_factors = [N]
    
    def check_partition(rem_e2, rem_e3, rem_e5, forbidden_values, current_max):
        """
        Finds if we can partition 2^rem_e2 * 3^rem_e3 * 5^rem_e5 into distinct factors
        all > 12, not in forbidden_values, minimizing the max factor.
        Returns (success, min_max_factor, factors_list)
        """
        if rem_e2 == 0 and rem_e3 == 0 and rem_e5 == 0:
            return True, current_max, []
            
        # Generate candidates
        # Candidates are divisors of 2^rem_e2 * 3^rem_e3 * 5^rem_e5
        # that are > 12 and not in forbidden_values
        candidates = []
        for e2 in range(rem_e2 + 1):
            for e3 in range(rem_e3 + 1):
                for e5 in range(rem_e5 + 1):
                    val = (2 ** e2) * (3 ** e3) * (5 ** e5)
                    if val > 12 and val not in forbidden_values:
                        candidates.append((val, e2, e3, e5))
        
        if not candidates:
            return False, None, None
            
        # Sort candidates by value to allow binary search on max
        candidates.sort(key=lambda x: x[0])
        
        # We want to find the smallest max_val such that a subset of candidates
        # with values <= max_val can multiply to the target.
        
        # Binary search on the index of the max allowed candidate
        # Note: The max factor of the whole set will be max(current_max, max_candidate_used)
        # We want to minimize this.
        
        # Let's iterate possible max values from the candidate list?
        # Or binary search.
        # The max value used must be one of the candidates.
        
        def can_form_target(target_e2, target_e3, target_e5, max_idx):
            # Uses candidates[0..max_idx]
            # Memoized DFS
            @lru_cache(maxsize=None)
            def dfs(rem2, rem3, rem5, start_idx):
                if rem2 == 0 and rem3 == 0 and rem5 == 0:
                    return []
                
                if start_idx > max_idx:
                    return None
                
                # Pruning: check if remaining sum of exponents is sufficient
                # This is expensive to compute every time, but max_idx is small.
                # Actually, we can just iterate.
                
                for i in range(start_idx, max_idx + 1):
                    val, e2, e3, e5 = candidates[i]
                    if e2 <= rem2 and e3 <= rem3 and e5 <= rem5:
                        res = dfs(rem2 - e2, rem3 - e3, rem5 - e5, i + 1)
                        if res is not None:
                            return [val] + res
                return None
            
            return dfs(target_e2, target_e3, target_e5, 0)

        # Binary search for the smallest max_idx that works
        # The max factor value will be candidates[max_idx][0]
        
        # We can just iterate max_idx from 0 to len(candidates)-1?
        # Since we want to minimize the max, we want the smallest max_idx that yields a solution.
        # Wait, if a solution exists with max_idx=k, it might also exist with max_idx=k+1.
        # We want the *smallest* max factor.
        # So we want the smallest value $V$ such that a solution exists using only candidates $\le V$.
        # This corresponds to the smallest index $k$ such that $can\_form\_target(..., k)$ is True.
        
        # However, it's possible that a solution with a *larger* max factor is required?
        # No, we want to minimize the max.
        # So we find the minimum $V$.
        
        # But wait. If we use a small max $V$, we are restricted to small factors.
        # If we can't form the target with small factors, we must allow larger factors.
        # So we want the smallest $V$ such that a partition exists.
        
        # Let's iterate $k$ from 0 to len(candidates)-1.
        # Return the first one that works?
        # Yes, because candidates are sorted.
        
        for k in range(len(candidates)):
            res_list = can_form_target(rem_e2, rem_e3, rem_e5, k)
            if res_list is not None:
                max_val = candidates[k][0]
                final_max = max(current_max, max_val)
                return True, final_max, res_list
                
        return False, None, None

    # Case 1: 7 and 11 in separate factors
    # Iterate m_7 and m_11
    # m_7 is a divisor of 2^10 3^5 5^2
    # m_11 is a divisor of 2^10 3^5 5^2
    
    # To speed up, we can pre-filter m_7, m_11 such that 7*m_7 > 12 and 11*m_11 > 12
    valid_m7 = [(v, e) for v, e in divs_235 if 7 * v > 12]
    valid_m11 = [(v, e) for v, e in divs_235 if 11 * v > 12]
    
    for v7, e7 in valid_m7:
        f7 = 7 * v7
        for v11, e11 in valid_m11:
            f11 = 11 * v11
            if f7 == f11:
                continue
            
            # Check exponent subtraction
            rem_e2 = exps_235[0] - e7[0] - e11[0]
            rem_e3 = exps_235[1] - e7[1] - e11[1]
            rem_e5 = exps_235[2] - e7[2] - e11[2]
            
            if rem_e2 < 0 or rem_e3 < 0 or rem_e5 < 0:
                continue
                
            current_max = max(f7, f11)
            
            # Check partition of remaining
            success, final_max, rem_factors = check_partition(
                rem_e2, rem_e3, rem_e5, 
                {f7, f11}, 
                current_max
            )
            
            if success:
                if final_max < best_max_factor:
                    best_max_factor = final_max
                    best_factors = [f7, f11] + rem_factors
                    
    # Case 2: 7 and 11 in same factor
    # f = 77 * m_77
    valid_m77 = divs_235 # 77 * m_77 is always > 12
    
    for v77, e77 in valid_m77:
        f77 = 77 * v77
        current_max = f77
        
        rem_e2 = exps_235[0] - e77[0]
        rem_e3 = exps_235[1] - e77[1]
        rem_e5 = exps_235[2] - e77[2]
        
        if rem_e2 < 0 or rem_e3 < 0 or rem_e5 < 0:
            continue
            
        success, final_max, rem_factors = check_partition(
            rem_e2, rem_e3, rem_e5,
            {f77},
            current_max
        )
        
        if success:
            if final_max < best_max_factor:
                best_max_factor = final_max
                best_factors = [f77] + rem_factors

    return best_factors
