import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    # Sort indices by divisor value descending to constrain search early
    # Larger divisors have fewer multiples in a small window, so they are more constrained
    sorted_indices = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    sorted_divisors = [divisors[i] for i in sorted_indices]
    
    # LCM of the divisors
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # Primes: 2, 3, 5, 7, 11, 13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    lcm_val = 30030
    
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    # Fallback if search fails or times out
    def get_fallback():
        # Greedy packing around a central value
        target = 100000
        assigned = {}
        used_vals = set()
        for d in sorted_divisors:
            k = target // d
            candidates = []
            for delta in range(-100, 101):
                c = (k + delta) * d
                if c > 0 and c <= 1000000 and c not in used_vals:
                    candidates.append(c)
            # Pick the one closest to target
            if candidates:
                best_c = min(candidates, key=lambda x: abs(x - target))
            else:
                # Find any multiple
                c = d
                while c in used_vals:
                    c += d
                best_c = c
            assigned[d] = best_c
            used_vals.add(best_c)
        
        result = [0] * n
        for i in sorted_indices:
            result[i] = assigned[divisors[i]]
        return result

    # Iterate window size W
    # Theoretical lower bound: We need 10 distinct numbers.
    # Max density is when divisors are small.
    # Let's just start from 1.
    for W in range(1, 1000):
        if time.time() - start_time > time_limit:
            break
            
        # Iterate start L in one period
        # We can step L by 1.
        for L in range(1, lcm_val + 1):
            if time.time() - start_time > time_limit:
                break
            
            # Pre-compute candidates for each divisor in [L, L+W]
            candidates = []
            valid = True
            for d in sorted_divisors:
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                if start_k > end_k:
                    valid = False
                    break
                # Generate list of values
                # List comprehension is fast
                cand_list = [k * d for k in range(start_k, end_k + 1)]
                if not cand_list:
                    valid = False
                    break
                candidates.append(cand_list)
            
            if not valid:
                continue
            
            # Backtracking search
            used = set()
            values = [0] * n
            success = False
            
            def backtrack(idx):
                nonlocal success
                if idx == n:
                    success = True
                    return True
                
                d = sorted_divisors[idx]
                orig_idx = sorted_indices[idx]
                cand_list = candidates[idx]
                
                # Try candidates
                # Heuristic: try values closer to the center of the window?
                # Or just iterate. Since lists are small, iteration is fine.
                for v in cand_list:
                    if v not in used:
                        used.add(v)
                        values[orig_idx] = v
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                return False
            
            if backtrack(0):
                return values[:]
            
    return get_fallback()
