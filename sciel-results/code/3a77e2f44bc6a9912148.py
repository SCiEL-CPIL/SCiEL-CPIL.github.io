import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    period = 30030
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We search for W from 9 upwards
    # The minimum possible range for 10 distinct integers is 9 (e.g., 1..10)
    # But due to divisibility, it might be larger.
    
    for W in range(9, 200):
        if time.time() > deadline:
            break
        
        # Pre-compute valid L masks for each divisor
        # valid_L_mask[L] is True if [L, L+W] contains at least one multiple of d
        # L ranges from 1 to period
        
        # We can use a boolean list of size period + 1 (index 0 unused or used for L=0)
        # But L=0 is not allowed (positive integers).
        # Let's use 1..period
        
        # Optimization: Instead of full array, we can just generate the valid Ls for the 
        # most constrained divisor and check others?
        # No, intersection is safer.
        
        # Let's try a different approach: 
        # For a given W, the "gaps" between multiples of d are d.
        # If W >= d, all L are valid for d.
        # If W < d, only specific L ranges are valid.
        
        # Let's just iterate L from 1 to period and check quickly.
        # The check is: for all d, ceil(L/d)*d <= L+W.
        
        # To speed up, we can pre-calculate the "bad" Ls?
        # No, let's stick to the mask idea if it's fast enough to build.
        
        # Building mask for d:
        # A multiple m is in [L, L+W] iff L <= m <= L+W.
        # So for a specific m, L can be in [m-W, m].
        # We want the union of these intervals for all m.
        
        # Let's just iterate L and check. It's O(period * n) per W.
        # period=30030, n=10. 300k ops per W.
        # If we test 50 Ws, that's 15M ops. Should be fast in Python? Maybe 1-2s.
        # The matching is the expensive part, but we only run it on valid Ls.
        
        found_solution = False
        for L in range(1, period + 1):
            if time.time() > deadline:
                found_solution = True # Break out to return fallback or previous? 
                # Actually, if we break, we haven't found a solution for this W.
                # We should break the W loop too.
                break
            
            # Check if all divisors have at least one candidate
            valid = True
            candidates = []
            for d in divisors:
                # First multiple >= L
                # k = ceil(L/d)
                # m = k*d
                # Check if m <= L+W
                
                # Integer arithmetic for ceil(L/d)
                k = (L + d - 1) // d
                m_start = k * d
                if m_start > L + W:
                    valid = False
                    break
                
                # Count candidates
                # Last multiple <= L+W
                # k_end = floor((L+W)/d)
                # m_end = k_end*d
                
                # Number of candidates = (m_end - m_start) // d + 1
                # We don't strictly need the count for the check, just existence.
                # But we need the list for matching.
                
                # Let's build the list if valid so far
                # But if valid becomes False, we discard.
                
                # To avoid building lists if invalid, we can just check existence first.
                # But we need the list for matching.
                
                # Let's just build the list.
                m_end = ((L + W) // d) * d
                num_cands = (m_end - m_start) // d + 1
                
                # If num_cands == 0, it's invalid (but m_start <= L+W ensures >=1)
                # So we just append.
                
                # Store (start, d, count) to generate on fly in backtracking?
                # Or store the list?
                # List is small (W/d + 1).
                
                c_list = list(range(m_start, m_end + 1, d))
                candidates.append(c_list)
            
            if not valid:
                continue
            
            # Check for perfect matching
            # Sort by length
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            sorted_cands = [(i, candidates[i]) for i in order]
            
            used = set()
            assignment = [None] * n
            
            # Backtracking
            # Use a non-recursive or optimized recursive approach
            # Since n=10, recursion is fine.
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                orig_idx, c_list = sorted_cands[idx]
                for val in c_list:
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                result = [int(assignment[i]) for i in range(n)]
                return result
        
        # If we finished L loop without finding solution, try next W
        
    # Fallback
    return [int(d) for d in divisors]
