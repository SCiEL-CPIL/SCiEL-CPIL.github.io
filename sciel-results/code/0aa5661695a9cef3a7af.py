import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    
    # Calculate LCM of divisors to determine the period of the pattern
    # LCM(6, 10, 14, 15, 21, 22, 26, 33, 35, 39)
    # Primes: 2, 3, 5, 7, 11, 13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    period = 30030
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Sort divisors by index to keep track of original positions for output
    # But for matching, we can just work with the list as is or sorted by constraint tightness
    # Let's keep original indices for the result
    
    # We search for W from 1 upwards
    for W in range(1, 1000):
        # Check time
        if time.time() > deadline:
            # If we run out of time, we should ideally return a valid solution found earlier
            # But since we iterate W from 1, the first solution is the best.
            # If we haven't found one, we are in trouble.
            # However, a solution should be found quickly.
            # To be safe, if we break, we can't return a valid solution if best is None.
            # But the problem says "Any broken rule makes the whole result invalid".
            # So we must return a valid list.
            # We assume a solution exists for small W.
            # If we hit deadline before finding any, we have a problem.
            # But W=1..100 is very likely to contain a solution.
            # Let's assume we find one.
            pass
        
        # Iterate L from 1 to period
        # We can optimize L range: 
        # The window [L, L+W] must contain at least one multiple for each d.
        # This is automatically handled by the candidate generation being non-empty.
        
        for L in range(1, period + 1):
            if time.time() > deadline:
                # This is a hard stop. We can't return invalid.
                # We should probably break out and return a fallback if we have one.
                # But we don't have a fallback yet.
                # Let's hope it doesn't happen.
                # To be robust, we can store the best valid solution found in previous W iterations?
                # No, we are looking for the MINIMUM W.
                # If we find a solution for W, it's the global minimum because we iterate W ascending.
                # So we can return immediately upon finding the first valid solution for a given W.
                # Wait, is the first L for a given W the best?
                # The score is max - min = W.
                # So ANY solution for a specific W has the same score W.
                # So we can return the first valid assignment found for the smallest W.
                
                # So we can just return immediately when we find a match.
                break
            
            # Generate candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # Find multiples of d in [L, L+W]
                # First multiple >= L: ceil(L/d) * d
                k_start = (L + d - 1) // d
                m_start = k_start * d
                # Last multiple <= L+W: floor((L+W)/d) * d
                m_end = ((L + W) // d) * d
                
                c_list = []
                if m_start <= m_end:
                    # Add multiples
                    m = m_start
                    while m <= m_end:
                        c_list.append(m)
                        m += d
                else:
                    # No multiple in range
                    valid = False
                    break
                
                # If number of candidates is 0, invalid
                if not c_list:
                    valid = False
                    break
                candidates.append(c_list)
            
            if not valid:
                continue
            
            # Check for perfect matching
            # We need to assign one candidate from each list to each divisor such that all assigned values are distinct.
            # Use backtracking.
            # Sort divisors by number of candidates (ascending) to speed up backtracking
            # But we need to map back to original indices.
            # indices_sorted = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Simple backtracking with used set
            used = set()
            
            def backtrack(i):
                if i == n:
                    return True
                for val in candidates[i]:
                    if val not in used:
                        used.add(val)
                        if backtrack(i + 1):
                            return True
                        used.remove(val)
                return False
            
            # Optimization: Check if total candidates >= n (pigeonhole)
            total_cands = sum(len(c) for c in candidates)
            if total_cands < n:
                continue
            
            # Try matching
            # To make backtracking faster, sort by constraint tightness
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # We need to be careful with the 'used' set and the 'candidates' list indexing if we reorder.
            # Let's create a list of (original_index, candidate_list) sorted by len(candidate_list)
            sorted_cands = [(i, candidates[i]) for i in order]
            
            def backtrack_sorted(idx):
                if idx == n:
                    return True
                orig_idx, c_list = sorted_cands[idx]
                for val in c_list:
                    if val not in used:
                        used.add(val)
                        if backtrack_sorted(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack_sorted(0):
                # Construct the result
                result = [0] * n
                # We need to know which value was assigned to which original index.
                # The backtracking above doesn't store the assignment, it just returns True.
                # We need to modify it to store the assignment or just re-run to get it?
                # Better to store it during backtracking.
                
                used = set()
                assignment = [None] * n
                
                def backtrack_store(idx):
                    if idx == n:
                        return True
                    orig_idx, c_list = sorted_cands[idx]
                    for val in c_list:
                        if val not in used:
                            used.add(val)
                            assignment[orig_idx] = val
                            if backtrack_store(idx + 1):
                                return True
                            used.remove(val)
                            assignment[orig_idx] = None
                    return False
                
                if backtrack_store(0):
                    # Verify
                    # Distinct? Yes, by construction.
                    # Divisible? Yes, from candidates.
                    # <= 1000000?
                    # Max val is L+W. L <= 30030, W < 1000. Max ~ 31000 < 1000000.
                    # So valid.
                    return assignment
            
        # If we finished the L loop for this W without finding a solution, try next W
        # The 'break' in the time check inside the L loop breaks out of the L loop, 
        # but we need to handle the time check properly.
        # If time.time() > deadline, we should probably abort and return a known valid solution if we had one?
        # But we return immediately on first success.
        # If we run out of time before finding ANY solution, we crash or return invalid.
        # Given the small search space, this is unlikely.
        
    # Fallback: return the divisors themselves (which is valid but not optimal)
    # This ensures we always return a valid list if the search fails or times out.
    return divisors[:]
