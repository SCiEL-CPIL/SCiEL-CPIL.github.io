import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    # LCM of divisors for periodicity
    def lcm(a, b):
        return a * b // __import__('math').gcd(a, b)
    
    from math import gcd
    LCM = 1
    for d in divisors:
        LCM = lcm(LCM, d)
    # LCM = 30030
    
    best_val = float('inf')
    best_result = None
    
    # Try increasing window widths
    for W in range(1, 500):
        if time.time() > deadline - 5.0:
            break
        
        # For this W, find L values in [0, LCM) where each divisor has a multiple in [L, L+W]
        # and we can assign distinct multiples
        
        # Precompute for each divisor d, the set of L mod d that work
        # For divisor d: [L, L+W] contains a multiple of d iff (d - L % d) % d <= W
        # i.e., L % d >= d - W (if W < d) or always (if W >= d)
        
        # We'll iterate L over a subset. Since LCM is 30030, we can check all L in [0, LCM)
        # but that's 30030 iterations per W, which is fine.
        
        # For efficiency, only check L where all divisors have at least one multiple
        # Precompute valid residues for each divisor
        valid_residues = []
        for d in divisors:
            residues = set()
            if W >= d:
                residues = set(range(d))
            else:
                # L % d in [d-W, d-1] or {0}
                residues = set(range(d - W, d)) | {0}
            valid_residues.append(residues)
        
        # Now find L in [0, LCM) such that for all i, L % divisors[i] in valid_residues[i]
        # We can do this by iterating L and checking, or use a smarter method.
        # 30030 iterations is fast.
        
        candidate_Ls = []
        for L in range(LCM):
            ok = True
            for i, d in enumerate(divisors):
                if L % d not in valid_residues[i]:
                    ok = False
                    break
            if ok:
                candidate_Ls.append(L)
        
        if not candidate_Ls:
            continue
        
        # For each candidate L, try to assign distinct multiples
        # For divisor d and window [L, L+W], the multiples are:
        #   start = ((L + d - 1) // d) * d  # ceil(L/d)*d
        #   then start, start+d, ... up to L+W
        # Since W < 39 for small W, there's at most 1 or 2 multiples per divisor
        
        found = False
        for L in candidate_Ls:
            if time.time() > deadline - 5.0:
                break
            
            # Get possible multiples for each divisor
            choices = []
            for d in divisors:
                start = ((L + d - 1) // d) * d
                mults = []
                m = start
                while m <= L + W:
                    if m <= MAX_V:
                        mults.append(m)
                    m += d
                if mults:
                    choices.append(mults)
                else:
                    break
            else:
                # All divisors have at least one multiple, try to assign distinct ones
                # Backtracking
                used = set()
                assignment = [None] * n
                
                def backtrack(idx):
                    if idx == n:
                        return True
                    for m in choices[idx]:
                        if m not in used:
                            used.add(m)
                            assignment[idx] = m
                            if backtrack(idx + 1):
                                return True
                            used.remove(m)
                            assignment[idx] = None
                    return False
                
                if backtrack(0):
                    # Found a valid assignment
                    val = max(assignment) - min(assignment)
                    if val < best_val:
                        best_val = val
                        best_result = assignment[:]
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: just return the divisors themselves (they are distinct and valid)
        best_result = divisors[:]
    
    # Ensure types are correct
    return [int(x) for x in best_result]
