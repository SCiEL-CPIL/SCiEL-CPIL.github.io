import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    
    # Fallback solution: smallest multiples
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # Max 39, Min 6, Width 33
    fallback = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    
    # Precompute offsets for each W on the fly
    # We search W from 9 to 32
    # If we find a solution, return it
    # If we finish all W < 33, return fallback
    
    start_time = time.time()
    
    for W in range(9, 33):
        # Precompute offsets for this W
        # offsets[i][r] = list of o in [0, W] such that (L+o) % d_i == 0 given L % d_i == r
        # (L+o) % d_i == 0 => o % d_i == (-L) % d_i == (-r) % d_i
        offsets = []
        for d in divisors:
            res = [[] for _ in range(d)]
            for o in range(W + 1):
                r = (-o) % d
                res[r].append(o)
            offsets.append(res)
        
        # Initialize residues for L=1
        residues = [1 % d for d in divisors]
        
        max_L = max_val - W
        
        for L in range(1, max_L + 1):
            # Time check
            if time.time() - start_time > 55:
                return fallback
                
            # Build candidates
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                r = residues[i]
                # Get offsets for this residue
                os = offsets[i][r]
                if not os:
                    valid = False
                    break
                # Convert offsets to values
                # L + o
                candidates.append([L + o for o in os])
            
            if not valid:
                # Update residues and continue
                for i in range(len(divisors)):
                    residues[i] = (residues[i] + 1) % divisors[i]
                continue
                
            # Backtracking with MRV
            # Sort indices by candidate list length
            order = sorted(range(len(divisors)), key=lambda i: len(candidates[i]))
            
            assignment = [None] * len(divisors)
            used = set()
            
            # Recursive backtracker
            def backtrack(idx):
                if idx == len(order):
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                        assignment[i] = None
                return False

            if backtrack(0):
                return assignment
                
            # Update residues
            for i in range(len(divisors)):
                residues[i] = (residues[i] + 1) % divisors[i]
                
    return fallback
