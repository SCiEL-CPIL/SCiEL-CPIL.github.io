import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030
    
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    # For each divisor, precompute the "phase" pattern
    # For a window [L, L+W], divisor d has multiples at positions
    # L + ((d - L % d) % d), L + ((d - L % d) % d) + d, etc.
    
    # Strategy: iterate W from 1 upward
    # For each W, iterate L from 1 to LCM
    # Use fast backtracking with constraint propagation
    
    def check_window(L, W):
        """Check if we can assign distinct multiples in [L, L+W] for all divisors."""
        # Build available options for each divisor
        options = []
        for d in divisors:
            # First multiple >= L
            k = (L + d - 1) // d
            first = k * d
            if first > L + W:
                return None
            opts = []
            x = first
            while x <= L + W:
                opts.append(x)
                x += d
            options.append(opts)
        
        # Check if any divisor has no options
        for i in range(n):
            if not options[i]:
                return None
        
        # Sort by number of options (most constrained first)
        order = sorted(range(n), key=lambda i: len(options[i]))
        
        # Backtracking with forward checking
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            i = order[idx]
            for val in options[i]:
                if val not in used:
                    used.add(val)
                    assignment[i] = val
                    
                    # Forward checking: ensure all remaining unassigned divisors
                    # still have at least one available option
                    feasible = True
                    for j in range(idx + 1, n):
                        k = order[j]
                        has_option = False
                        for v in options[k]:
                            if v not in used:
                                has_option = True
                                break
                        if not has_option:
                            feasible = False
                            break
                    
                    if feasible and backtrack(idx + 1):
                        return True
                    
                    used.remove(val)
                    assignment[i] = None
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Iterate W from 1 upward
    for W in range(1, 200):
        if time.time() - start_time > deadline:
            # Return a known valid solution (the divisors themselves)
            return [int(d) for d in divisors]
        
        for L in range(1, LCM + 1):
            if time.time() - start_time > deadline:
                return [int(d) for d in divisors]
            
            result = check_window(L, W)
            if result is not None:
                # Verify the solution
                if len(set(result)) == n:
                    if all(0 < x <= 1000000 for x in result):
                        if all(result[i] % divisors[i] == 0 for i in range(n)):
                            return [int(x) for x in result]
    
    # Fallback: return divisors (valid but not optimal)
    return [int(d) for d in divisors]
