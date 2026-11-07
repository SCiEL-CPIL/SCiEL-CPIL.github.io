import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM_ALL = 1
for d in DIVISORS:
    _LCM_ALL = _lcm(_LCM_ALL, d)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # The span must be at least N-1 = 9
    # We search for the minimum span W
    # Due to periodicity with period _LCM_ALL, we only need to check L in [0, _LCM_ALL)
    
    # Pre-compute some info to speed up domain generation?
    # Not strictly necessary, but let's keep it simple.
    
    best_assignment = None
    best_span = None
    
    # We know span 33 is feasible (the divisors themselves are distinct and within span 39-6=33? 
    # Wait, max(39)-min(6)=33. Yes. So W=33 is a valid upper bound.
    # Actually, we can likely do much better.
    
    for W in range(9, 34):
        if time.time() > deadline:
            break
            
        # Iterate over all possible start positions L modulo _LCM_ALL
        # Note: L can be any integer. The pattern of multiples repeats every _LCM_ALL.
        # So checking L in 0.._LCM_ALL-1 is sufficient.
        for L in range(_LCM_ALL):
            if time.time() > deadline:
                break
                
            # Generate domains for each divisor
            domains = []
            feasible = True
            for i in range(N):
                d = DIVISORS[i]
                # Multiples of d in [L, L+W]
                # k*d >= L => k >= ceil(L/d)
                # k*d <= L+W => k <= floor((L+W)/d)
                
                k_min = (L + d - 1) // d
                k_max = (L + W) // d
                
                if k_min > k_max:
                    feasible = False
                    break
                
                # Create list of values
                # We store the values themselves
                vals = [k * d for k in range(k_min, k_max + 1)]
                domains.append(vals)
            
            if not feasible:
                continue
            
            # Try to find a System of Distinct Representatives (SDR)
            # Use backtracking with ordering by domain size for efficiency
            # Sort indices by domain size
            indexed_domains = [(len(domains[i]), i, domains[i]) for i in range(N)]
            indexed_domains.sort()
            
            assignment = [None] * N
            used = set()
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == N:
                    return True
                _, orig_idx, domain = indexed_domains[idx]
                for val in domain:
                    if val not in used:
                        used.add(val)
                        assignment[orig_idx] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                best_assignment = assignment
                best_span = W
                break # Found minimum span for this W, but we are iterating W upwards, so this is the global min
        
        if best_assignment is not None:
            break # Found the minimum span
            
    if best_assignment is None:
        # Fallback: return the divisors themselves (span 33)
        # Ensure they are distinct and valid
        # Divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39. All distinct.
        # Max 39, Min 6. Span 33.
        return [int(x) for x in DIVISORS]
        
    return [int(x) for x in best_assignment]
