"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

def _find_sdr(domains, deadline):
    """
    Find a System of Distinct Representatives for the given list of domains.
    domains[i] is a list of integers.
    Returns a list of chosen integers or None if impossible.
    Uses backtracking. N is small (10).
    """
    n = len(domains)
    assignment = [None] * n
    used = set()
    
    # Sort domains by size to prune earlier
    # We need to keep track of original indices
    indexed_domains = [(len(d), i, d) for i, d in enumerate(domains)]
    indexed_domains.sort()
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
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
        return assignment
    return None

def _is_feasible(span, deadline):
    """
    Check if there exists a window of length 'span' containing distinct multiples.
    Returns (True, assignment) if found, else (False, None).
    """
    # For small spans, limit the search space for L
    # We only need to check L up to a reasonable bound
    max_L = 10000
    
    for L in range(1, max_L + 1):
        if time.time() > deadline:
            return False, None
            
        domains = []
        feasible = True
        for i in range(N):
            d = DIVISORS[i]
            # Multiples of d in [L, L+span]
            # k*d >= L => k >= L/d
            # k*d <= L+span => k <= (L+span)/d
            k_min = (L + d - 1) // d
            k_max = (L + span) // d
            
            if k_min > k_max:
                feasible = False
                break
                
            # Generate the actual values
            vals = []
            for k in range(k_min, k_max + 1):
                vals.append(k * d)
            domains.append(vals)
            
        if not feasible:
            continue
            
        result = _find_sdr(domains, deadline)
        if result is not None:
            return True, result
            
    return False, None

def construct():
    deadline = time.time() + 50.0  # Safety margin
    
    # We know span 33 is feasible (the divisors themselves).
    # Try spans from 9 upward to find the minimum feasible span.
    best_span = 33
    best_assignment = DIVISORS[:]  # Fallback: the divisors themselves
    
    for span in range(9, 34):
        if time.time() > deadline:
            break
        feasible, assignment = _is_feasible(span, deadline)
        if feasible:
            best_span = span
            best_assignment = assignment
            break  # Found the minimum, stop
            
    # Ensure types are correct
    if best_assignment is None:
        return DIVISORS[:]
        
    return [int(x) for x in best_assignment]

# EVOLVE-BLOCK-END
