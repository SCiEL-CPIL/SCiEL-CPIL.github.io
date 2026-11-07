"""Improved program: exploit LCM periodicity + backtracking for span minimization."""
import time
import math
from itertools import product

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAXV = 1000000
N = len(DIVISORS)

def _compute_lcm():
    """Compute LCM of all divisors."""
    result = 1
    for d in DIVISORS:
        result = result * d // math.gcd(result, d)
    return result

LCM = _compute_lcm()  # 32730

DEADLINE = time.time() + 85  # safety margin vs 90s wall-clock


def _backtrack_assign(candidates, index=0, assigned=None, used=None):
    """
    Backtrack to assign distinct values from candidates[i] to divisor i.
    candidates: list of lists, candidates[i] = list of possible values for divisor i
    Returns the assignment list or None.
    """
    if assigned is None:
        assigned = [None] * N
        used = set()
    
    if index == N:
        return assigned
    
    # Try each candidate for this divisor
    for val in candidates[index]:
        if val not in used:
            used.add(val)
            assigned[index] = val
            result = _backtrack_assign(candidates, index + 1, assigned, used)
            if result is not None:
                return result
            used.remove(val)
            assigned[index] = None
    
    return None


def _check_span_feasible(S, max_L=None):
    """
    Check if there exists a window [L, L+S] where we can assign distinct multiples.
    Returns (L, reps) if found, else None.
    """
    if max_L is None:
        max_L = LCM  # periodicity
    
    for L in range(1, min(max_L, MAXV - S) + 1):
        # For each divisor, find all multiples in [L, L+S]
        candidates = []
        feasible = True
        for i in range(N):
            d = DIVISORS[i]
            # First multiple of d >= L
            first_k = (L + d - 1) // d
            first_val = first_k * d
            # Last multiple of d <= L+S
            last_val = L + S
            
            if first_val > last_val:
                feasible = False
                break
            
            # Generate all multiples in [L, L+S]
            vals = []
            v = first_val
            while v <= last_val:
                vals.append(v)
                v += d
            candidates.append(vals)
        
        if not feasible:
            continue
        
        # Use backtracking to find distinct assignment
        # Sort by domain size for efficiency
        indices = list(range(N))
        indices.sort(key=lambda i: len(candidates[i]))
        
        # Reorder candidates
        sorted_candidates = [candidates[i] for i in indices]
        
        result = _backtrack_assign(sorted_candidates)
        if result is not None:
            # Map back to original indices
            reps = [None] * N
            for orig_idx, val in zip(indices, result):
                reps[orig_idx] = val
            return (L, reps)
    
    return None


def _local_search(reps, max_iterations=500):
    """Local search: try perturbing each representative by +/- k*d for small k."""
    best_span = max(reps) - min(reps)
    best_reps = reps[:]
    current_reps = reps[:]
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            d = DIVISORS[i]
            for k in [-2, -1, 1, 2]:
                new_val = current_reps[i] + k * d
                if new_val <= 0 or new_val > MAXV:
                    continue
                new_reps = current_reps[:]
                new_reps[i] = new_val
                if len(set(new_reps)) != len(new_reps):
                    continue
                span = max(new_reps) - min(new_reps)
                if span < best_span:
                    best_span = span
                    best_reps = new_reps[:]
                    current_reps = new_reps[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return (best_span, best_reps)


def construct():
    best_span = float('inf')
    best_reps = None
    
    # Strategy: Search for minimum feasible span S
    # Start from a reasonable lower bound and increase
    # The theoretical minimum span is at least N-1 = 9 (10 distinct integers)
    # But practical minimum depends on divisor structure
    
    # Quick estimate: try spans from 10 upward
    # For each span, check feasibility using periodicity
    
    start_S = 10
    # Upper bound: in the worst case, span could be large, but let's cap reasonably
    # If we can't find a solution with span <= 200 in the first LCM period, 
    # we'll fall back to scanning
    
    for S in range(start_S, 300):
        if time.time() > DEADLINE - 5:
            break
        
        result = _check_span_feasible(S)
        if result is not None:
            L, reps = result
            # Verify all constraints
            valid = True
            for i in range(N):
                if reps[i] <= 0 or reps[i] > MAXV:
                    valid = False
                    break
                if reps[i] % DIVISORS[i] != 0:
                    valid = False
                    break
            if not valid:
                continue
            if len(set(reps)) != len(reps):
                continue
            
            span = max(reps) - min(reps)
            if span <= S:  # Should be exactly S or less
                best_span = span
                best_reps = reps[:]
                break  # Found minimum feasible span
        
        # Progress check
        if S % 20 == 0 and time.time() > DEADLINE - 10:
            break
    
    # If we found a solution, try to improve with local search
    if best_reps is not None:
        span, reps = _local_search(best_reps, max_iterations=1000)
        if span < best_span:
            best_span = span
            best_reps = reps
    
    # Fallback: if no solution found via span search, do a broader scan
    if best_reps is None:
        # Scan L values and find best window
        def _smallest_multiple_at_least(d, L):
            return ((L + d - 1) // d) * d
        
        for L in range(1, 5000):
            if time.time() > DEADLINE - 5:
                break
            reps = []
            valid = True
            for d in DIVISORS:
                m = _smallest_multiple_at_least(d, L)
                if m > MAXV:
                    valid = False
                    break
                reps.append(m)
            if not valid:
                continue
            if len(set(reps)) != len(reps):
                continue
            span = max(reps) - min(reps)
            if span < best_span:
                best_span = span
                best_reps = reps[:]
        
        if best_reps is not None:
            span, reps = _local_search(best_reps, max_iterations=500)
            if span < best_span:
                best_span = span
                best_reps = reps
    
    if best_reps is None:
        # Last resort fallback
        return DIVISORS[:]
    
    # Ensure we return Python ints
    return [int(x) for x in best_reps]
# EVOLVE-BLOCK-END
