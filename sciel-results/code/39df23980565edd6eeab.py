"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (evaluator kills at 90s)
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = reduce(_lcm, divisors)
    # lcm_val = 30030
    
    def get_candidates(d, s, R):
        """Get all multiples of d in [s, s+R]"""
        k_min = max(1, (s + d - 1) // d)  # ceil(s/d), but at least 1
        k_max = (s + R) // d
        if k_min > k_max:
            return []
        return [k * d for k in range(k_min, k_max + 1)]
    
    def feasible_window(s, R):
        """
        Check if there's a valid assignment in window [s, s+R].
        Returns assignment list or None.
        Uses backtracking with constraint propagation.
        """
        # Get candidates for each divisor
        cands = []
        for i, d in enumerate(divisors):
            c = get_candidates(d, s, R)
            if not c:
                return None
            cands.append(c)
        
        # Quick check: total distinct values must be >= n
        all_vals = set()
        for c in cands:
            all_vals.update(c)
        if len(all_vals) < n:
            return None
        
        # For each value, which divisors can use it
        val_to_divs = {}
        for i, c in enumerate(cands):
            for v in c:
                if v not in val_to_divs:
                    val_to_divs[v] = []
                val_to_divs[v].append(i)
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(cands[i]))
        
        # Backtracking
        assignment = [None] * n
        used_vals = set()
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            div_idx = order[idx]
            d = divisors[div_idx]
            
            for v in cands[div_idx]:
                if v in used_vals:
                    continue
                
                # Check if this assignment is consistent
                # For all remaining unassigned divisors, they must still have at least one candidate
                # (simplified: just try it)
                assignment[div_idx] = v
                used_vals.add(v)
                
                if backtrack(idx + 1):
                    return True
                
                assignment[div_idx] = None
                used_vals.remove(v)
            
            return False
        
        if not backtrack(0):
            return None
        
        # Verify
        for i in range(n):
            if assignment[i] is None or assignment[i] <= 0 or assignment[i] > 1000000:
                return None
            if assignment[i] % divisors[i] != 0:
                return None
        
        # Check distinctness
        if len(set(assignment)) != n:
            return None
        
        return assignment
    
    # Strategy: try small R values first with smart window positioning
    # The window [s, s+R] should contain multiples of each divisor
    # Since LCM = 30030, we can look at s modulo LCM
    
    best_result = None
    best_R = float('inf')
    
    # First, try a range of R values from small to large
    # For each R, try multiple window positions
    
    # Key insight: for a given R, the best window position s should be such that
    # the window contains good coverage of multiples. Try s = 0, 1, 2, ... up to some limit,
    # but also try s aligned with multiples of small divisors.
    
    # Try R from 1 upward, but skip to promising values
    # A lower bound: we need 10 distinct values, each at least their divisor.
    # The minimum possible max-min is at least the 10th smallest multiple minus the 1st.
    
    # Let's try R values: 1, 2, 3, ..., but be smart about window positions
    # For each R, try s in a set of promising positions
    
    max_R_to_try = 500  # Try up to R=500 with limited window positions
    
    for R in range(1, max_R_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Determine window positions to try
        # Try s = 0, and s = d - 1 for each divisor d (so the window starts just before a multiple)
        # Also try s = 1, 2, 3, ... up to min(lcm_val, R + 20)
        
        positions_to_try = set()
        positions_to_try.add(0)
        positions_to_try.add(1)
        for d in divisors:
            positions_to_try.add(d - 1)
            positions_to_try.add(d)
        # Add some random-ish positions
        for s in range(0, min(50, R + 10)):
            positions_to_try.add(s)
        
        # Also try positions that are multiples of small numbers
        for m in [2, 3, 5, 7]:
            for k in range(0, min(20, (R + 10) // m + 1)):
                positions_to_try.add(k * m)
        
        for s in sorted(positions_to_try):
            if time.time() > deadline - 10:
                break
            
            result = feasible_window(s, R)
            if result is not None:
                best_result = result
                best_R = R
                break
        
        if best_result is not None:
            break
    
    # If we found a solution, try to improve by checking nearby R values
    if best_result is not None and best_R > 1:
        # Try R-1, R-2, etc.
        for R in range(best_R - 1, max(0, best_R - 5), -1):
            if time.time() > deadline - 5:
                break
            positions_to_try = set(range(0, min(30, R + 10)))
            for d in divisors:
                positions_to_try.add(d - 1)
                positions_to_try.add(d)
            for s in sorted(positions_to_try):
                if time.time() > deadline - 5:
                    break
                result = feasible_window(s, R)
                if result is not None:
                    best_result = result
                    best_R = R
                    break
            if best_R == R:
                break
    
    # If no solution found, use fallback
    if best_result is None:
        # Fallback: use the divisors themselves (they are distinct and positive)
        # But we need to check they're <= 1000000 (they are)
        best_result = divisors[:]
    
    # Ensure Python ints in a list
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
