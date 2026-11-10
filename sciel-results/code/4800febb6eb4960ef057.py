"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    lcm_1_12 = 27720  # lcm(1..12)
    
    # Precompute for each i: the "density" - how many multiples of i in any interval of length L
    # For a given L, the number of multiples of i in (m, m+L] is either floor(L/i) or ceil(L/i)
    # The rarest constraint is the one with fewest candidates.
    # Order constraints by ascending candidate count for a typical L to prune early.
    # For L=144: candidates for i are ~L/i = 144/i
    # i=12: 12, i=11: 13, i=10: 14, i=9: 16, i=8: 18, i=7: 20, i=6: 24, i=5: 28, i=4: 36, i=3: 48, i=2: 72, i=1: 144
    # So process in order: 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    order = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    def can_assign(m, L):
        """Check if we can assign distinct b_i in (m, m+L] with i | b_i using bitmask matching."""
        end_val = m + L
        
        # Build candidate bitmasks for each i
        # We'll use a dictionary: value -> bitmask of which i's can use it
        # And for each i, a list of values it can use
        
        # First pass: check if any i has 0 candidates -> immediate fail
        # Also build the value->i mapping
        val_to_is = {}  # value -> list of i's that can use it
        i_to_vals = {}  # i -> list of values
        
        for i in range(1, n + 1):
            first_mult = (m // i + 1) * i
            vals = []
            for v in range(first_mult, end_val + 1, i):
                vals.append(v)
                if v not in val_to_is:
                    val_to_is[v] = []
                val_to_is[v].append(i)
            i_to_vals[i] = vals
            if not vals:
                return False
        
        # Check: if any i has fewer candidates than needed... well we need at least 1 each
        # But more importantly, check Hall's condition for small subsets
        # For now, just do the matching with constraint ordering
        
        # For efficiency, use a simple greedy + backtracking approach
        # Assign in order of rarity (fewest candidates first)
        # Use a set of used values
        
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in i_to_vals[i]:
                if v in used:
                    continue
                # Check if this assignment is "safe" - no future constraint is left with 0 options
                # Quick check: for each unassigned i', count available vals
                # This is expensive, so only do it for the last few
                if idx >= n - 3:
                    feasible = True
                    for j in range(idx + 1, n):
                        i2 = order[j]
                        has_avail = False
                        for v2 in i_to_vals[i2]:
                            if v2 != v and v2 not in used:
                                has_avail = True
                                break
                        if not has_avail:
                            feasible = False
                            break
                    if not feasible:
                        continue
                
                used.add(v)
                if backtrack(idx + 1):
                    return True
                used.remove(v)
            return False
        
        return backtrack(0)
    
    def f(m):
        """Compute f(12, m) = least positive L such that assignment exists."""
        # Binary search on L: can_assign is monotone in L (if it works for L, it works for L+1)
        # Actually, is it monotone? If (m, m+L] has a valid assignment, then (m, m+L+1] also does
        # (we can use the same assignment). Yes, monotone.
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_f = 0
    
    # Search all 27720 residue classes
    for m in range(0, lcm_1_12):
        if time.time() > deadline - 2.0:
            break
        
        val = f(m)
        if val > best_f:
            best_f = val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
