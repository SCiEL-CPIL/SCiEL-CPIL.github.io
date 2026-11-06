import math
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    # Step 1: Find all conflict sets
    # A conflict is a set C such that sum_{y in C \ {x}} V[y] == V[x] for some x in C.
    # Actually, the condition is: 1/x = sum_{y in S} 1/y.
    # So V[x] = sum_{y in S} V[y].
    # The conflict set is C = {x} U S.
    # Constraint: sum_{v in C} z_v <= |C| - 1.
    
    conflicts = []
    
    # Pre-compute suffix sums or just use DFS
    # For each x, find all subsets S of [1..n]\{x} summing to V[x].
    
    def find_subsets(target, available_indices):
        """
        Find all subsets of available_indices whose V values sum to target.
        Returns a list of lists of indices.
        """
        results = []
        # Sort available indices by V value descending to prune faster?
        # Or ascending?
        # Let's sort by index to be deterministic.
        # Pruning: if sum of remaining V < target, return.
        
        # We can use a recursive helper
        def dfs(start, current_sum, current_path):
            if current_sum == target:
                if current_path: # Must be non-empty? Yes, subset of OTHER elements.
                    results.append(current_path[:])
                return
            if current_sum > target:
                return
            if start >= len(available_indices):
                return
            
            # Pruning: max possible sum from start onwards
            # Pre-compute suffix sums of V for available_indices?
            # Since N is small, simple sum is okay.
            remaining = available_indices[start:]
            max_add = sum(V[idx] for idx in remaining)
            if current_sum + max_add < target:
                return
            
            # Option 1: Include available_indices[start]
            idx = available_indices[start]
            dfs(start + 1, current_sum + V[idx], current_path + [idx])
            
            # Option 2: Exclude available_indices[start]
            dfs(start + 1, current_sum, current_path)
            
        dfs(0, 0, [])
        return results

    for x in range(1, n + 1):
        target = V[x]
        available = [i for i in range(1, n + 1) if i != x]
        # Optimization: If sum of all available V < target, no solution.
        if sum(V[i] for i in available) < target:
            continue
        
        subsets = find_subsets(target, available)
        for S in subsets:
            C = [x] + S
            conflicts.append(C)
            
    # Remove duplicate conflicts? 
    # It's possible to have same set C from different x? No, x is unique in C as target.
    # But different S for same x produce different C.
    # Are there duplicate C's? 
    # If C = {x} U S1 and C = {x} U S2, then S1=S2.
    # So no duplicates for same x.
    # Can C be generated for different x? 
    # If x1 is in C, and x2 is in C.
    # If C is a conflict for x1, then V[x1] = sum_{y in C\{x1}} V[y].
    # If C is a conflict for x2, then V[x2] = sum_{y in C\{x2}} V[y].
    # This would imply a specific structure. Unlikely to have duplicates, but safe to check.
    
    unique_conflicts = []
    seen = set()
    for C in conflicts:
        key = tuple(sorted(C))
        if key not in seen:
            seen.add(key)
            unique_conflicts.append(C)
            
    conflicts = unique_conflicts

    # Step 2: Formulate ILP
    # Maximize sum z_i
    # c = -1 for all i (since milp minimizes)
    c = -np.ones(n)
    
    # Constraints:
    # For each conflict C_k: sum_{i in C_k} z_i <= |C_k| - 1
    # This is a linear constraint.
    
    if not conflicts:
        # No conflicts, select all
        return list(range(1, n + 1))
        
    # Number of constraints M, Number of variables N
    M = len(conflicts)
    N = n
    
    A_ub = np.zeros((M, N))
    b_ub = np.zeros(M)
    
    for i, C in enumerate(conflicts):
        for idx in C:
            A_ub[i, idx - 1] = 1.0
        b_ub[i] = len(C) - 1
        
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality
    integrality = np.ones(N)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), bounds=bounds, integrality=integrality)
        if res.status == 0 and res.x is not None:
            # res.x is the solution vector
            # Convert to set of indices
            selected = []
            for i in range(N):
                if res.x[i] > 0.5:
                    selected.append(i + 1)
            if selected:
                # Verify validity just in case
                # (Optional, but good for safety)
                # If valid, return
                return sorted(selected)
            else:
                # Should not happen if maximizing sum of positive vars
                pass
        else:
            # Fallback if solver fails
            pass
    except Exception:
        pass

    # Fallback: Use the current greedy logic or a simple heuristic
    # Since we have the conflicts, we can do a greedy packing
    # Or just return a known good set
    # Let's try a simple greedy: add elements if they don't violate constraints
    # Constraint: For each C, not all in C.
    
    S = set()
    # Try adding in order 1..24? Or 24..1?
    # Larger numbers have smaller reciprocals, so they are "safer" to include?
    # Actually, small numbers are more likely to be part of sums.
    # Let's try descending order.
    for i in range(n, 0, -1):
        candidate = S | {i}
        valid = True
        for C in conflicts:
            # Check if C is subset of candidate
            # If C subset candidate, then all z in C are 1, which is forbidden.
            # So we must ensure C is NOT a subset of candidate.
            # This means for every C, there must be at least one element in C \ candidate.
            # Equivalent: C is not a subset of candidate.
            if all(x in candidate for x in C):
                valid = False
                break
        if valid:
            S.add(i)
            
    if S:
        return sorted(S)
        
    # If even greedy fails (should not happen, single element is always valid)
    return [1]
