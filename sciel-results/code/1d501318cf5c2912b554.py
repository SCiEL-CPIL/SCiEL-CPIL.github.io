"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    nums = list(range(1, N + 1))
    
    # Build conflict graph: conflict[i][j] = True if gcd(i,j) > 1
    # Use bitmasks for efficiency since N=256
    # Actually, let's use adjacency as sets of indices
    
    # Index: 0..255 corresponds to nums[0]=1, nums[1]=2, ..., nums[255]=256
    n = len(nums)
    
    # Build adjacency: adj[i] = set of j where gcd(nums[i], nums[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(nums[i], nums[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy initial solution for lower bound
    def greedy_solution():
        remaining = set(range(n))
        selected = []
        # Sort by value descending, pick if no conflict with already selected
        for idx in sorted(remaining, key=lambda x: nums[x], reverse=True):
            if all(nums[idx] and gcd(nums[idx], nums[s]) == 1 for s in selected):
                selected.append(idx)
        return selected
    
    initial_sel = greedy_solution()
    best_sum = sum(nums[i] for i in initial_sel)
    best_set = set(initial_sel)
    
    # Branch and bound
    # State: current selection (as frozenset for hashing), remaining candidates
    # We'll do recursive B&B with pruning
    
    # Precompute: for each vertex, the set of vertices that conflict with it
    # Upper bound: sum of all remaining vertices that don't conflict with current selection
    
    def upper_bound(current_selected, remaining):
        # Sum of all remaining vertices (optimistic upper bound)
        # Better: only count vertices in remaining that don't conflict with current_selected
        total = 0
        for v in remaining:
            if all(gcd(nums[v], nums[s]) == 1 for s in current_selected):
                total += nums[v]
        return total
    
    # More efficient: precompute conflict masks
    # Since n=256, we can use Python's set operations
    
    def solve_bnb(current_selected, remaining, best_sum_ref):
        nonlocal best_sum, best_set
        
        if time.time() > DEADLINE:
            return
        
        if not remaining:
            s = sum(nums[i] for i in current_selected)
            if s > best_sum:
                best_sum = s
                best_set = set(current_selected)
            return
        
        # Upper bound: sum of all remaining that are compatible with current selection
        compatible_remaining = set()
        for v in remaining:
            if all(gcd(nums[v], nums[s]) == 1 for s in current_selected):
                compatible_remaining.add(v)
        
        if not compatible_remaining:
            s = sum(nums[i] for i in current_selected)
            if s > best_sum:
                best_sum = s
                best_set = set(current_selected)
            return
        
        ub = sum(nums[v] for v in compatible_remaining) + sum(nums[i] for i in current_selected)
        if ub <= best_sum:
            return
        
        # Choose next vertex to branch on: pick the one with highest value among compatible
        # Sort compatible by value descending
        candidates = sorted(compatible_remaining, key=lambda x: nums[x], reverse=True)
        
        for v in candidates:
            if time.time() > DEADLINE:
                return
            
            # Branch 1: include v
            new_selected = current_selected | {v}
            new_remaining = compatible_remaining - {v}
            # Remove all vertices that conflict with v from remaining
            new_remaining = new_remaining - adj[v]
            solve_bnb(new_selected, new_remaining, best_sum)
            
            # Branch 2: exclude v
            new_remaining2 = compatible_remaining - {v}
            solve_bnb(current_selected, new_remaining2, best_sum)
            
            if time.time() > DEADLINE:
                return
    
    # Run B&B
    all_indices = set(range(n))
    solve_bnb(frozenset(), all_indices, best_sum)
    
    # Convert to list of ints
    result = [nums[i] for i in sorted(best_set)]
    
    # Ensure types are correct
    assert isinstance(result, list)
    for x in result:
        assert isinstance(x, int) and not isinstance(x, bool)
    
    return result
# EVOLVE-BLOCK-END
