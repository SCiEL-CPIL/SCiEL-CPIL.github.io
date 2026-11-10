"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from functools import lru_cache

# EVOLVE-BLOCK-START
import time as _time

def construct():
    """
    Find a 3-AP-free subset of [1, 256] maximizing the sum of reciprocals.
    Uses branch-and-bound with bitset operations and tight upper bounds.
    """
    deadline = _time.time() + 80.0  # Safety margin before 90s kill
    
    n = 256
    # Vertices are numbers 1..256, indexed 0..255
    # Weight of vertex i is 1/(i+1)
    weights = [Fraction(1, i+1) for i in range(n)]
    
    # Build conflict graph: two vertices conflict if they can be part of a 3-AP
    # A 3-AP is (a, b, c) with a < b < c and b-a = c-b, i.e., a+c = 2b
    # Two numbers x, y conflict if there exists z such that {x,y,z} forms a 3-AP
    # Cases:
    #   1. x, y are endpoints: z = (x+y)/2 must be integer and in [1,n]
    #   2. x, y are (a,b) or (b,c): z = 2y-x or z = 2x-y must be in [1,n]
    
    # conflict_mask[i] is a bitset of all j such that i and j conflict
    conflict_mask = [0] * n
    for i in range(n):
        x = i + 1
        for j in range(n):
            if i == j:
                continue
            y = j + 1
            # Check if x, y can be part of a 3-AP with some z
            # Case 1: x and y are endpoints, z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= n and z != x and z != y:
                    conflict_mask[i] |= (1 << j)
                    continue
            # Case 2: x is first, y is second, z = 2y - x
            z = 2 * y - x
            if 1 <= z <= n and z != x and z != y:
                conflict_mask[i] |= (1 << j)
                continue
            # Case 3: y is first, x is second, z = 2x - y
            z = 2 * x - y
            if 1 <= z <= n and z != x and z != y:
                conflict_mask[i] |= (1 << j)
    
    # Precompute: for each vertex, the set of vertices it conflicts with
    # Also precompute sorted order by weight (descending) for branching
    
    # Branch and bound
    # State: (available_mask, current_sum, selected_count)
    # We want to maximize current_sum + (max possible additional sum)
    
    # Upper bound: sum of weights of all available vertices (relaxation)
    # Tighter: greedy fractional relaxation
    
    # For speed, use float weights for bounding, exact Fraction for final answer
    float_weights = [1.0 / (i + 1) for i in range(n)]
    
    # Sort vertices by weight descending for branching order
    order = sorted(range(n), key=lambda i: -float_weights[i])
    
    # Precompute prefix sums of float weights in sorted order for quick upper bounds
    # Actually, we need upper bound on available vertices, not just prefix
    
    # Better approach: maintain available mask, compute upper bound as sum of weights
    # of available vertices that don't conflict with each other... that's the problem itself.
    # Simple upper bound: sum of all available weights
    
    # Let's use a different strategy: greedy construction + local search
    
    # First, get a good initial solution with greedy
    def greedy_solution():
        selected = set()
        available = [True] * n
        for i in order:
            if available[i]:
                selected.add(i)
                # Remove all conflicts
                mask = conflict_mask[i]
                for j in range(n):
                    if mask & (1 << j):
                        available[j] = False
        return selected
    
    best_selected = greedy_solution()
    best_weight = sum(weights[i] for i in best_selected)
    best_float_weight = sum(float_weights[i] for i in best_selected)
    
    # Local search: try to swap vertices to improve
    def local_search(selected):
        selected_set = set(selected)
        improved = True
        while improved:
            improved = False
            # Try removing each selected vertex and adding a better one
            for i in list(selected_set):
                # Try removing i
                remaining = selected_set - {i}
                # Find best vertex to add
                best_add = None
                best_add_weight = weights[i]  # Must beat removing i
                for j in range(n):
                    if j in selected_set:
                        continue
                    # Check if j conflicts with any in remaining
                    if any(conflict_mask[j] & (1 << k) for k in remaining):
                        continue
                    if weights[j] > best_add_weight:
                        best_add_weight = weights[j]
                        best_add = j
                if best_add is not None:
                    selected_set.remove(i)
                    selected_set.add(best_add)
                    improved = True
        return selected_set
    
    best_selected = local_search(best_selected)
    best_weight = sum(weights[i] for i in best_selected)
    best_float_weight = sum(float_weights[i] for i in best_selected)
    
    # Branch and bound
    def branch_and_bound(available_mask, current_weight, current_float_weight, selected):
        nonlocal best_weight, best_float_weight, best_selected
        
        if _time.time() > deadline:
            return
        
        # Upper bound: current + sum of all available weights
        # Compute sum of available weights
        avail_sum = 0.0
        m = available_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            avail_sum += float_weights[idx]
            m ^= lb
        
        if current_float_weight + avail_sum <= best_float_weight + 1e-15:
            return
        
        # If no available vertices, update best
        if available_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_float_weight = current_float_weight
                best_selected = set(selected)
            return
        
        # Pick the highest-weight available vertex
        # Find max weight available
        best_i = -1
        best_w = -1
        m = available_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if float_weights[idx] > best_w:
                best_w = float_weights[idx]
                best_i = idx
            m ^= lb
        
        if best_i == -1:
            return
        
        # Branch 1: include best_i
        new_selected = list(selected)
        new_selected.append(best_i)
        new_available = available_mask & ~(1 << best_i)
        # Remove conflicts
        new_available &= ~conflict_mask[best_i]
        branch_and_bound(new_available, current_weight + weights[best_i], 
                        current_float_weight + float_weights[best_i], new_selected)
        
        if _time.time() > deadline:
            return
        
        # Branch 2: exclude best_i
        branch_and_bound(available_mask & ~(1 << best_i), current_weight, 
                        current_float_weight, selected)
    
    # Initialize with full available mask
    full_mask = (1 << n) - 1
    branch_and_bound(full_mask, Fraction(0, 1), 0.0, [])
    
    # Convert back to list of numbers (1-indexed)
    result = sorted(i + 1 for i in best_selected)
    
    # Verify no 3-AP
    def has_3ap(s):
        s_set = set(s)
        for a in s:
            for b in s:
                if a < b:
                    c = 2 * b - a
                    if c in s_set and c != a and c != b:
                        return True
        return False
    
    if has_3ap(result):
        # Fallback to greedy if verification fails
        result = sorted(i + 1 for i in greedy_solution())
    
    return result
# EVOLVE-BLOCK-END
