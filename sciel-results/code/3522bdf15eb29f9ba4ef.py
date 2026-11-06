import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    N = 512
    # Candidates: 1..511
    # Value of a: 1/(512-a)
    # Conflict: gcd(a,b) > 1

    # Precompute prime factors for each number
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors
    pf = [set() for _ in range(N)]
    for a in range(1, N):
        pf[a] = prime_factors(a)

    # Precompute conflict masks
    # conflict_mask[a] has bit b set if gcd(a,b) > 1
    conflict_mask = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and any(p in pf[b] for p in pf[a]):
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Sort vertices by weight descending
    # Weights: w_a = 1/(512-a)
    vertices = list(range(1, N))
    vertices.sort(key=lambda x: 1.0 / (N - x), reverse=True)
    
    # Map original value to index in sorted list
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    M = len(vertices)
    
    # Weights in sorted order
    weights = [1.0 / (N - v) for v in vertices]
    
    # Conflict masks in sorted order
    # sorted_conflict[i] is a bitmask of indices j such that vertices[i] and vertices[j] conflict
    sorted_conflict = [0] * M
    for i, v in enumerate(vertices):
        mask = 0
        orig_mask = conflict_mask[v]
        for j, u in enumerate(vertices):
            if (orig_mask >> u) & 1:
                mask |= (1 << j)
        sorted_conflict[i] = mask

    # Initial Greedy Solution
    chosen_mask = 0
    greedy_sum = 0.0
    for i in range(M):
        if (chosen_mask >> i) & 1:
            continue
        # Check if i conflicts with any chosen
        # We can just check if i is in the "forbidden" set
        # Let's maintain a forbidden mask
        pass

    # Better Greedy:
    available = (1 << M) - 1
    greedy_sum = 0.0
    chosen_indices = []
    for i in range(M):
        if not (available & (1 << i)):
            continue
        # Add i
        chosen_indices.append(i)
        greedy_sum += weights[i]
        # Remove i and its neighbors from available
        available &= ~(1 << i)
        available &= ~sorted_conflict[i]

    best_sum = greedy_sum
    best_mask = 0
    for i in chosen_indices:
        best_mask |= (1 << i)

    # Branch and Bound
    # State: idx, available_mask, current_sum
    # We use a stack or recursion. Recursion depth 511 is fine.
    
    import sys
    sys.setrecursionlimit(10000)

    def bound(available_mask):
        # Sum of weights of available vertices
        # This is a valid upper bound
        s = 0.0
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # Tighter Bound: Greedy Coloring
    # For a set of vertices, color them such that adjacent vertices have different colors.
    # The sum of the max weight in each color is an upper bound.
    # This is expensive. Let's try the simple sum first.
    # If the simple sum is too slow or doesn't prune well, we can switch.
    # Given the weights are sorted descending, the sum of the first k available items is a good proxy?
    # No, sum of all available is the standard bound.

    def dfs(idx, available_mask, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        if time.time() > DEADLINE:
            return

        if idx >= M:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = current_mask
            return

        # Pruning
        # Upper bound: current_sum + sum(weights of available vertices from idx onwards)
        # Note: available_mask might contain vertices < idx that were skipped?
        # No, we iterate idx from 0 to M-1.
        # If a vertex < idx is still in available_mask, it means we skipped it in a previous branch?
        # Actually, in this formulation:
        # At step idx, we decide to include or exclude vertices[idx].
        # If we exclude it, we move to idx+1.
        # If we include it, we remove it and its neighbors from available_mask, and move to idx+1.
        # So available_mask only contains vertices >= idx?
        # Not necessarily. If we included a vertex at j < idx, it might have removed vertices > j.
        # But vertices < idx are already decided.
        # So we should only consider vertices >= idx in the bound?
        # Actually, any vertex in available_mask is a candidate.
        # But we are processing in order.
        # If a vertex k < idx is in available_mask, it means we chose to exclude it.
        # So it shouldn't be counted in the bound for the *remaining* search space?
        # Wait. If we exclude k, we don't add its weight.
        # The bound should be on the *additional* weight we can get.
        # The additional weight comes from vertices we haven't decided yet.
        # Since we process sequentially, the undecided vertices are those with index >= idx.
        # However, `available_mask` might still have bits set for indices < idx if we didn't clear them?
        # In my `dfs`, I don't explicitly clear bits for excluded vertices.
        # I just pass `available_mask` to the next call.
        # So `available_mask` contains all vertices that are *not* conflicting with *included* vertices.
        # It includes vertices we have already decided to *exclude*.
        # So the bound `sum(weights of available_mask)` is an overestimate because it includes weights of excluded vertices.
        # To fix this, we should mask out vertices < idx?
        # Or better, maintain a `remaining_mask` that only contains vertices >= idx?
        
        # Let's refine the state:
        # `available_mask` contains vertices that are *eligible* (no conflict with chosen).
        # We iterate `idx` from 0 to M-1.
        # If `idx` is not in `available_mask`, we must skip it (it's either excluded or removed by conflict).
        # If `idx` is in `available_mask`, we have two choices:
        # 1. Exclude `idx`: recurse with same `available_mask`, `idx+1`.
        # 2. Include `idx`: recurse with `available_mask` & ~(neighbors), `idx+1`.
        
        # For the bound:
        # The maximum additional weight is the sum of weights of all vertices in `available_mask` with index >= idx.
        # Let's precompute prefix sums? No, the mask is dynamic.
        # Let's just sum the weights of bits in `available_mask` that are >= idx.
        
        # Optimization:
        # If `current_sum` + `sum(weights[i] for i in range(idx, M) if (available_mask >> i) & 1)` <= `best_sum`, prune.
        
        # This sum calculation is O(M) per node. With 2^M nodes, this is too slow.
        # We need a faster bound.
        
        # Alternative:
        # Use the fact that weights are sorted descending.
        # The max possible sum is the sum of the top k weights in `available_mask`?
        # No, they must be independent.
        
        # Let's try a different approach: **Maximum Weight Independent Set via Integer Linear Programming (ILP) using SciPy?**
        # SciPy doesn't have a built-in ILP solver that is fast for 511 variables without external libs like `scip` or `glpk` which might not be installed.
        # The prompt says "NumPy and SciPy are available".
        
        # Let's stick to B&B but optimize the bound.
        # A common tight bound for MWIS is the **Greedy Coloring Bound**.
        # Let's implement that.
        
        def coloring_bound(available_mask, start_idx):
            # Only consider vertices >= start_idx
            # Get list of available vertices >= start_idx
            verts = []
            m = available_mask
            # Mask out bits < start_idx
            m &= ~((1 << start_idx) - 1)
            
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb
                
            if not verts:
                return 0.0
            
            # Greedy Coloring
            # Order by weight descending (they already are)
            colors = {}
            max_weight_per_color = {}
            
            # To speed up, we can just assign colors greedily
            # For each vertex, find the first color it doesn't conflict with
            # Conflict check: (sorted_conflict[v] & color_mask[color]) == 0
            
            color_masks = []
            
            for v in verts:
                placed = False
                for c, cm in enumerate(color_masks):
                    if (sorted_conflict[v] & cm) == 0:
                        # Can use color c
                        color_masks[c] = cm | (1 << v)
                        # Update max weight for this color
                        if v not in max_weight_per_color or weights[v] > max_weight_per_color[v]:
                            # Wait, max_weight_per_color should store the max weight *vertex* in that color?
                            # No, the bound is sum of max weights *per color*.
                            # So we need to track the max weight in each color class.
                            pass
                        placed = True
                        break
                if not placed:
                    color_masks.append(1 << v)
                    
            # Calculate bound
            # For each color, find the max weight vertex in it
            bound_val = 0.0
            for cm in color_masks:
                m = cm
                max_w = 0.0
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if weights[idx] > max_w:
                        max_w = weights[idx]
                    m ^= lsb
                bound_val += max_w
                
            return bound_val

        # This is O(V^2) per node. Might be too slow.
        
        # Let's try a simpler bound first: **Sum of weights of available vertices >= idx**.
        # And hope the search space is pruned enough by the greedy initial solution.
        # The greedy solution is very strong.
        
        # Let's optimize the sum calculation.
        # Precompute `suffix_sum[i]` = sum(weights[i:])?
        # But `available_mask` is sparse.
        
        # Let's try the B&B with the simple sum bound, but break early if time runs out.
        
        if current_sum > best_sum:
            # Update best if we find a better solution at a leaf?
            # Or can we update at any node?
            # The current set is a valid independent set.
            # So we can update best_sum anytime.
            best_sum = current_sum
            best_mask = current_mask

        # Prune
        # Calculate sum of weights of available vertices with index >= idx
        # We can do this quickly if we precompute prefix sums of weights?
        # No, because `available_mask` is arbitrary.
        
        # Let's just do the sum.
        rem_sum = 0.0
        m = available_mask & ~((1 << idx) - 1)
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            rem_sum += weights[i]
            m ^= lsb
            
        if current_sum + rem_sum <= best_sum:
            return

        # Branch
        # 1. Include vertices[idx]
        if (available_mask >> idx) & 1:
            # Include
            new_mask = available_mask & ~(1 << idx) & ~sorted_conflict[idx]
            dfs(idx + 1, new_mask, current_sum + weights[idx], current_mask | (1 << idx))
            
            # Exclude
            dfs(idx + 1, available_mask, current_sum, current_mask)
        else:
            # Must exclude (already excluded or conflicting)
            dfs(idx + 1, available_mask, current_sum, current_mask)

    # Start B&B
    # Initial available mask is all 1s
    initial_mask = (1 << M) - 1
    dfs(0, initial_mask, 0.0, 0)

    # Extract result
    result = []
    for i in range(M):
        if (best_mask >> i) & 1:
            result.append(vertices[i])
            
    return result
