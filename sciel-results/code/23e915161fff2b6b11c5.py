"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    # Numbers 2..40
    nums = list(range(2, 41))
    n = len(nums)
    
    # Build conflict graph: two numbers conflict if one divides the other
    # If a|b (a divides b), they cannot both be in the set because:
    # product of subset {b} = b, and if we have a and b/a in set, product = b
    # More conservatively: if a|b, mark as conflict
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nums[i], nums[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Maximum independent set via branch and bound
    # Use bitmasks for efficiency
    n_bits = 1 << n
    
    # Precompute: for each vertex, the set of vertices it conflicts with (as bitmask)
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if conflict[i][j]:
                conflict_mask[i] |= (1 << j)
    
    # Branch and bound for maximum independent set
    best_set = set()
    best_size = 0
    
    def branch_and_bound(available_mask, current_set, current_size, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + popcount(available_mask)
        remaining = bin(available_mask).count('1')
        if current_size + remaining <= best_size:
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Try to add a vertex
        # Pick a vertex from available
        # Use greedy coloring bound for tighter pruning
        # Simple approach: pick first available vertex
        
        # Get list of available vertices
        avail_list = []
        m = available_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            avail_list.append(v)
            m ^= lsb
        
        if not avail_list:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Try adding each available vertex
        for v in avail_list:
            if (available_mask & (1 << v)) == 0:
                continue
            
            # If we add v, remove v and all its neighbors from available
            new_available = available_mask & ~(1 << v) & ~conflict_mask[v]
            current_set.append(v)
            branch_and_bound(new_available, current_set, current_size + 1, depth + 1)
            current_set.pop()
            
            # Also try NOT adding v (just remove v from available)
            new_available_no = available_mask & ~(1 << v)
            branch_and_bound(new_available_no, current_set, current_size, depth + 1)
    
    # Initial call with all vertices available
    full_mask = (1 << n) - 1
    branch_and_bound(full_mask, [], 0, 0)
    
    # Convert best_set (indices) to values
    result = [nums[i] for i in best_set]
    
    # Verify: check all 2^k subset products are distinct
    k = len(result)
    if k <= 20:  # Only verify if feasible
        products = set()
        for mask in range(1 << k):
            prod = 1
            for i in range(k):
                if mask & (1 << i):
                    prod *= result[i]
            if prod in products:
                # Collision found - this shouldn't happen with our conflict graph
                # Fall back to a simpler valid set
                result = [nums[i] for i in range(n) if all(nums[i] % nums[j] != 0 and nums[j] % nums[i] != 0 for j in range(n) if j != i)]
                break
            products.add(prod)
    else:
        # For large sets, trust the conflict graph
        pass
    
    # Ensure we return Python ints in a list
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
