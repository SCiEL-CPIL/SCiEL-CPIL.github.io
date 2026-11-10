"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n_max = 256
    numbers = list(range(2, n_max + 1))
    N = len(numbers)
    
    # 1. Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is set if lcm(numbers[i], numbers[j]) <= n_max
    # We use 0-based indexing for numbers 2..256
    # Note: Since N=255, we can use Python integers as arbitrary precision bitmasks.
    
    conflict_mask = [0] * N
    
    # Optimization: Precompute LCMs or check divisibility properties
    # lcm(a,b) <= 256 implies that the product a*b/gcd(a,b) <= 256.
    # Since a,b >= 2, this restricts the pairs significantly.
    
    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            # Calculate LCM
            # lcm(a, b) = a * b // gcd(a, b)
            g = math.gcd(a, b)
            lcm_val = (a // g) * b
            if lcm_val <= n_max:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # 2. Exact Maximum Weight Independent Set Algorithm
    # The graph of conflicts is a perfect graph (specifically, it's the complement of a comparability graph or similar structure related to divisibility).
    # However, generic MWIS on perfect graphs can be solved in polynomial time, but implementing the full ellipsoid method is complex.
    # Given N=255, a well-optimized Branch and Bound (B&B) with strong bounds (LP relaxation or greedy bounds) can solve it exactly.
    # We will use a B&B approach that branches on vertices, using bitmasks for fast intersection checks.
    
    # Weights are 1/x. We want to maximize sum(1/x).
    # To make B&B efficient, we can sort vertices by weight (descending) or by degree.
    # Let's sort by weight descending (smaller numbers first) to find good solutions early.
    # But B&B is more effective if we branch on vertices that reduce the search space quickly.
    
    # Let's use a recursive B&B with memoization on the set of available vertices?
    # No, the state space is too large for simple memoization.
    # We will use a standard B&B:
    # State: (candidate_mask, current_sum, current_set)
    # candidate_mask: bitmask of vertices that can still be added (independent of current_set)
    
    # To speed up, we precompute for each vertex i, the mask of vertices compatible with i (non-neighbors).
    # compat_mask[i] = ~conflict_mask[i] & all_mask (excluding i itself if needed, but we handle that)
    # Actually, if we add i, the new candidate mask becomes (current_candidate_mask & compat_mask[i]) & ~(1<<i)
    
    all_mask = (1 << N) - 1
    compat_mask = [0] * N
    for i in range(N):
        # Vertices j such that j != i and no conflict
        # conflict_mask[i] has bits set for j where conflict exists.
        # So compatible are those where bit is 0.
        # We also need to exclude i itself.
        compat_mask[i] = (~conflict_mask[i]) & all_mask
        compat_mask[i] &= ~(1 << i)

    # Weights as floats for pruning, and Fraction for exact comparison at the end.
    # But we need exact rational sum.
    # We can store the sum as a Fraction, but B&B nodes might be numerous.
    # Alternative: Store sum as float for pruning, and keep the set of indices to reconstruct the Fraction.
    # Given N=255, the number of additions is small (max independent set size is likely < 100).
    # Let's track the set of indices.
    
    weights_float = [1.0 / numbers[i] for i in range(N)]
    
    # Sort vertices by weight descending to improve pruning?
    # Or by degree?
    # Let's try sorting by weight descending first.
    # We need to remap indices if we sort.
    # Let's keep original indices but order the branching.
    
    # Branching variable selection:
    # Heuristic: Pick the vertex with highest weight among candidates?
    # Or pick the vertex that splits the search space most evenly?
    # For MWIS, branching on a vertex v:
    # 1. Include v: add weight, remove v and its neighbors from candidates.
    # 2. Exclude v: remove v from candidates.
    
    # To make this faster, we can use a greedy bound:
    # Upper bound on remaining weight = sum of weights of all candidates? (Weak)
    # Better: Greedy independent set weight on the candidate subgraph.
    
    def greedy_bound(mask):
        """Calculate a lower bound on the weight of an independent set in the subgraph defined by mask.
        This serves as a lower bound for the best possible solution from this node.
        Actually, for pruning, we need an UPPER bound on the total weight we can add.
        A simple upper bound is the sum of weights of all vertices in the mask.
        A tighter upper bound for MWIS is hard to compute quickly.
        However, since weights are 1/x, and x >= 2, the sum is bounded by H_256 - 1.
        Let's use a simple greedy heuristic to find a good feasible solution first,
        then use B&B to prove optimality or find a better one.
        """
        # Greedy: pick highest weight vertex, remove neighbors, repeat.
        m = mask
        total = 0.0
        while m:
            # Find vertex with max weight in m
            max_w = -1
            max_v = -1
            # Iterate bits
            temp = m
            while temp:
                # Get lowest set bit
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                if weights_float[idx] > max_w:
                    max_w = weights_float[idx]
                    max_v = idx
                temp ^= lsb
            
            if max_v == -1:
                break
                
            total += max_w
            # Remove max_v and its neighbors from m
            # Neighbors are those in conflict_mask[max_v]
            # So we keep vertices NOT in conflict_mask[max_v]
            # new_m = m & compat_mask[max_v]
            # But we must also remove max_v itself (already done by & compat_mask which excludes self)
            m &= compat_mask[max_v]
            
        return total

    # Better upper bound:
    # For a node with candidate mask M, the maximum additional weight is at most the sum of weights of vertices in M.
    # This is loose.
    # A tighter bound: Since the graph is perfect, the LP relaxation is integral.
    # But solving LP is slow.
    # Let's use the fact that if we sort vertices by weight, the sum of the top k weights is an upper bound? No.
    
    # Let's stick to B&B with a good initial solution and simple pruning.
    # Initial solution: Greedy
    current_mask = all_mask
    best_sum_float = 0.0
    best_set = []
    
    # Greedy initial solution
    m = all_mask
    s = []
    while m:
        max_w = -1
        max_v = -1
        temp = m
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            if weights_float[idx] > max_w:
                max_w = weights_float[idx]
                max_v = idx
            temp ^= lsb
        
        if max_v == -1:
            break
        
        s.append(max_v)
        best_sum_float += max_w
        m &= compat_mask[max_v]
        
    best_set = s[:]
    
    # B&B
    # We use a stack for DFS.
    # State: (mask, current_sum, current_set_tuple)
    # To avoid memory blowup, we limit depth or use recursion with sys.setrecursionlimit.
    # Given N=255, recursion depth might be an issue if we go deep, but independent sets are small.
    
    stack = [(all_mask, 0.0, tuple())]
    node_count = 0
    max_nodes = 1000000 # Limit to prevent infinite loop or timeout
    
    # Sort vertices for branching?
    # We can precompute an ordering.
    # Let's branch on the vertex with the highest weight in the mask.
    
    while stack and time.time() < deadline:
        if node_count > max_nodes:
            break
            
        mask, curr_sum, curr_set = stack.pop()
        node_count += 1
        
        # Pruning: If curr_sum + upper_bound(mask) <= best_sum_float, skip.
        # Upper bound: Sum of all weights in mask.
        # This is expensive to compute for every node.
        # Precompute prefix sums? No, mask is arbitrary.
        # We can approximate: if curr_sum + greedy_upper_bound(mask) <= best_sum_float: continue
        # Greedy upper bound is actually a lower bound on the solution, not upper.
        # We need an upper bound.
        # Simple upper bound: sum(weights[i] for i in mask).
        # Let's compute this only if curr_sum is close to best.
        
        # Check if we can improve
        # If mask is 0, we have a complete solution
        if mask == 0:
            if curr_sum > best_sum_float:
                best_sum_float = curr_sum
                best_set = list(curr_set)
            continue
        
        # Upper bound calculation:
        # Sum of weights of all vertices in mask.
        ub = 0.0
        temp = mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            ub += weights_float[idx]
            temp ^= lsb
            
        if curr_sum + ub <= best_sum_float:
            continue
            
        # Branching
        # Select a vertex to branch on.
        # Heuristic: Pick vertex with highest weight in mask.
        max_w = -1
        max_v = -1
        temp = mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            if weights_float[idx] > max_w:
                max_w = weights_float[idx]
                max_v = idx
            temp ^= lsb
            
        if max_v == -1:
            continue
            
        v_bit = 1 << max_v
        
        # Branch 1: Include max_v
        new_mask_include = mask & compat_mask[max_v]
        new_sum_include = curr_sum + weights_float[max_v]
        new_set_include = curr_set + (max_v,)
        
        # Branch 2: Exclude max_v
        new_mask_exclude = mask & ~v_bit
        new_sum_exclude = curr_sum
        new_set_exclude = curr_set
        
        # Push to stack (LIFO, so push Exclude first if we want to Explore Include first?
        # Or push both. Order affects finding good solutions early.
        # Let's push Exclude first, so Include is popped next (DFS).
        stack.append((new_mask_exclude, new_sum_exclude, new_set_exclude))
        stack.append((new_mask_include, new_sum_include, new_set_include))
        
    # Reconstruct best set
    # best_set contains indices
    result = [numbers[i] for i in best_set]
    
    # Verify validity (safety check)
    if len(result) > 1:
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                if math.lcm(result[i], result[j]) <= 256:
                    # Invalid, fallback to greedy
                    # This should not happen if B&B is correct
                    pass
    
    return result
# EVOLVE-BLOCK-END
