"""Maximum Weight Independent Set on 3-AP hypergraph via specialized branch-and-bound."""

from fractions import Fraction
import time

N = 256

def construct():
    start_time = time.time()
    
    # 1. Precompute conflict pairs
    # conflict_pairs[v] is a list of tuples (bit_a, bit_b) such that {a, b, v} is a 3-AP.
    # Indices 0..255 corresponding to 1..256.
    conflict_pairs = [[] for _ in range(N)]
    
    for v in range(1, N + 1):
        vi = v - 1
        # v as smallest: (v, v+d, v+2d) -> pair (v+d, v+2d)
        d = 1
        while v + 2 * d <= N:
            a = v + d - 1
            b = v + 2 * d - 1
            conflict_pairs[vi].append((1 << a, 1 << b))
            d += 1
        # v as middle: (v-d, v, v+d) -> pair (v-d, v+d)
        d = 1
        while v - d >= 1:
            a = v - d - 1
            b = v + d - 1
            conflict_pairs[vi].append((1 << a, 1 << b))
            d += 1
        # v as largest: (v-2d, v-d, v) -> pair (v-2d, v-d)
        d = 1
        while v - 2 * d >= 1:
            a = v - 2 * d - 1
            b = v - d - 1
            conflict_pairs[vi].append((1 << a, 1 << b))
            d += 1
            
    # 2. Order vertices by decreasing weight (1/x)
    # Weight is 1/x, so order is 1, 2, ..., 256.
    # Index 0 is vertex 1, Index 1 is vertex 2, etc.
    order = list(range(N))
    
    # 3. Precompute weights
    weights = [Fraction(1, i + 1) for i in range(N)]
    
    # Precompute suffix sums of weights for loose bound
    suffix_sum = [Fraction(0)] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
    
    # 4. Greedy initialization
    def can_add(u, mask):
        # u is 0-indexed
        for bit_a, bit_b in conflict_pairs[u]:
            if (mask & bit_a) and (mask & bit_b):
                return False
        return True

    greedy_mask = 0
    greedy_set = []
    for u in order:
        if can_add(u, greedy_mask):
            greedy_mask |= (1 << u)
            greedy_set.append(u + 1)
            
    best_set = list(greedy_set)
    best_weight = sum(weights[u] for u in range(N) if (greedy_mask >> u) & 1)
    
    # 5. Branch and Bound
    # To speed up, we track the current mask and weight.
    # We use a recursive function.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bnb(i, mask, current_weight):
        nonlocal best_set, best_weight
        
        # Time check
        if time.time() - start_time > 58:
            return
        
        # Base case
        if i == N:
            if current_weight > best_weight:
                best_weight = current_weight
                # Reconstruct set from mask
                best_set = [u + 1 for u in range(N) if (mask >> u) & 1]
            return
        
        # Loose bound check
        if current_weight + suffix_sum[i] <= best_weight:
            return
            
        # Tighter bound: Sum weights of remaining vertices that are compatible with current mask
        ub = current_weight
        for j in range(i, N):
            u = order[j]
            if can_add(u, mask):
                ub += weights[u]
                
        if ub <= best_weight:
            return
            
        v = order[i]
        
        # Option 1: Include v
        if can_add(v, mask):
            new_mask = mask | (1 << v)
            bnb(i + 1, new_mask, current_weight + weights[v])
            
        # Option 2: Exclude v
        bnb(i + 1, mask, current_weight)
        
    # Start B&B
    bnb(0, 0, Fraction(0))
    
    return best_set
