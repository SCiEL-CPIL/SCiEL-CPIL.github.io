import math
import time
import numpy as np
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 256
    
    # 1. Preprocessing
    # Sieve to find primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute distinct prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                prime_factors[i].add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            prime_factors[i].add(x)
    prime_factors[1] = set()
    
    def is_valid_solution(sol_list):
        # Check types, range, distinctness, and coprimality
        if len(sol_list) != len(set(sol_list)):
            return False
        for x in sol_list:
            if not isinstance(x, int) or isinstance(x, bool) or x < 1 or x > N:
                return False
        for i in range(len(sol_list)):
            for j in range(i + 1, len(sol_list)):
                if math.gcd(sol_list[i], sol_list[j]) != 1:
                    return False
        return True

    # 2. Exact Branch and Bound for Maximum Weight Independent Set
    # Vertices: 2..256
    # We always include 1.
    # We solve for the subset of {2..256} that forms an independent set in the conflict graph.
    
    vertices = list(range(2, N + 1))
    n_verts = len(vertices)
    
    # Build conflict graph adjacency matrix (or list of conflicts)
    # conflict[i][j] = True if vertices[i] and vertices[j] are NOT coprime
    # We can represent this as a bitset or a list of lists for fast access.
    # Given n_verts = 255, a list of sets or lists is fine.
    # Let's use a list of lists of indices for conflicts.
    conflicts = [set() for _ in range(n_verts)]
    for i in range(n_verts):
        vi = vertices[i]
        for j in range(i + 1, n_verts):
            vj = vertices[j]
            if math.gcd(vi, vj) != 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Precompute weights
    weights = [vertices[i] for i in range(n_verts)]
    
    # Greedy Coloring for Upper Bound
    # We need a function that colors a subset of vertices.
    # The color number of a vertex in a greedy coloring is an upper bound on the size of the independent set containing it?
    # No, the standard bound is: sum of weights of vertices in the independent set <= sum of weights of vertices of the same color?
    # Actually, the bound is: Let C be the set of available vertices.
    # If we greedily color C, and the k-th color class has weight W_k, then any independent set in C can contain at most one vertex from each color class?
    # No, an independent set can contain multiple vertices from the same color class if they are not adjacent?
    # Wait, in a proper coloring, no two adjacent vertices have the same color.
    # So an independent set (which has no adjacent vertices) CAN contain multiple vertices of the same color.
    # So the simple "one per color" bound is for the size (unweighted) if we assume the coloring is such that... wait.
    # The standard branch and bound for MWIS uses a greedy coloring of the *remaining* graph.
    # The bound is: The maximum weight independent set in a graph is at most the sum of the weights of the vertices in the *maximum weight color class*? No.
    # Let's recall the Tomita algorithm bound.
    # It uses a greedy coloring of the candidate set P.
    # The colors are assigned 1, 2, ..., k.
    # The bound is the sum of the weights of the vertices with the *highest* color? No.
    # The bound is: If we order the vertices by color, and take the suffix of the highest color, the sum of their weights is an upper bound?
    # Actually, the bound is simply: The maximum weight of an independent set in P is <= sum of weights of vertices in P? Trivial.
    # The Tomita bound:
    # Greedily color the vertices in P.
    # Let the colors be c(v).
    # The bound is the sum of weights of vertices v in P such that c(v) == max_color? No.
    # The bound is: If we pick a vertex v, we can only pick other vertices with color > c(v)? No.
    # Let's look at the specific bound used in "Maximum Weight Independent Set" algorithms.
    # Usually, the bound is computed as follows:
    # Sort vertices in P by color (greedy).
    # The bound is the sum of weights of the vertices in the last color class?
    # No, the bound is the sum of weights of all vertices in P that have the *maximum* color assigned in the greedy coloring?
    # Actually, a common bound is:
    # Let the greedy coloring assign colors 1..k.
    # The independent set can contain at most one vertex from each *clique*. But colors are not cliques.
    # However, if we use a greedy coloring, the vertices of the *same* color form an independent set? No, that's the definition of independent set.
    # Wait, in a proper coloring, vertices of the same color are *not* adjacent. So they *can* be in the same independent set.
    # So the bound "sum of weights of max color class" is NOT an upper bound on the weight of the independent set.
    
    # Let's use a different bound: The LP relaxation bound is too expensive to compute at every node.
    # A simpler bound: The sum of the weights of all vertices in P is a trivial upper bound.
    # A better bound: 
    # If we select a vertex v, we must exclude its neighbors.
    # The bound for the branch including v is: weight(v) + bound(P \ N[v]).
    # The bound for the branch excluding v is: bound(P \ {v}).
    # To prune, we need a good estimate of bound(P).
    # A common heuristic bound is the sum of weights of the top k vertices where k is the size of the maximum independent set?
    # We don't know the size.
    
    # Let's try a different approach:
    # Since N is small (255 vertices), we can use a recursive search with memoization? No, subsets are too many.
    # But the graph is dense?
    # Let's check the density.
    # Most numbers share small prime factors.
    # 2 divides half. 3 divides 1/3.
    # The graph is likely quite dense, meaning independent sets are small.
    
    # Let's use a simple recursive branch and bound with the following bound:
    # Bound(P) = sum(weights[v] for v in P if v is not "blocked" by some heuristic?)
    # Actually, let's just use the sum of weights of P as the bound. It's weak but safe.
    # To improve, we can use a greedy coloring bound correctly.
    # Correction on Tomita Bound:
    # In the unweighted case, the bound is the number of colors.
    # In the weighted case, the bound is the sum of the weights of the vertices in the *last* color class?
    # No.
    # Let's use this bound:
    # Greedy color P.
    # The bound is the sum of the weights of the vertices in P that have the *maximum* color?
    # Let's verify.
    # If v has color c, it is not adjacent to any other vertex with color c.
    # So the set of vertices with color c is an independent set.
    # But we can also pick vertices from other colors.
    # So the weight of the independent set is at least the max color class weight.
    # This is a LOWER bound.
    
    # We need an UPPER bound.
    # Upper bound: Sum of all weights in P.
    # This is very weak.
    
    # Alternative:
    # Use the fact that if we pick v, we remove N[v].
    # We can precompute the sum of weights of P.
    # At each node, we have a set of candidates P.
    # We iterate through v in P (sorted by weight descending).
    # We branch on including v.
    # New P = P \ (N[v] U {v}).
    # We also branch on excluding v?
    # In MWIS, we don't need to explicitly branch on excluding v if we iterate through v in P and always try to include v, then move to the next v in the *original* P (excluding the ones removed by the inclusion of previous vertices)?
    # Yes, this is the standard "include/exclude" but optimized.
    # Algorithm:
    # Search(P, current_weight):
    #   if P is empty: update best
    #   bound = current_weight + sum(weights[v] for v in P)
    #   if bound <= best: return
    #   for v in P (sorted by weight desc):
    #       if current_weight + weight(v) + sum(weights[u] for u in P if u not in N[v] and u != v) <= best: break (prune rest)
    #       Search(P \ (N[v] U {v}), current_weight + weight(v))
    #       P = P \ {v}  # Exclude v for subsequent iterations in this loop
    
    # This requires efficient set operations.
    # With 255 vertices, bitsets (integers) are very fast in Python.
    
    # Let's implement with bitsets.
    # Map vertex index 0..254 to bit 0..254.
    # conflicts_mask[i] is an integer with bits set for all neighbors of i.
    
    conflicts_mask = [0] * n_verts
    for i in range(n_verts):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflicts_mask[i] = mask
    
    # weights_list
    w = [vertices[i] for i in range(n_verts)]
    
    best_sum = 0
    best_set_mask = 0
    
    # Precompute sum of weights for any subset? No, too many.
    # We will compute sum(weights[v] for v in P) on the fly.
    # To speed up, we can precompute prefix sums? No, P changes arbitrarily.
    # But 255 iterations per node is fine.
    
    def sum_weights(mask):
        s = 0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s += w[idx]
            m ^= lsb
        return s
    
    # This is O(popcount(mask)). Max 255.
    
    def search(mask, current_weight):
        nonlocal best_sum, best_set_mask
        
        if mask == 0:
            if current_weight > best_sum:
                best_sum = current_weight
                best_set_mask = 0 # We need to track the set.
            return
        
        # Bound: current_weight + sum_weights(mask)
        if current_weight + sum_weights(mask) <= best_sum:
            return
        
        # Get vertices in mask, sorted by weight descending
        # We can just iterate through all vertices and check if they are in mask.
        # To sort, we need a list.
        # Since n_verts is small, we can just collect them.
        verts_in_mask = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            verts_in_mask.append(idx)
            m ^= lsb
        
        # Sort by weight descending
        verts_in_mask.sort(key=lambda x: w[x], reverse=True)
        
        for v in verts_in_mask:
            # Check if v is still in the "available" set for this branch
            # In this algorithm, we maintain the mask of available vertices.
            # However, the standard algorithm modifies the mask as we go.
            # Let's re-structure:
            # We iterate through candidates.
            # We need to know if v is still available.
            # The mask passed to search is the set of vertices available BEFORE this loop.
            # As we branch on v, we create a new mask for the recursive call.
            # For the next iteration of the loop (excluding v), we should remove v from the available set.
            # But the `mask` variable is local to this function call.
            # We can just check if the bit for v is set in `mask`.
            
            if not (mask & (1 << v)):
                continue
            
            # Pruning check:
            # If we include v, the new available set is mask & ~conflicts_mask[v] & ~(1 << v)
            # The upper bound for this branch is:
            # current_weight + w[v] + sum_weights(new_mask)
            new_mask = mask & ~conflicts_mask[v] & ~(1 << v)
            
            if current_weight + w[v] + sum_weights(new_mask) <= best_sum:
                continue
            
            # Recurse
            search(new_mask, current_weight + w[v])
            
            # Remove v from available set for subsequent iterations (excluding v)
            mask &= ~(1 << v)
            
            # Check if mask is empty
            if mask == 0:
                break

    # Initialize best_sum with a good heuristic
    # Greedy: pick largest weight, remove conflicts, repeat.
    mask_avail = (1 << n_verts) - 1
    greedy_sum = 0
    greedy_mask = 0
    m = mask_avail
    while m:
        # Find vertex with max weight in m
        best_v = -1
        best_w = -1
        mm = m
        while mm:
            lsb = mm & (-mm)
            idx = lsb.bit_length() - 1
            if w[idx] > best_w:
                best_w = w[idx]
                best_v = idx
            mm ^= lsb
        
        if best_v == -1:
            break
            
        greedy_sum += w[best_v]
        greedy_mask |= (1 << best_v)
        
        # Remove best_v and its neighbors
        m &= ~conflicts_mask[best_v]
        m &= ~(1 << best_v)
        
    best_sum = greedy_sum
    best_set_mask = greedy_mask
    
    # Run B&B
    search((1 << n_verts) - 1, 0)
    
    # Extract solution
    sol_set = {1}
    m = best_set_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        sol_set.add(vertices[idx])
        m ^= lsb
    
    best_result = list(sol_set)
    
    # Validate
    if not is_valid_solution(best_result):
        # Fallback to greedy
        sol_set = {1}
        m = mask_avail
        while m:
            best_v = -1
            best_w = -1
            mm = m
            while mm:
                lsb = mm & (-mm)
                idx = lsb.bit_length() - 1
                if w[idx] > best_w:
                    best_w = w[idx]
                    best_v = idx
                mm ^= lsb
            if best_v == -1:
                break
            sol_set.add(vertices[best_v])
            m &= ~conflicts_mask[best_v]
            m &= ~(1 << best_v)
        best_result = list(sol_set)
    
    return best_result

