import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a,b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    """
    N = 255
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Vertices: numbers in [1, 255] that are divisible by at least one of {3, 5, 17}
    # (i.e., gcd(x, 255) > 1)
    vertices = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    n_v = len(vertices)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask where bit j is set if gcd(vertices[i], vertices[j]) > 1
    adj = [0] * n_v
    for i in range(n_v):
        vi = vertices[i]
        mask = 0
        for j in range(n_v):
            if i != j and math.gcd(vi, vertices[j]) > 1:
                mask |= (1 << j)
        adj[i] = mask
    
    # 255 must be in the set
    idx_255 = v_to_idx[255]
    
    # Helper: iter bits of a mask
    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
    
    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Improved greedy with multiple orderings
    def greedy_clique(order):
        """
        Greedy clique construction using a specific vertex ordering.
        Returns the clique as a list of vertex values.
        """
        R = []
        P = (1 << n_v) - 1
        
        for v in order:
            if not (P & (1 << v)):
                continue
            # Check if v is compatible with all in R
            # v is compatible with all in R if v is in P (which tracks common neighbors)
            R.append(vertices[v])
            P &= adj[v]
            if not P:
                break
        
        return R
    
    # Generate several orderings
    # Ordering 1: by degree (highest first)
    degrees = [popcount(adj[i]) for i in range(n_v)]
    order_by_degree = sorted(range(n_v), key=lambda i: -degrees[i])
    
    # Ordering 2: by value (ascending)
    order_by_value = list(range(n_v))
    
    # Ordering 3: by value (descending)
    order_by_value_desc = list(range(n_v - 1, -1, -1))
    
    # Ordering 4: random-like (interleaved)
    order_interleaved = []
    left = 0
    right = n_v - 1
    while left <= right:
        order_interleaved.append(left)
        if left != right:
            order_interleaved.append(right)
        left += 1
        right -= 1
    
    # Try all orderings and keep the best
    best_greedy = []
    for order in [order_by_degree, order_by_value, order_by_value_desc, order_interleaved]:
        clique = greedy_clique(order)
        if len(clique) > len(best_greedy):
            best_greedy = clique
    
    # Local search: try to improve the greedy clique
    def local_search_improve(clique_values, max_iterations=100):
        """
        Try to improve a clique by swapping or adding vertices.
        Returns the improved clique.
        """
        R = clique_values[:]
        R_set = set(R)
        R_indices = [v_to_idx[v] for v in R]
        
        # Compute common neighbors of all vertices in R
        if not R_indices:
            return R
        
        common = (1 << n_v) - 1
        for idx in R_indices:
            common &= adj[idx]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try adding a vertex from common neighbors
            best_add = -1
            best_add_degree = -1
            for v in iter_bits(common):
                if v in R_set:
                    continue
                # Check if adding v improves something (just check if it's valid)
                # It's valid by construction (in common neighbors)
                # Prefer vertices with high degree for future expansion
                deg = popcount(common & adj[v])
                if deg > best_add_degree:
                    best_add_degree = deg
                    best_add = v
            
            if best_add >= 0:
                R.append(vertices[best_add])
                R_set.add(vertices[best_add])
                common &= adj[best_add]
                improved = True
                continue
            
            # Try swapping: remove one vertex, add a better one
            for remove_idx in R_indices[:]:
                if vertices[remove_idx] not in R_set:
                    continue
                # Temporarily remove
                R_set.remove(vertices[remove_idx])
                R_indices.remove(remove_idx)
                
                # Recompute common neighbors without this vertex
                if not R_indices:
                    temp_common = (1 << n_v) - 1
                else:
                    temp_common = (1 << n_v) - 1
                    for idx in R_indices:
                        temp_common &= adj[idx]
                
                # Try to add a vertex that connects to all remaining
                best_swap = -1
                for v in iter_bits(temp_common):
                    if v in R_set:
                        continue
                    # Check if this swap improves the clique
                    # Simple heuristic: prefer vertices with higher degree
                    deg = popcount(temp_common & adj[v])
                    if best_swap == -1 or deg > popcount(common & adj[best_swap]):
                        best_swap = v
                
                if best_swap >= 0 and best_swap not in R_set:
                    # Perform the swap
                    R.append(vertices[best_swap])
                    R_set.add(vertices[best_swap])
                    R_indices.append(best_swap)
                    common = temp_common & adj[best_swap]
                    improved = True
                    break
                else:
                    # Revert
                    R_set.add(vertices[remove_idx])
                    R_indices.append(remove_idx)
        
        return R
    
    # Apply local search to the best greedy result
    improved_clique = local_search_improve(best_greedy)
    
    max_clique_size = len(improved_clique)
    max_clique_set = improved_clique[:]
    
    # Bron-Kerbosch with Pivoting for Maximum Clique
    # We want a clique that includes idx_255.
    # Since 255 is universal in this graph, the max clique containing 255
    # is exactly the max clique in the whole graph.
    
    def bron_kerbosch_pruned(R, P, X):
        nonlocal max_clique_size, max_clique_set
        
        if time.time() > deadline:
            return
            
        if not P and not X:
            if len(R) > max_clique_size:
                max_clique_size = len(R)
                max_clique_set = R[:]
            return
        
        # Pruning: if we can't beat current best, return
        if len(R) + popcount(P) <= max_clique_size:
            return
            
        # Pivot selection
        PX = P | X
        if not PX:
            return

        best_u = -1
        max_deg = -1
        
        for u in iter_bits(PX):
            deg = popcount(P & adj[u])
            if deg > max_deg:
                max_deg = deg
                best_u = u
        
        if best_u == -1:
            return

        P_not_N_u = P & ~adj[best_u]
        
        for v in iter_bits(P_not_N_u):
            R.append(vertices[v])
            P_new = P & adj[v]
            X_new = X & adj[v]
            
            bron_kerbosch_pruned(R, P_new, X_new)
            
            R.pop()
            
            P &= ~(1 << v)
            X |= (1 << v)
            
            if time.time() > deadline:
                return

    initial_P = (1 << n_v) - 1
    bron_kerbosch_pruned([], initial_P, 0)
    
    # Ensure 255 is in the result
    result = max_clique_set[:]
    if 255 not in result:
        result.append(255)
        
    # Fallback if something went wrong
    if not result:
        result = [255]
        
    return result
