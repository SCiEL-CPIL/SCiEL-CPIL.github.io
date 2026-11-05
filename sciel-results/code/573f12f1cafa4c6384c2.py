import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 18
    
    def build_adj_mask(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def is_tf_mask(mask, adj):
        """Check if vertex set represented by bitmask is triangle-free."""
        m = mask
        while m:
            u_bit = m & (-m)
            u = u_bit.bit_length() - 1
            m ^= u_bit
            common = adj[u] & mask
            while common:
                v_bit = common & (-common)
                v = v_bit.bit_length() - 1
                common ^= v_bit
                if v < u:
                    continue
                w_common = adj[u] & adj[v] & mask
                if w_common:
                    return False
        return True
    
    def max_tf_size(adj):
        """Find the maximum size of a triangle-free induced subgraph using branch and bound."""
        # Check from high to low, but start from reasonable upper bound
        # For n=18, try sizes 12 down to 5
        for size in range(12, 4, -1):
            if _has_tf_of_size(size, adj):
                return size
        return 4
    
    def _has_tf_of_size(target_size, adj):
        """Check if there exists a triangle-free set of the given size using backtracking with pruning."""
        all_verts = list(range(n))
        found = [False]
        
        def search(mask, start_idx, count):
            if found[0]:
                return
            if count == target_size:
                if is_tf_mask(mask, adj):
                    found[0] = True
                return
            # Pruning: not enough vertices left
            if n - start_idx < target_size - count:
                return
            
            for i in range(start_idx, n):
                if found[0]:
                    return
                new_mask = mask | (1 << i)
                # Quick prune: if adding vertex i creates a triangle with existing vertices
                # Check if i forms a triangle with any two existing vertices in mask
                adj_i_in_mask = adj[i] & mask
                # For each neighbor j of i in mask, check if j has another neighbor k in mask that is also neighbor of i
                # Actually, just check: does i with two neighbors in mask form a triangle?
                # A triangle involving i needs two neighbors of i in mask that are also adjacent to each other
                neighbors_of_i_in_mask = adj[i] & mask
                # Check if any two neighbors of i in mask are adjacent to each other
                m2 = neighbors_of_i_in_mask
                while m2:
                    j_bit = m2 & (-m2)
                    j = j_bit.bit_length() - 1
                    m2 ^= j_bit
                    # Check if j has a neighbor in (neighbors_of_i_in_mask & ~(1<<j)) that is also in mask
                    if adj[j] & neighbors_of_i_in_mask & ~(1 << j):
                        # Adding i would create a triangle, so skip
                        break
                    else:
                        # Safe to add, but we still need full check at the end
                        pass
                else:
                    # No triangle created by adding i (at least not immediately)
                    search(new_mask, i + 1, count + 1)
                    return
                # If we broke out of the while, i creates a triangle, skip it
                # Actually the logic above is flawed; let me just do full check at target size
        
        # Simpler approach: just enumerate with early termination
        for subset in itertools.combinations(range(n), target_size):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_tf_mask(mask, adj):
                return True
            if time.time() > deadline - 10:
                return False
        return False
    
    def check_k4_free(adj, u, v):
        """Check if adding edge (u,v) would create a K4."""
        common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        m = common_uv
        while m:
            w_bit = m & (-m)
            w = w_bit.bit_length() - 1
            m ^= w_bit
            common_uvw = adj[u] & adj[v] & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            if common_uvw:
                return False
        return True
    
    def random_k4_free_graph(rng):
        """Generate a random dense K4-free graph."""
        edges = set()
        # Start with a random graph and remove edges that create K4
        # Use a greedy approach: add edges randomly, skip if creates K4
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_possible)
        
        adj = [0] * n
        for u, v in all_possible:
            # Check if adding (u,v) creates a K4
            common_uv = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
            creates_k4 = False
            m = common_uv
            while m:
                w_bit = m & (-m)
                w = w_bit.bit_length() - 1
                m ^= w_bit
                if adj[u] & adj[v] & adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << w):
                    creates_k4 = True
                    break
            if not creates_k4:
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return edges
    
    def max_tf_size_fast(adj):
        """Fast max TF size check."""
        for size in range(12, 4, -1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_tf_mask(mask, adj):
                    return size
                if time.time() > deadline - 10:
                    return 5  # conservative default
        return 4
    
    rng = random.Random(42)
    
    best_score = 18
    best_edges = set()
    
    # Try multiple random initializations
    for init in range(15):
        if time.time() > deadline - 10:
            break
        
        edges = random_k4_free_graph(rng)
        adj = build_adj_mask(edges)
        current_score = max_tf_size_fast(adj)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges)
        
        if time.time() > deadline - 10:
            break
        
        # Local search: try to add edges to reduce max TF size
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            # Try a random edge addition
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                no_improve += 1
                continue
            u, v = min(u, v), max(u, v)
            
            if (u, v) in edges:
                no_improve += 1
                continue
            
            adj = build_adj_mask(edges)
            if not check_k4_free(adj, u, v):
                no_improve += 1
                continue
            
            edges.add((u, v))
            adj = build_adj_mask(edges)
            new_score = max_tf_size_fast(adj)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(edges)
            else:
                edges.discard((u, v))
                no_improve += 1
    
    return [[u, v] for u, v in sorted(best_edges)]
