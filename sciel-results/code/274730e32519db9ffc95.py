import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    def popcount(x):
        return bin(x).count('1')

    def to_result(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return {'n': int(n), 'edges': edges}

    def adj_to_matrix(adj_masks, n):
        """Convert adjacency masks to numpy boolean matrix."""
        mat = np.zeros((n, n), dtype=bool)
        for u in range(n):
            m = adj_masks[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                mat[u, v] = True
                m ^= lsb
        return mat

    def is_regular_subset(mat, vertices):
        """Check if induced subgraph on vertices is regular."""
        sub = mat[np.ix_(vertices, vertices)]
        degrees = sub.sum(axis=1)
        return np.all(degrees == degrees[0])

    def find_violation_bitmask(adj_masks, n, max_k=None):
        """Find a regular induced subgraph of size >= 6 using bitmask iteration.
        Returns (mask, k) or (None, None)."""
        if max_k is None:
            max_k = n
        for k in range(6, min(max_k, n) + 1):
            mask = (1 << k) - 1
            while mask < (1 << n):
                temp = mask
                first_deg = -1
                is_reg = True
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = popcount(adj_masks[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                    temp ^= lsb
                if is_reg:
                    return mask, k
                c = mask & -mask
                r = mask + c
                mask = r | (((r ^ mask) >> 2) // c)
                if mask >= (1 << n):
                    break
        return None, None

    def validate_full(adj_masks, n):
        """Check all k >= 6. Returns True if valid."""
        mask, k = find_violation_bitmask(adj_masks, n)
        return mask is None

    def validate_full_numpy(adj_masks, n):
        """Fast validation using numpy for larger n."""
        mat = adj_to_matrix(adj_masks, n)
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_regular_subset(mat, comb):
                    return False
        return True

    def local_search(adj_masks, n, max_steps, time_limit):
        """Local search to eliminate regular induced subgraphs."""
        for step in range(max_steps):
            if time.time() > time_limit:
                return adj_masks, False
            
            mask, k = find_violation_bitmask(adj_masks, n, max_k=12)
            if mask is None:
                return adj_masks, True
            
            vertices = []
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vertices.append(v)
                temp ^= lsb
            
            if len(vertices) >= 2:
                u, v = random.sample(vertices, 2)
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
        
        return adj_masks, False

    def greedy_grow(target_n, time_limit):
        """Greedy vertex growth: start with small valid graph, add vertices one at a time."""
        if time.time() > time_limit:
            return None
        
        # Start with a small valid graph (empty graph on 5 vertices is valid)
        adj_masks = [0] * 5
        n = 5
        
        while n < target_n:
            if time.time() > time_limit:
                return None
            
            # Try to add vertex n by trying different edge patterns
            found = False
            for trial in range(200):
                if time.time() > time_limit:
                    break
                
                # Random edges from new vertex to existing vertices
                new_adj = [0] * (n + 1)
                for i in range(n):
                    new_adj[i] = adj_masks[i]
                # Set edges from new vertex to existing
                for i in range(n):
                    if random.random() < 0.5:
                        new_adj[n] |= (1 << i)
                        new_adj[i] |= (1 << n)
                new_adj[n] = new_adj[n]  # no self-loop
                
                # Check if adding this vertex creates any regular induced subgraph of size >= 6
                # Only need to check subsets that include the new vertex
                valid = True
                for k in range(6, min(n + 1, 12) + 1):
                    # Check all k-subsets that include vertex n
                    for others in itertools.combinations(range(n), k - 1):
                        subset = list(others) + [n]
                        # Check regularity
                        first_deg = -1
                        is_reg = True
                        for v in subset:
                            deg = popcount(new_adj[v] & ((1 << subset[0]) | 0))  # placeholder
                            # Actually compute degree within subset
                            mask = 0
                            for w in subset:
                                mask |= (1 << w)
                            deg = popcount(new_adj[v] & mask)
                            if first_deg == -1:
                                first_deg = deg
                            elif deg != first_deg:
                                is_reg = False
                                break
                        if is_reg:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    adj_masks = new_adj
                    n += 1
                    found = True
                    break
            
            if not found:
                return None
        
        # Final validation
        if validate_full(adj_masks, n):
            return adj_masks
        return None

    def try_n(n, max_restarts=5, max_steps_per_restart=200):
        """Try to find a valid graph on n vertices."""
        if time.time() > deadline - 10:
            return None
        
        # First try greedy growth
        greedy_time = min(deadline - 20, start_time + 30)
        adj = greedy_grow(n, greedy_time)
        if adj is not None:
            return adj
        
        # Then try random + local search
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            adj_masks, success = local_search(adj_masks, n, max_steps_per_restart, deadline - 15)
            
            if success:
                if validate_full(adj_masks, n):
                    return adj_masks
        
        return None

    # Fallback: small valid graph
    n_fallback = 5
    adj_fallback = [0] * n_fallback
    best_result = to_result(n_fallback, adj_fallback)
    
    # Try n values in descending order
    for n_target in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        adj = try_n(n_target)
        if adj is not None:
            best_result = to_result(n_target, adj)
            break
    
    return best_result
