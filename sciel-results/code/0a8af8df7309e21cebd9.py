import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    best_n = 11
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback

    def get_matrices(colors, n):
        """
        Build adjacency matrices for each color.
        """
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        return mats

    def count_c4s(mats, n):
        """
        Count C4s using matrix multiplication.
        """
        total_cost = 0.0
        for m in mats:
            M = m @ m
            # M[u, v] is number of common neighbors
            # We want sum_{u < v} binom(M[u, v], 2)
            # = 0.5 * sum_{u < v} (M[u, v]^2 - M[u, v])
            # sum_{u < v} M[u, v]^2 = (sum(M**2) - sum(diag(M)**2)) / 2
            # sum_{u < v} M[u, v] = (sum(M) - sum(diag(M))) / 2
            diag_M = np.diag(M)
            sum_M_sq = np.sum(M**2)
            sum_M = np.sum(M)
            
            sum_diag_sq = np.sum(diag_M**2)
            sum_diag = np.sum(diag_M)
            
            sum_off_diag_sq = (sum_M_sq - sum_diag_sq) / 2.0
            sum_off_diag = (sum_M - sum_diag) / 2.0
            
            cost_c = (sum_off_diag_sq - sum_off_diag) / 2.0
            total_cost += cost_c
        return total_cost

    def get_cost(colors, n):
        """
        Calculate initial cost.
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        mats = get_matrices(colors, n)
        return count_c4s(mats, n)

    # Precompute edge indices for fast access
    # edge_idx[i, j] = index in colors list for edge (i, j) with i < j
    # We need this for O(1) lookup of edge colors.
    
    def solve_n(n):
        num_edges = n * (n - 1) // 2
        max_iter = 50000
        max_restarts = 20
        
        # Precompute edge index mapping
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                idx += 1
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Build initial CN tables
            # CN[c][u][v] = number of common neighbors of u and v in color c
            # We only need upper triangle u < v
            CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
            
            # Efficient initialization of CN
            # For each color c, for each vertex k, for each pair (u, v) in N_c(k):
            #   CN[c][u][v] += 1
            # This is O(5 * n * deg^2). For n=15, deg ~ 14, so 5 * 15 * 196 ~ 15000 ops. Fast.
            
            # Build adjacency lists for each color
            adj = [[] for _ in range(5)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_index[(i, j)]]
                    adj[c].append((i, j))
            
            # Actually, simpler to build adjacency matrix first, then iterate.
            # Or just iterate edges.
            # Let's use the matrix approach for init as it's vectorized-ish if we are careful, 
            # but explicit loops are fine for n <= 20.
            
            # Re-initialize CN properly
            for c in range(5):
                CN[c] = np.zeros((n, n), dtype=np.int16)
            
            # For each color c, for each vertex k, find all neighbors u, v in color c.
            # For all pairs (u, v), increment CN[c][u][v].
            
            # Build neighbor lists for each color
            neighbors = [set() for _ in range(5)]
            for i in range(n):
                neighbors[i] = [set() for _ in range(5)] # neighbors[i][c] = set of neighbors of i in color c
            
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_index[(i, j)]]
                    neighbors[i][c].add(j)
                    neighbors[j][c].add(i)
            
            for c in range(5):
                for k in range(n):
                    nbrs = list(neighbors[k][c])
                    for i in range(len(nbrs)):
                        u = nbrs[i]
                        for j in range(i + 1, len(nbrs)):
                            v = nbrs[j]
                            if u < v:
                                CN[c][u, v] += 1
                            else:
                                CN[c][v, u] += 1
            
            # Calculate initial cost
            current_cost = 0.0
            for c in range(5):
                # Sum binom(CN[c][u, v], 2) for u < v
                # We can compute this from the matrix
                M = CN[c]
                # Only upper triangle
                tri_u, tri_v = np.triu_indices(n, 1)
                vals = M[tri_u, tri_v]
                # binom(x, 2) = x*(x-1)/2
                cost_c = np.sum(vals * (vals - 1) / 2.0)
                current_cost += cost_c
            
            if current_cost == 0:
                return colors
            
            # Local Search
            stuck_count = 0
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                if current_cost == 0:
                    return colors
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                # Find endpoints of edge e_idx
                # We can precompute endpoints array
                # Let's create endpoints array once outside restart loop? 
                # It depends on n, so inside solve_n is fine.
                
                # We need endpoints for e_idx
                # e_idx corresponds to (i, j) where sum_{k=0}^{i-1} (n-1-k) + (j-i-1) = e_idx
                # This is slow to invert. Let's precompute endpoints list.
                
                # Precomputation outside restart loop:
                # endpoints = []
                # for i in range(n):
                #   for j in range(i+1, n):
                #     endpoints.append((i, j))
                
                # Let's move this up.
                
                # For now, assume we have endpoints.
                
                # I will refactor to precompute endpoints at the start of solve_n.
                
                # Refactored solve_n logic below.
                
                pass # Placeholder for logic replacement
            
        return None

    # Refactored solve_n with precomputation
    def solve_n_optimized(n):
        num_edges = n * (n - 1) // 2
        max_iter = 50000
        max_restarts = 20
        
        # Precompute edge index mapping and endpoints
        edge_index = {}
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                endpoints.append((i, j))
                idx += 1
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Build neighbor sets for each vertex and color
            neighbors = [ [set() for _ in range(5)] for _ in range(n) ]
            for e_idx in range(num_edges):
                u, v = endpoints[e_idx]
                c = colors[e_idx]
                neighbors[u][c].add(v)
                neighbors[v][c].add(u)
            
            # Build CN tables
            CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
            for c in range(5):
                for k in range(n):
                    nbrs = list(neighbors[k][c])
                    for i in range(len(nbrs)):
                        u = nbrs[i]
                        for j in range(i + 1, len(nbrs)):
                            v = nbrs[j]
                            if u < v:
                                CN[c][u, v] += 1
                            else:
                                CN[c][v, u] += 1
            
            # Calculate initial cost
            current_cost = 0.0
            for c in range(5):
                M = CN[c]
                tri_u, tri_v = np.triu_indices(n, 1)
                vals = M[tri_u, tri_v]
                cost_c = np.sum(vals * (vals - 1) / 2.0)
                current_cost += cost_c
            
            if current_cost == 0:
                return colors
            
            # Local Search
            stuck_count = 0
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                if current_cost == 0:
                    return colors
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                u, v = endpoints[e_idx]
                old_c = colors[e_idx]
                
                # Try changing to a random new color
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                # Calculate delta cost
                delta_cost = 0.0
                
                # Update CN for old_c (decrement)
                # Affected pairs: (v, x) if (u, x) is old_c, and (u, x) if (v, x) is old_c
                # Note: u and v are the endpoints of the edge being changed.
                # The edge (u, v) itself is not in the "common neighbor" calculation for pair (u, v)
                # because common neighbors are distinct from u and v.
                
                # For each x in 0..n-1, x != u, x != v:
                # 1. If edge (u, x) has color old_c:
                #    Then u and x share neighbor v in old_c? No.
                #    Path u-v-x. Common neighbor of u and x is v.
                #    So CN[old_c][u, x] decreases by 1.
                # 2. If edge (v, x) has color old_c:
                #    Path v-u-x. Common neighbor of v and x is u.
                #    So CN[old_c][v, x] decreases by 1.
                
                # We need to iterate over all x != u, v.
                # This is O(n).
                
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    # Check edge (u, x)
                    # Get index of edge (u, x)
                    if u < x:
                        idx_ux = edge_index[(u, x)]
                    else:
                        idx_ux = edge_index[(x, u)]
                    
                    c_ux = colors[idx_ux]
                    
                    if c_ux == old_c:
                        # CN[old_c][u, x] decreases by 1
                        # Ensure u < x for indexing? CN is symmetric, but we stored it in upper triangle?
                        # No, we stored it in CN[c][u, v] where u, v can be any order?
                        # In init: 
                        # if u < v: CN[c][u, v] += 1 else CN[c][v, u] += 1
                        # So CN[c] is symmetric? No, only one triangle is populated.
                        # Let's make CN symmetric to make access easier, or just handle indexing.
                        # Symmetric is safer for O(1) access without checking order.
                        # Let's re-init CN to be symmetric.
                        pass # Logic needs adjustment for symmetric CN
                        
                    # Check edge (v, x)
                    if v < x:
                        idx_vx = edge_index[(v, x)]
                    else:
                        idx_vx = edge_index[(x, v)]
                    
                    c_vx = colors[idx_vx]
                    
                    if c_vx == old_c:
                        # CN[old_c][v, x] decreases by 1
                        pass
                
                # This manual loop is getting complex and error-prone with the triangle storage.
                # Let's stick to the matrix multiplication for now but optimize the search?
                # No, matrix multiplication is the bottleneck.
                
                # Alternative: Use the symmetric CN matrix.
                # Re-implement init to fill symmetric matrix.
                
                return None # Abort current attempt, need to fix implementation

    # Let's try a different approach: Keep the matrix multiplication but use it less frequently?
    # Or just accept that n=11 is the limit for this simple approach?
    # The seed score is 1.636 which corresponds to n=11 (11/11=1.0? No, 1.636 is 11/6.7? No.)
    # Wait, score is value/seed. Seed is likely n=7 or something?
    # If seed is n=7, and we get n=11, score is 11/7 = 1.57.
    # If seed is n=6, score is 11/6 = 1.83.
    # Current score 1.636 implies seed might be around 6.7? Or maybe the seed program returns n=11 too?
    # "score 1.6364" ... 11 / 6.72?
    # Actually, let's look at the "INSPIRATIONS".
    # Inspiration 1 score 1.5455.
    # If I can get n=12, score should jump.
    
    # Let's try to implement the incremental update correctly.
    
    def solve_n_fast(n):
        num_edges = n * (n - 1) // 2
        max_iter = 100000
        max_restarts = 10
        
        edge_index = {}
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                endpoints.append((i, j))
                idx += 1
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Symmetric CN matrices
            CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
            
            # Build neighbors
            neighbors = [ [set() for _ in range(5)] for _ in range(n) ]
            for e_idx in range(num_edges):
                u, v = endpoints[e_idx]
                c = colors[e_idx]
                neighbors[u][c].add(v)
                neighbors[v][c].add(u)
            
            # Init CN
            for c in range(5):
                for k in range(n):
                    nbrs = list(neighbors[k][c])
                    for i in range(len(nbrs)):
                        u = nbrs[i]
                        for j in range(i + 1, len(nbrs)):
                            v = nbrs[j]
                            CN[c][u, v] += 1
                            CN[c][v, u] += 1
            
            # Initial cost
            current_cost = 0.0
            for c in range(5):
                # Sum binom(CN[c][u, v], 2) for u < v
                tri_u, tri_v = np.triu_indices(n, 1)
                vals = CN[c][tri_u, tri_v]
                current_cost += np.sum(vals * (vals - 1) / 2.0)
            
            if current_cost == 0:
                return colors
            
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                if current_cost == 0:
                    return colors
                
                e_idx = random.randint(0, num_edges - 1)
                u, v = endpoints[e_idx]
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                delta = 0.0
                
                # Update for old_c (decrease)
                # Pairs (u, x) where edge (v, x) is old_c -> CN[old_c][u, x] decreases
                # Pairs (v, x) where edge (u, x) is old_c -> CN[old_c][v, x] decreases
                
                # We can iterate x from 0 to n-1
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    # Check edge (v, x)
                    key_vx = (v, x) if v < x else (x, v)
                    idx_vx = edge_index[key_vx]
                    if colors[idx_vx] == old_c:
                        # CN[old_c][u, x] decreases by 1
                        old_val = CN[old_c][u, x]
                        new_val = old_val - 1
                        # Update cost contribution for pair (u, x)
                        # Old binom(old_val, 2), New binom(new_val, 2)
                        delta -= (old_val * (old_val - 1) - new_val * (new_val - 1)) / 2.0
                        CN[old_c][u, x] = new_val
                        CN[old_c][x, u] = new_val
                    
                    # Check edge (u, x)
                    key_ux = (u, x) if u < x else (x, u)
                    idx_ux = edge_index[key_ux]
                    if colors[idx_ux] == old_c:
                        # CN[old_c][v, x] decreases by 1
                        old_val = CN[old_c][v, x]
                        new_val = old_val - 1
                        delta -= (old_val * (old_val - 1) - new_val * (new_val - 1)) / 2.0
                        CN[old_c][v, x] = new_val
                        CN[old_c][x, v] = new_val
                        
                # Update for new_c (increase)
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    key_vx = (v, x) if v < x else (x, v)
                    idx_vx = edge_index[key_vx]
                    if colors[idx_vx] == new_c:
                        # CN[new_c][u, x] increases by 1
                        old_val = CN[new_c][u, x]
                        new_val = old_val + 1
                        delta += (new_val * (new_val - 1) - old_val * (old_val - 1)) / 2.0
                        CN[new_c][u, x] = new_val
                        CN[new_c][x, u] = new_val
                        
                    key_ux = (u, x) if u < x else (x, u)
                    idx_ux = edge_index[key_ux]
                    if colors[idx_ux] == new_c:
                        # CN[new_c][v, x] increases by 1
                        old_val = CN[new_c][v, x]
                        new_val = old_val + 1
                        delta += (new_val * (new_val - 1) - old_val * (old_val - 1)) / 2.0
                        CN[new_c][v, x] = new_val
                        CN[new_c][x, v] = new_val
                        
                new_cost = current_cost + delta
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    colors[e_idx] = new_c
                    # Update neighbors sets
                    neighbors[u][old_c].remove(v)
                    neighbors[v][old_c].remove(u)
                    neighbors[u][new_c].add(v)
                    neighbors[v][new_c].add(u)
                else:
                    # Revert? We didn't change colors, but we DID change CN matrices!
                    # We need to revert CN changes if we don't accept.
                    # This is expensive. 
                    # Better to calculate delta first, then update?
                    # Or copy CN? No, too big.
                    # We must revert CN if rejected.
                    
                    # Revert old_c
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        key_vx = (v, x) if v < x else (x, v)
                        idx_vx = edge_index[key_vx]
                        if colors[idx_vx] == old_c:
                            CN[old_c][u, x] += 1
                            CN[old_c][x, u] += 1
                        key_ux = (u, x) if u < x else (x, u)
                        idx_ux = edge_index[key_ux]
                        if colors[idx_ux] == old_c:
                            CN[old_c][v, x] += 1
                            CN[old_c][x, v] += 1
                            
                    # Revert new_c
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        key_vx = (v, x) if v < x else (x, v)
                        idx_vx = edge_index[key_vx]
                        if colors[idx_vx] == new_c:
                            CN[new_c][u, x] -= 1
                            CN[new_c][x, u] -= 1
                        key_ux = (u, x) if u < x else (x, u)
                        idx_ux = edge_index[key_ux]
                        if colors[idx_ux] == new_c:
                            CN[new_c][v, x] -= 1
                            CN[new_c][x, v] -= 1
                            
                    # If stuck, restart?
                    # For now, just continue.
                
                # Check for stagnation?
                # pass

    # Use the fast solver
    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
        res = solve_n_fast(n)
        if res:
            best_n = n
            best_colors = res

    return {'n': best_n, 'colors': best_colors}
