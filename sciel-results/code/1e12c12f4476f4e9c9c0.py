"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for speed
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        np.add.at(A, (eu, ev), signs.astype(np.float64))
        np.add.at(A, (ev, eu), signs.astype(np.float64))
        return A
    
    def spectral_radius(A):
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        # Fast connectivity check using the twist structure
        # The 2-lift is connected iff the "twist graph" (subgraph of -1 edges)
        # has an odd number of twists on every cycle that separates the two sides
        # Simpler: do BFS on the lift but with precomputed adjacency
        # For prism graphs, use the fact that connectivity depends on sign products
        
        # Build lift adjacency efficiently
        # Each vertex v in base has 3 neighbors in base
        # In lift: (v,0) connects to (u, s(v,u)) and (v,1) connects to (u, 1-s(v,u))
        # where s(v,u) = 0 if sign=+1, 1 if sign=-1
        
        # Use a visited array and BFS
        visited = np.zeros(2 * n, dtype=bool)
        queue = [0]  # (vertex 0, layer 0)
        visited[0] = True
        
        # Precompute for each edge: sign as 0 (+1) or 1 (-1)
        # twist[i] = 0 if signs[i]==1 else 1
        twist = (signs == -1).astype(np.int8)
        
        # For each vertex, we need to know its 3 neighbors and the corresponding edge signs
        # Build vertex-to-edge mapping
        # Each vertex in prism has exactly 3 incident edges
        vert_edges = [[] for _ in range(n)]
        for i in range(num_edges):
            vert_edges[eu[i]].append(i)
            vert_edges[ev[i]].append(i)
        
        while queue:
            node = queue.pop(0)
            v = node // 2
            layer = node % 2
            for ei in vert_edges[v]:
                u = eu[ei] if ev[ei] == v else ev[ei]
                # The sign of edge ei
                t = twist[ei]
                # From (v, layer), we go to (u, layer ^ t)
                nl = layer ^ t
                nn = u * 2 + nl
                if not visited[nn]:
                    visited[nn] = True
                    queue.append(nn)
        
        return all(visited)
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    # Local search with incremental updates
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Helper: try a move and return new radius if valid, else None
    def try_move(signs, A, indices):
        """Flip signs at given indices, check validity, return (new_A, new_radius) or None"""
        old_signs = signs[list(indices)].copy()
        for idx in indices:
            signs[idx] = -signs[idx]
        
        if not is_valid(signs):
            for j, idx in enumerate(indices):
                signs[idx] = old_signs[j]
            return None
        
        # Update A incrementally
        for idx in indices:
            u, v = eu[idx], ev[idx]
            old_s = old_signs[list(indices).index(idx)] if len(indices) == 1 else -signs[idx]
            # Actually, let's just rebuild for multi-edge moves (rare)
            pass
        
        # For single edge flip, update A in place
        if len(indices) == 1:
            idx = indices[0]
            u, v = eu[idx], ev[idx]
            new_s = float(signs[idx])
            old_s = -new_s
            A[u, v] = new_s
            A[v, u] = new_s
        
        A_new = build_signed_matrix(signs)
        r = spectral_radius(A_new)
        
        # Restore if we'll reject (handled by caller)
        return (A_new, r)
    
    # Better approach: keep A in sync with signs
    def flip_and_eval(signs, A, idx):
        """Flip edge idx, update A, check validity, return radius or None (restoring if invalid)"""
        u, v = eu[idx], ev[idx]
        old_s = float(signs[idx])
        signs[idx] = -signs[idx]
        new_s = float(signs[idx])
        
        A[u, v] = new_s
        A[v, u] = new_s
        
        if not is_valid(signs):
            # Restore
            signs[idx] = -signs[idx]
            A[u, v] = old_s
            A[v, u] = old_s
            return None
        
        r = spectral_radius(A)
        return r
    
    def flip_two_and_eval(signs, A, i1, i2):
        """Flip edges i1 and i2, update A, check validity, return radius or None"""
        u1, v1 = eu[i1], ev[i1]
        u2, v2 = eu[i2], ev[i2]
        old_s1 = float(signs[i1])
        old_s2 = float(signs[i2])
        
        signs[i1] = -signs[i1]
        signs[i2] = -signs[i2]
        new_s1 = float(signs[i1])
        new_s2 = float(signs[i2])
        
        A[u1, v1] = new_s1
        A[v1, u1] = new_s1
        A[u2, v2] = new_s2
        A[v2, u2] = new_s2
        
        if not is_valid(signs):
            # Restore
            signs[i1] = -signs[i1]
            signs[i2] = -signs[i2]
            A[u1, v1] = old_s1
            A[v1, u1] = old_s1
            A[u2, v2] = old_s2
            A[v2, u2] = old_s2
            return None
        
        r = spectral_radius(A)
        return r
    
    # Try multiple random restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Ensure valid start
        if not is_valid(signs):
            valid = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                flip_idx = rng.integers(num_edges)
                signs[flip_idx] *= -1
                if is_valid(signs):
                    valid = True
                    break
            if not valid:
                continue
        
        A = build_signed_matrix(signs)
        cur_radius = spectral_radius(A)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # Local search with single and double flips
        improved = True
        stall_count = 0
        while improved and time.time() < deadline:
            improved = False
            # Shuffle edge order for diversity
            order = rng.permutation(num_edges)
            
            # Single edge flips
            for idx in order:
                if time.time() > deadline:
                    break
                new_r = flip_and_eval(signs, A, idx)
                if new_r is not None and new_r < cur_radius:
                    cur_radius = new_r
                    improved = True
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
            
            # Double edge flips (for escaping local optima)
            if not improved and stall_count < 5:
                for _ in range(min(20, num_edges * (num_edges - 1) // 2)):
                    if time.time() > deadline:
                        break
                    i1, i2 = rng.choice(num_edges, size=2, replace=False)
                    new_r = flip_two_and_eval(signs, A, i1, i2)
                    if new_r is not None and new_r < cur_radius:
                        cur_radius = new_r
                        improved = True
                        if cur_radius < best_radius:
                            best_radius = cur_radius
                            best_signs = signs.copy()
            
            if not improved:
                stall_count += 1
            else:
                stall_count = 0
    
    # If no valid solution found, return all +1
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(2000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
