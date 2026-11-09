import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin
    
    n = 16
    m = 65
    
    # Bitmask representation
    adj = [0] * n
    deg = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def _score():
        """Compute max triangle degree sum."""
        best = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        c ^= lsb
        return best
    
    def _penalty_score():
        """Smooth surrogate: sum of squares of triangle degree sums above threshold.
        This encourages flattening the distribution of high-degree vertices in triangles."""
        total = 0.0
        threshold = 20  # Only penalize triangles with high degree sums
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > threshold:
                            total += (s - threshold) ** 2
                        c ^= lsb
        return total
    
    def _spectral_init():
        """Use spectral method to construct initial graph.
        Take eigenvectors of a random perturbed Laplacian to guide edge selection."""
        # Create random perturbation
        rng = np.random.default_rng(42)
        W = rng.random((n, n))
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0)
        
        # Compute Laplacian
        d = W.sum(axis=1)
        L = np.diag(d) - W
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            return False
        
        # Use top eigenvectors to score edges
        # Higher eigenvalue components indicate "community" boundaries
        # We want to favor edges that connect vertices with similar spectral positions
        # but also balance degrees
        
        # Get the first few non-trivial eigenvectors
        # (skip the trivial constant eigenvector)
        spectral_scores = np.zeros((n, n))
        
        # Use eigenvectors 1, 2, 3 (0-indexed, skipping 0)
        for k in [1, 2, 3]:
            if k < n:
                vec = eigenvectors[:, k]
                # Score edge (i,j) by similarity in spectral space
                # But we also want to encourage bipartite-like structure
                # Use difference in eigenvector values to encourage cross-partition edges
                for i in range(n):
                    for j in range(i + 1, n):
                        spectral_scores[i, j] += abs(vec[i] - vec[j])
        
        # Also add a degree-balancing term: prefer edges that connect
        # vertices that are currently under-represented
        # Start with uniform preference, will be refined
        
        # Rank all edges by spectral score
        edge_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_scores.append((spectral_scores[i, j], i, j))
        
        # Sort by score (higher = more preferred for cross-partition)
        edge_scores.sort(reverse=True)
        
        # Select top m edges, but ensure we get a valid simple graph
        # Also try to balance degrees
        selected = set()
        local_deg = [0] * n
        target_deg = m * 2.0 / n  # Average degree
        
        for score, i, j in edge_scores:
            if len(selected) >= m:
                break
            # Prefer edges that balance degrees
            deg_balance = abs(local_deg[i] - target_deg) + abs(local_deg[j] - target_deg)
            adjusted_score = score + 0.1 * deg_balance
            # Simple heuristic: just take the edge
            selected.add((i, j))
            local_deg[i] += 1
            local_deg[j] += 1
        
        # If we didn't get enough edges (shouldn't happen), fill randomly
        if len(selected) < m:
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)
            for i, j in all_edges:
                if len(selected) >= m:
                    break
                if (i, j) not in selected and (j, i) not in selected:
                    selected.add((i, j))
                    local_deg[i] += 1
                    local_deg[j] += 1
        
        # Build adjacency from selected edges
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        for i, j in selected:
            add_edge(i, j)
        
        return True
    
    # Try spectral initialization, fall back to K_{8,8}+1 if it fails
    spectral_ok = _spectral_init()
    if not spectral_ok:
        # Fallback: K_{8,8} + one edge in A
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
    
    cur_score = _score()
    cur_penalty = _penalty_score()
    best_score = cur_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    def pick_random_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and has_edge(u, v):
                return u, v
                
    def pick_random_non_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not has_edge(u, v):
                return u, v

    no_improve = 0
    max_no_improve = 1500
    T = 5.0
    cool_rate = 0.9995
    
    # Phase 1: Optimize penalty score (smooth surrogate) for a while
    # Phase 2: Switch to exact score optimization
    
    phase1_end = t0 + (DEADLINE - (time.time() - t0)) * 0.4  # Use 40% of time on penalty
    
    while time.time() - t0 < DEADLINE:
        in_phase1 = (time.time() - t0) < (phase1_end - t0)
        
        # Pick swap
        u1, v1 = pick_random_edge()
        u2, v2 = pick_random_non_edge()
        
        # Perform swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        if in_phase1:
            new_penalty = _penalty_score()
            new_score = _score()
            
            if new_penalty < cur_penalty:
                cur_penalty = new_penalty
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj = adj[:]
                    best_deg = deg[:]
            else:
                delta = new_penalty - cur_penalty
                if delta <= 2.0 and random.random() < 0.15:
                    cur_penalty = new_penalty
                    cur_score = new_score
                    no_improve += 1
                else:
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    no_improve += 1
        else:
            new_score = _score()
            
            if new_score < cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj = adj[:]
                    best_deg = deg[:]
            else:
                delta = new_score - cur_score
                if delta <= 1 and random.random() < 0.1:
                    cur_score = new_score
                    no_improve += 1
                else:
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    no_improve += 1
        
        T *= cool_rate
        if T < 0.5:
            T = 0.5
            
        if no_improve > max_no_improve:
            # Restart with perturbation
            for _ in range(5):
                eu1, ev1 = pick_random_edge()
                eu2, ev2 = pick_random_non_edge()
                remove_edge(eu1, ev1)
                add_edge(eu2, ev2)
            cur_score = _score()
            cur_penalty = _penalty_score()
            no_improve = 0
            T = 5.0
            max_no_improve = 1500
            
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
