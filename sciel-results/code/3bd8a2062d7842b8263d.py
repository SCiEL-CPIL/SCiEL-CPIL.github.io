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
    
    def _softmax_score(beta=3.0):
        """Log-sum-exp soft approximation of max triangle degree sum.
        Smoother than hard max, better for gradient-free optimization."""
        # Collect all triangle degree sums
        sums = []
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
                        sums.append(s)
                        c ^= lsb
        if not sums:
            return 0.0
        # Log-sum-exp for numerical stability
        max_s = max(sums)
        # beta controls smoothness: larger = closer to hard max
        log_sum = sum(np.exp(beta * (s - max_s)) for s in sums)
        return max_s + np.log(log_sum) / beta
    
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
        spectral_scores = np.zeros((n, n))
        
        for k in [1, 2, 3]:
            if k < n:
                vec = eigenvectors[:, k]
                for i in range(n):
                    for j in range(i + 1, n):
                        spectral_scores[i, j] += abs(vec[i] - vec[j])
        
        # Rank all edges by spectral score
        edge_scores = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_scores.append((spectral_scores[i, j], i, j))
        
        edge_scores.sort(reverse=True)
        
        selected = set()
        local_deg = [0] * n
        target_deg = m * 2.0 / n
        
        for score, i, j in edge_scores:
            if len(selected) >= m:
                break
            selected.add((i, j))
            local_deg[i] += 1
            local_deg[j] += 1
        
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
        
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        for i, j in selected:
            add_edge(i, j)
        
        return True
    
    spectral_ok = _spectral_init()
    if not spectral_ok:
        for i in range(n):
            adj[i] = 0
            deg[i] = 0
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
    
    cur_score = _score()
    cur_soft = _softmax_score()
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
    
    phase1_end = t0 + (DEADLINE - (time.time() - t0)) * 0.4
    phase2_end = t0 + (DEADLINE - (time.time() - t0)) * 0.85
    
    while time.time() - t0 < phase2_end:
        in_phase1 = (time.time() - t0) < phase1_end
        
        u1, v1 = pick_random_edge()
        u2, v2 = pick_random_non_edge()
        
        remove_edge(u1, v1)
        add_edge(u2, v2)
        
        if in_phase1:
            new_soft = _softmax_score()
            new_score = _score()
            
            if new_soft < cur_soft:
                cur_soft = new_soft
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj = adj[:]
                    best_deg = deg[:]
            else:
                delta = new_soft - cur_soft
                if delta <= 2.0 and random.random() < 0.15:
                    cur_soft = new_soft
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
            for _ in range(5):
                eu1, ev1 = pick_random_edge()
                eu2, ev2 = pick_random_non_edge()
                remove_edge(eu1, ev1)
                add_edge(eu2, ev2)
            cur_score = _score()
            cur_soft = _softmax_score()
            no_improve = 0
            T = 5.0
            max_no_improve = 1500
            
    # Phase 3: Final polish - exhaustive single-swap local search on exact score
    adj[:] = best_adj
    deg[:] = best_deg
    cur_score = best_score
    
    improved = True
    while improved and time.time() - t0 < DEADLINE - 5:
        improved = False
        # Try all (remove edge, add non-edge) swaps
        for u1 in range(n):
            for v1 in range(u1 + 1, n):
                if not has_edge(u1, v1):
                    continue
                if time.time() - t0 > DEADLINE - 5:
                    break
                remove_edge(u1, v1)
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if u2 == v2 or has_edge(u2, v2):
                            continue
                        if time.time() - t0 > DEADLINE - 5:
                            add_edge(u1, v1)
                            break
                        add_edge(u2, v2)
                        new_score = _score()
                        if new_score < cur_score:
                            cur_score = new_score
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_adj = adj[:]
                                best_deg = deg[:]
                        else:
                            remove_edge(u2, v2)
                    if time.time() - t0 > DEADLINE - 5:
                        break
                add_edge(u1, v1)
            if time.time() - t0 > DEADLINE - 5:
                break
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
