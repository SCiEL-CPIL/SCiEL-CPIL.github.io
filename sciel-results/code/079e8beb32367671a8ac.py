"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Pre-compute adjacency structure for fast connectivity check
    adj = [[] for _ in range(n)]  # adj[u] = list of (v, edge_index)
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    def compute_radius_fast(signs):
        """Compute spectral radius using precomputed structure."""
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            S[u, v] = s
            S[v, u] = s
        eigvals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected_fast(signs):
        """Fast connectivity check using BFS with sign propagation."""
        label = [0] * n
        label[0] = 1
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            lu = label[u]
            for v, ei in adj[u]:
                s = signs[ei]
                expected = s * lu
                if label[v] == 0:
                    label[v] = expected
                    queue.append(v)
                elif label[v] != expected:
                    return True
        return False
    
    def evaluate(signs):
        """Check validity and compute radius. Returns (radius, is_valid)."""
        if not is_connected_fast(signs):
            return None, False
        try:
            radius, _ = check_lift(graph, signs)
            return radius, True
        except:
            return None, False
    
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Initial random sampling
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        radius, valid = evaluate(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        signs = np.ones(num_edges, dtype=np.int8)
        signs[0] = -1
        return signs
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Local search with single flips
    while time.time() < deadline:
        improved = False
        for i in range(num_edges):
            if time.time() >= deadline:
                break
            test_signs = current_signs.copy()
            test_signs[i] = -test_signs[i]
            radius, valid = evaluate(test_signs)
            if valid and radius < current_radius - 1e-12:
                current_signs = test_signs
                current_radius = radius
                best_signs = test_signs.copy()
                best_radius = radius
                improved = True
        
        if not improved:
            # Two-flip moves
            two_improved = False
            for i in range(num_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() >= deadline:
                        break
                    test_signs = current_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    radius, valid = evaluate(test_signs)
                    if valid and radius < current_radius - 1e-12:
                        current_signs = test_signs
                        current_radius = radius
                        best_signs = test_signs.copy()
                        best_radius = radius
                        two_improved = True
                        break
                if two_improved:
                    break
            
            if not two_improved:
                # Random restarts
                found = False
                for _ in range(500):
                    if time.time() >= deadline:
                        break
                    signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                    radius, valid = evaluate(signs)
                    if valid and radius < current_radius - 1e-12:
                        current_signs = signs
                        current_radius = radius
                        best_signs = signs.copy()
                        best_radius = radius
                        found = True
                        break
                if not found:
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
