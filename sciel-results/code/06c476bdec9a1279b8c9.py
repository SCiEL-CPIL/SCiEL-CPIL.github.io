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
    
    # Pre-compute edge indices
    edge_idx = []
    for u, v in edges:
        edge_idx.append((u, v))
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edge_idx):
            s = signs[i]
            S[u, v] = s
            S[v, u] = s
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the 2-lift is connected using product of signs around cycles."""
        # For a 2-lift, the lift is connected iff the product of signs around 
        # every fundamental cycle is -1 (i.e., at least one cycle has negative product)
        # Actually, the lift is disconnected iff the product of signs around EVERY 
        # cycle is +1, which means there exists a vertex labeling with +/-1 such that
        # for every edge (u,v), sign(u)*sign(v) = sign(edge).
        # 
        # Equivalently: BFS/DFS assigning labels. Start with label[0]=1.
        # For each edge (u,v) with sign s: label[v] = s * label[u].
        # If consistent, lift is disconnected. If inconsistent, lift is connected.
        
        label = [0] * n  # 0 = unvisited, 1 or -1 = assigned
        label[0] = 1
        
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for i, (a, b) in enumerate(edge_idx):
                if a == u:
                    v = b
                elif b == u:
                    v = a
                else:
                    continue
                
                s = signs[i]
                expected_label_v = s * label[u]
                
                if label[v] == 0:
                    label[v] = expected_label_v
                    queue.append(v)
                elif label[v] != expected_label_v:
                    # Inconsistency means the lift IS connected
                    return True
        
        # All labels consistent, lift is disconnected
        return False
    
    def check_and_evaluate(signs):
        """Check validity and compute radius. Returns (radius, is_valid)."""
        if not is_valid_lift(signs):
            return None, False
        try:
            radius, details = check_lift(graph, signs)
            return radius, True
        except:
            return None, False
    
    # Start with random search using the seed's approach
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # First, try the seed's approach: sample budget random vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        radius, valid = check_and_evaluate(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try single -1 edge patterns
        for i in range(num_edges):
            signs = np.ones(num_edges, dtype=np.int8)
            signs[i] = -1
            radius, valid = check_and_evaluate(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    if best_signs is None:
        # Try pairs of -1 edges
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                signs = np.ones(num_edges, dtype=np.int8)
                signs[i] = -1
                signs[j] = -1
                radius, valid = check_and_evaluate(signs)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        # Emergency fallback - try all -1
        signs = np.ones(num_edges, dtype=np.int8)
        signs[0] = -1
        return signs
    
    # Now do local search to improve
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Local search: flip one edge at a time, with multiple passes
    while time.time() < deadline:
        improved = False
        
        # Single flip moves
        for i in range(num_edges):
            if time.time() >= deadline:
                break
            
            test_signs = current_signs.copy()
            test_signs[i] = -test_signs[i]
            
            radius, valid = check_and_evaluate(test_signs)
            if valid and radius < current_radius - 1e-10:
                current_signs = test_signs
                current_radius = radius
                best_signs = test_signs.copy()
                best_radius = radius
                improved = True
        
        if not improved:
            # Try 2-flip moves
            two_flip_improved = False
            for i in range(num_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() >= deadline:
                        break
                    
                    test_signs = current_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    
                    radius, valid = check_and_evaluate(test_signs)
                    if valid and radius < current_radius - 1e-10:
                        current_signs = test_signs
                        current_radius = radius
                        best_signs = test_signs.copy()
                        best_radius = radius
                        two_flip_improved = True
                        break
                if two_flip_improved:
                    break
            
            if not two_flip_improved:
                # Try 3-flip moves (limited)
                three_flip_improved = False
                for i in range(num_edges):
                    if time.time() >= deadline:
                        break
                    for j in range(i + 1, num_edges):
                        if time.time() >= deadline:
                            break
                        for k in range(j + 1, num_edges):
                            if time.time() >= deadline:
                                break
                            
                            test_signs = current_signs.copy()
                            test_signs[i] = -test_signs[i]
                            test_signs[j] = -test_signs[j]
                            test_signs[k] = -test_signs[k]
                            
                            radius, valid = check_and_evaluate(test_signs)
                            if valid and radius < current_radius - 1e-10:
                                current_signs = test_signs
                                current_radius = radius
                                best_signs = test_signs.copy()
                                best_radius = radius
                                three_flip_improved = True
                                break
                        if three_flip_improved:
                            break
                    if three_flip_improved:
                        break
                
                if not three_flip_improved:
                    # Try random restarts
                    restart_improved = False
                    for _ in range(200):
                        if time.time() >= deadline:
                            break
                        
                        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                        
                        radius, valid = check_and_evaluate(signs)
                        if valid and radius < current_radius - 1e-10:
                            current_signs = signs
                            current_radius = radius
                            best_signs = signs.copy()
                            best_radius = radius
                            restart_improved = True
                            break
                    
                    if not restart_improved:
                        # No improvement found, stop
                        break
    
    return best_signs
# EVOLVE-BLOCK-END
