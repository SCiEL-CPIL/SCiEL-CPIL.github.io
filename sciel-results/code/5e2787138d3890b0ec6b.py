import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge list
    edge_list = [(u, v) for u, v in edges]
    
    # Precompute adjacency for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edge_list):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Precompute edge types: cycle_top, cycle_bottom, matching
    edge_types = []
    for u, v in edge_list:
        if u < m and v < m:
            edge_types.append('top')
        elif u >= m and v >= m:
            edge_types.append('bottom')
        else:
            edge_types.append('match')
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edge_list):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        # Check if signing is a coboundary using BFS
        t = np.zeros(n, dtype=int)
        t[0] = 1
        
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, ei in adj[u]:
                if not visited[v]:
                    t[v] = t[u] * signs[ei]
                    visited[v] = True
                    queue.append(v)
                else:
                    if t[v] != t[u] * signs[ei]:
                        return True
        return False
    
    def get_structured_signing(pattern_type, rng):
        """Generate structured sign patterns based on prism symmetry"""
        signs = np.ones(num_edges, dtype=int)
        
        if pattern_type == 0:
            # Alternating on top cycle
            for i, (u, v) in enumerate(edge_list):
                if edge_types[i] == 'top':
                    signs[i] = 1 if (u + v) % 4 == 0 else -1
        elif pattern_type == 1:
            # Alternating on bottom cycle
            for i, (u, v) in enumerate(edge_list):
                if edge_types[i] == 'bottom':
                    signs[i] = 1 if (u + v) % 4 == 0 else -1
        elif pattern_type == 2:
            # Alternating on matching
            for i, (u, v) in enumerate(edge_list):
                if edge_types[i] == 'match':
                    signs[i] = 1 if u % 2 == 0 else -1
        elif pattern_type == 3:
            # Random with bias toward anti-symmetric patterns
            for i in range(num_edges):
                if rng.random() < 0.3:
                    signs[i] = -1
        elif pattern_type == 4:
            # Try to make product around each cycle = -1
            # Top cycle edges
            top_edges = [i for i in range(num_edges) if edge_types[i] == 'top']
            bottom_edges = [i for i in range(num_edges) if edge_types[i] == 'bottom']
            match_edges = [i for i in range(num_edges) if edge_types[i] == 'match']
            
            # Make top cycle product = -1
            if top_edges:
                signs[top_edges[-1]] = -1
                for i in top_edges[:-1]:
                    signs[i] = 1
            # Make bottom cycle product = -1
            if bottom_edges:
                signs[bottom_edges[-1]] = -1
                for i in bottom_edges[:-1]:
                    signs[i] = 1
            # Matching edges random
            for i in match_edges:
                signs[i] = rng.choice([-1, 1])
        
        return signs
    
    best_radius = float('inf')
    best_signs = None
    
    rng = np.random.default_rng(3)
    
    # Phase 1: Try structured initial patterns
    for pattern_type in range(5):
        if time.time() > deadline - 5.0:
            break
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            signs = get_structured_signing(pattern_type, rng)
            if is_connected_lift(signs):
                current_radius = spectral_radius(signs)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = signs.copy()
    
    # Phase 2: Random restarts with local search
    max_restarts = 150
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random valid signing
        signs = None
        for _ in range(1000):
            candidate = rng.choice([-1, 1], size=num_edges)
            if is_connected_lift(candidate):
                signs = candidate
                break
        
        if signs is None:
            continue
        
        current_radius = spectral_radius(signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
        
        # Local search: multiple passes with shuffled edge order
        improved = True
        max_passes = 5
        for _pass in range(max_passes):
            if time.time() > deadline - 3.0:
                break
            improved = False
            edge_order = rng.permutation(num_edges)
            for i in edge_order:
                if time.time() > deadline - 2.0:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                if is_connected_lift(signs):
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                        if current_radius < best_radius:
                            best_radius = current_radius
                            best_signs = signs.copy()
                    else:
                        signs[i] = old_sign
                else:
                    signs[i] = old_sign
            if not improved:
                break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
    
    # Phase 3: Final polish - try all single-edge flips from best_signs
    if best_signs is not None and time.time() < deadline - 2.0:
        for i in range(num_edges):
            if time.time() > deadline - 1.0:
                break
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            if is_connected_lift(best_signs):
                new_radius = spectral_radius(best_signs)
                if new_radius < best_radius:
                    best_radius = new_radius
                else:
                    best_signs[i] = old_sign
            else:
                best_signs[i] = old_sign
    
    # Phase 4: Second polish pass if time permits
    if best_signs is not None and time.time() < deadline - 1.0:
        for i in range(num_edges):
            if time.time() > deadline - 0.5:
                break
            old_sign = best_signs[i]
            best_signs[i] = -old_sign
            if is_connected_lift(best_signs):
                new_radius = spectral_radius(best_signs)
                if new_radius < best_radius:
                    best_radius = new_radius
                else:
                    best_signs[i] = old_sign
            else:
                best_signs[i] = old_sign
    
    return best_signs
