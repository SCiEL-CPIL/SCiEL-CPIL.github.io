import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Deadline guard: 55 seconds to be safe (evaluator kills at 90s)
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute edge mapping for fast access
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i
        
    # Precompute adjacency list for balance checking
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Helper to check if a signing is balanced (lift disconnected)
    # Returns True if balanced, False if unbalanced (valid connected lift)
    def is_balanced(sign_dict):
        pot = [0] * n
        pot[0] = 1
        dq = deque([0])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                s = sign_dict[frozenset((u, v))]
                if pot[v] == 0:
                    pot[v] = pot[u] * s
                    dq.append(v)
                else:
                    if pot[v] != pot[u] * s:
                        return False  # Unbalanced -> connected lift
        return True  # Balanced -> disconnected lift (invalid)
    
    # Helper to compute spectral radius from a sign dict
    def compute_radius(sign_dict):
        A = np.zeros((n, n))
        for (u, v) in edges:
            s = sign_dict[frozenset((u, v))]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))
    
    # Helper to build sign array from dict
    def build_signs_from_dict(sign_dict):
        signs = np.ones(E, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            signs[i] = sign_dict[frozenset((u, v))]
        return signs
    
    # Helper to get sign dict from array
    def get_sign_dict(signs_arr):
        sign_dict = {}
        for i, (u, v) in enumerate(edges):
            sign_dict[frozenset((u, v))] = int(signs_arr[i])
        return sign_dict
    
    # 1. Structured Enumeration
    # Define the spanning tree: matching (i, i+m) and bottom path (i, i+1) for i in 0..m-2
    tree_edges = set()
    for i in range(m):
        tree_edges.add(frozenset((i, m+i)))
    for i in range(m-1):
        tree_edges.add(frozenset((i, i+1)))
    
    chord_edges = []
    chord_edges.append(frozenset((m-1, 0)))
    for i in range(m):
        chord_edges.append(frozenset((m+i, m+(i+1)%m)))
        
    num_chords = len(chord_edges)
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate 2^(m+1) patterns
    # For m=4,5,7,9, this is 2^5, 2^6, 2^8, 2^10 = 32, 64, 256, 1024. Very fast.
    for pattern in range(2 ** num_chords):
        if time.time() > deadline:
            break
            
        sign_dict = {}
        for e in tree_edges:
            sign_dict[e] = 1
        for j, e in enumerate(chord_edges):
            sign_dict[e] = 1 if (pattern >> j) & 1 else -1
        
        # Quick balance check before expensive eigvalsh
        if is_balanced(sign_dict):
            continue
            
        radius = compute_radius(sign_dict)
        if radius < best_radius:
            best_radius = radius
            best_signs = build_signs_from_dict(sign_dict)

    if best_signs is None:
        # Fallback if structured enumeration failed (should not happen)
        best_signs = np.ones(E, dtype=np.int8)
        best_radius = 3.0

    # 2. Local Search with 2-flip moves
    # Start from the best structured solution
    current_signs = best_signs.copy()
    current_sign_dict = get_sign_dict(current_signs)
    current_radius = best_radius
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flip
        for i in range(E):
            if time.time() > deadline:
                break
                
            # Flip edge i
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            u, v = edges[i]
            old_s_dict_val = current_sign_dict[frozenset((u, v))]
            current_sign_dict[frozenset((u, v))] = int(current_signs[i])
            
            if is_balanced(current_sign_dict):
                current_signs[i] = old_sign
                current_sign_dict[frozenset((u, v))] = old_s_dict_val
                continue
                
            new_radius = compute_radius(current_sign_dict)
            if new_radius < current_radius - 1e-9:
                current_radius = new_radius
                best_radius = new_radius
                best_signs = current_signs.copy()
                improved = True
            else:
                # Revert
                current_signs[i] = old_sign
                current_sign_dict[frozenset((u, v))] = old_s_dict_val

        # Double flip (if time permits)
        if time.time() > deadline - 5.0:
            break
            
        for i in range(E):
            if time.time() > deadline - 2.0:
                break
            for j in range(i+1, E):
                if time.time() > deadline - 1.0:
                    break
                    
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                old_sign_i = current_signs[i]
                old_sign_j = current_signs[j]
                
                current_signs[i] = -old_sign_i
                current_signs[j] = -old_sign_j
                current_sign_dict[frozenset((u1, v1))] = int(current_signs[i])
                current_sign_dict[frozenset((u2, v2))] = int(current_signs[j])
                
                if is_balanced(current_sign_dict):
                    current_signs[i] = old_sign_i
                    current_signs[j] = old_sign_j
                    current_sign_dict[frozenset((u1, v1))] = int(old_sign_i)
                    current_sign_dict[frozenset((u2, v2))] = int(old_sign_j)
                    continue
                    
                new_radius = compute_radius(current_sign_dict)
                if new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    best_radius = new_radius
                    best_signs = current_signs.copy()
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old_sign_i
                    current_signs[j] = old_sign_j
                    current_sign_dict[frozenset((u1, v1))] = int(old_sign_i)
                    current_sign_dict[frozenset((u2, v2))] = int(old_sign_j)

    # 3. Multi-start Local Search if time permits
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(3)
        # Try more random starts to explore landscape
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Generate random signs
            rand_signs = rng.choice([-1, 1], size=E).astype(np.int8)
            rand_dict = get_sign_dict(rand_signs)
            
            # Quick validity check
            if is_balanced(rand_dict):
                continue
                
            r = compute_radius(rand_dict)
            
            if r >= best_radius:
                continue
                
            # If better, do local search from here
            current_signs = rand_signs.copy()
            current_sign_dict = rand_dict
            current_radius = r
            best_radius = r
            best_signs = rand_signs.copy()
            
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for i in range(E):
                    if time.time() > deadline - 2.0:
                        break
                    u, v = edges[i]
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    current_sign_dict[frozenset((u, v))] = int(current_signs[i])
                    
                    if is_balanced(current_sign_dict):
                        current_signs[i] = old_sign
                        current_sign_dict[frozenset((u, v))] = int(old_sign)
                        continue
                        
                    new_radius = compute_radius(current_sign_dict)
                    if new_radius < current_radius - 1e-9:
                        current_radius = new_radius
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                        improved = True
                    else:
                        current_signs[i] = old_sign
                        current_sign_dict[frozenset((u, v))] = int(old_sign)

    return best_signs
