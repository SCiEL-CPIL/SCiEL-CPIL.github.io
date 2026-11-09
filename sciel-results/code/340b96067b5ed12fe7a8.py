import time
import random
import math
import numpy as np

def construct():
    # We know that for n=47, we can try to construct a graph based on P(41).
    # P(41) is C4-free (41 % 4 == 1).
    # We add 6 vertices and connect them to P(41) vertices.
    
    n_target = 47
    min_deg_needed = n_target - 39 # 8
    
    def get_paley_edges(q):
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        edges = []
        for u in range(q):
            for v in range(u + 1, q):
                if (v - u) % q in residues:
                    edges.append([u, v])
        return edges

    # Start with P(41)
    q_base = 41
    base_edges = get_paley_edges(q_base)
    
    # Build adjacency matrix for P(41) extended to n_target
    A = np.zeros((n_target, n_target), dtype=bool)
    for u, v in base_edges:
        A[u, v] = True
        A[v, u] = True
        
    deg = np.zeros(n_target, dtype=int)
    for u, v in base_edges:
        deg[u] += 1
        deg[v] += 1
        
    # Initialize common neighbors
    # C[i][j] = number of common neighbors of i and j
    C = A.astype(int) @ A.astype(int)
    common = C
    
    # Add edges for vertices 41..46
    random.seed(42)
    
    for v in range(41, n_target):
        while deg[v] < min_deg_needed:
            found = False
            candidates = [u for u in range(q_base) if not A[u, v]]
            random.shuffle(candidates)
            
            for u in candidates:
                # Check safety
                if common[u, v] > 1:
                    continue
                
                safe = True
                # Check condition: For all w in N(v), common[u, w] must be 0
                for w in range(n_target):
                    if A[v, w] and w != u:
                        if common[u, w] > 0:
                            safe = False
                            break
                if not safe:
                    continue
                
                # Check condition: For all w in N(u), common[v, w] must be 0
                for w in range(n_target):
                    if A[u, w] and w != v:
                        if common[v, w] > 0:
                            safe = False
                            break
                if not safe:
                    continue
                
                # Safe to add
                A[u, v] = True
                A[v, u] = True
                deg[u] += 1
                deg[v] += 1
                
                # Update common
                for w in range(n_target):
                    if A[v, w] and w != u:
                        common[u, w] += 1
                        common[w, u] += 1
                        
                for w in range(n_target):
                    if A[u, w] and w != v:
                        common[v, w] += 1
                        common[w, v] += 1
                        
                found = True
                break
            
            if not found:
                break
                
    # Verify
    C_final = A.astype(int) @ A.astype(int)
    c4_free = True
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if C_final[i, j] >= 2:
                c4_free = False
                break
        if not c4_free:
            break
            
    if c4_free and np.min(deg) >= min_deg_needed:
        final_edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if A[i, j]:
                    final_edges.append([i, j])
        return {'n': n_target, 'edges': final_edges}
        
    # Fallback: Return P(41)
    base_edges_final = get_paley_edges(q_base)
    return {'n': q_base, 'edges': base_edges_final}
