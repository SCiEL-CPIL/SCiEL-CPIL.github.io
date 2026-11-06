import time
import random
import numpy as np
from itertools import combinations

N = 20

# Precompute triples for each vertex
triples_for_vertex = [[] for _ in range(N)]
triples_indices = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            t_idx = len(triples_indices)
            triples_indices.append((i,j,k))
            triples_for_vertex[i].append((t_idx, j, k))
            triples_for_vertex[j].append((t_idx, i, k))
            triples_for_vertex[k].append((t_idx, i, j))

total_triples = len(triples_indices)
all_edges = [(i,j) for i in range(N) for j in range(i+1, N)]

def can_add_edge(M, C, u, v):
    """Check if adding edge (u,v) keeps the graph K3,3-free."""
    # Check triples containing u
    for t_idx, x, y in triples_for_vertex[u]:
        if M[x, v] == 1 and M[y, v] == 1:
            if C[t_idx] == 2:
                return False
    # Check triples containing v
    for t_idx, x, y in triples_for_vertex[v]:
        if M[x, u] == 1 and M[y, u] == 1:
            if C[t_idx] == 2:
                return False
    return True

def add_edge(M, C, edges, u, v):
    """Add edge (u,v) to the graph."""
    M[u, v] = 1
    M[v, u] = 1
    edges.append([u, v])
    # Update C
    for t_idx, x, y in triples_for_vertex[u]:
        if M[x, v] == 1 and M[y, v] == 1:
            C[t_idx] += 1
    for t_idx, x, y in triples_for_vertex[v]:
        if M[x, u] == 1 and M[y, u] == 1:
            C[t_idx] += 1

def remove_edge(M, C, edges, u, v):
    """Remove edge (u,v) from the graph."""
    M[u, v] = 0
    M[v, u] = 0
    edges.remove([u, v])
    # Update C
    for t_idx, x, y in triples_for_vertex[u]:
        if M[x, v] == 1 and M[y, v] == 1:
            C[t_idx] -= 1
    for t_idx, x, y in triples_for_vertex[v]:
        if M[x, u] == 1 and M[y, u] == 1:
            C[t_idx] -= 1

def local_search(M, C, edges, deadline):
    """Perform local search to improve the graph."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        for u, v in all_edges:
            if M[u, v] == 0:
                if can_add_edge(M, C, u, v):
                    add_edge(M, C, edges, u, v)
                    improved = True
                    break
        
        # Phase 2: If no edge could be added, try edge swaps
        if not improved:
            # Try removing one edge and adding two new ones
            found_swap = False
            for e1 in list(edges):
                u1, v1 = e1[0], e1[1]
                # Remove e1
                remove_edge(M, C, edges, u1, v1)
                
                # Try to add two new edges
                added = []
                for u2, v2 in all_edges:
                    if M[u2, v2] == 0 and (u2,v2) != (u1,v1) and (u2,v2) != (v1,u1):
                        if can_add_edge(M, C, u2, v2):
                            add_edge(M, C, edges, u2, v2)
                            added.append((u2, v2))
                            if len(added) == 2:
                                break
                
                if len(added) >= 2:
                    found_swap = True
                    improved = True
                    break
                else:
                    # Undo: add back e1, remove added edges
                    for u2, v2 in added:
                        remove_edge(M, C, edges, u2, v2)
                    add_edge(M, C, edges, u1, v1)
            
            if not found_swap:
                break
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_count = 0
    
    # Try multiple random restarts with greedy + local search
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Shuffle edge order for greedy
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        M = np.zeros((N, N), dtype=np.int8)
        C = np.zeros(total_triples, dtype=np.int8)
        edges = []
        
        # Greedy construction
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            if can_add_edge(M, C, u, v):
                add_edge(M, C, edges, u, v)
        
        # Local search
        if time.time() < deadline:
            edges = local_search(M, C, edges, deadline)
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = [e[:] for e in edges]
        
        # If we've found a good solution, we can stop early
        if best_count >= 75:
            break
    
    if best_edges is None:
        # Fallback: the seed K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
