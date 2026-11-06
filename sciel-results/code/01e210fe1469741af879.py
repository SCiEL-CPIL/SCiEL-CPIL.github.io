import time
import random
import numpy as np

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Check: does there exist b1, b2 in N(u) such that |N(b1) ∩ N(b2) ∩ N(v)| >= 2?
    # This would form K3,3 with parts {u, a1, a2} and {v, b1, b2}
    # where a1, a2 are in N(b1) ∩ N(b2) ∩ N(v)
    
    # Get neighbors of u as a bitmask
    # For each pair of neighbors b1, b2 of u, check if they have >= 2 common neighbors in N(v)
    neighbors_u = [x for x in range(N) if (nu >> x) & 1]
    
    for i in range(len(neighbors_u)):
        for j in range(i + 1, len(neighbors_u)):
            b1 = neighbors_u[i]
            b2 = neighbors_u[j]
            common = adj[b1] & adj[b2] & nv
            if common.bit_count() >= 2:
                return False
    
    return True

def add_edge(u, v, adj):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def remove_edge(u, v, adj):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def local_search(adj, deadline):
    """
    Perform local search on the graph.
    Returns the best adj found.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        for u, v in ALL_EDGES:
            if time.time() > deadline:
                break
            if not (adj[u] >> v) & 1:
                if is_valid_add(u, v, adj):
                    add_edge(u, v, adj)
                    improved = True
                    break
        
        if time.time() > deadline:
            break
            
        # Phase 2: If no edge could be added, try 1-swap (remove 1, add 2)
        if not improved:
            current_edges = []
            for u in range(N):
                for v in range(u+1, N):
                    if (adj[u] >> v) & 1:
                        current_edges.append((u, v))
                        
            random.shuffle(current_edges)
            
            for u1, v1 in current_edges:
                if time.time() > deadline:
                    break
                
                # Save state
                saved_adj = adj[:]
                
                # Remove edge (u1, v1)
                remove_edge(u1, v1, adj)
                
                added_count = 0
                added_edges = []
                
                # Try to add 2 edges
                for u2, v2 in ALL_EDGES:
                    if time.time() > deadline:
                        break
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if not (adj[u2] >> v2) & 1:
                        if is_valid_add(u2, v2, adj):
                            add_edge(u2, v2, adj)
                            added_edges.append((u2, v2))
                            added_count += 1
                            if added_count == 2:
                                break
                
                if added_count == 2:
                    improved = True
                    break
                else:
                    # Restore state
                    adj = saved_adj
            
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts
    for restart in range(500):
        if time.time() > deadline - 5:  # Leave 5s for final processing
            break
        
        # Greedy construction with random edge order
        adj = [0] * N
        shuffled_edges = ALL_EDGES[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 5:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
        
        # Local Search
        adj = local_search(adj, deadline - 2)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Early stop if we hit a high threshold
        if best_count >= 80:
            break

    if best_adj is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
