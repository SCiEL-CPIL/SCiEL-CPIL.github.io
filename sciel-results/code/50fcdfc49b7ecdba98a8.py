import time
import random
import numpy as np
from itertools import combinations

_START_TIME = None
_DEADLINE = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _add_edge_creates_q3(adj, u, v):
    """
    Check if adding edge (u,v) to graph with adjacency matrix adj creates a Q3 subgraph.
    Assumes (u,v) is NOT yet in adj.
    """
    # Get neighbors of u and v
    # We need x2, x4 such that (u,x2) and (u,x4) are in adj
    # We need x3, x5 such that (v,x3) and (v,x5) are in adj
    
    nu = [x for x in range(16) if x != u and x != v and adj[u, x]]
    nv = [x for x in range(16) if x != u and x != v and adj[v, x]]
    
    if len(nu) < 2 or len(nv) < 2:
        return False
    
    # Precompute intersections for speed?
    # For each pair (x2, x4) in nu, and (x3, x5) in nv:
    # x6 must be in N(x2) \cap N(x4)
    # x7 must be in N(x3) \cap N(x5)
    # (x6, x7) must be in adj
    # All 8 distinct.
    
    for x2 in nu:
        for x4 in nu:
            if x2 == x4: continue
            
            # Potential x6 candidates
            # x6 must be adjacent to x2 and x4
            # and x6 != u, v, x2, x4
            cand_x6 = [x for x in range(16) if x not in (u, v, x2, x4) and adj[x2, x] and adj[x4, x]]
            if not cand_x6: continue
            
            for x3 in nv:
                for x5 in nv:
                    if x3 == x5: continue
                    if x3 == x2 or x3 == x4 or x5 == x2 or x5 == x4: continue
                    
                    # Potential x7 candidates
                    cand_x7 = [x for x in range(16) if x not in (u, v, x2, x4, x3, x5) and adj[x3, x] and adj[x5, x]]
                    if not cand_x7: continue
                    
                    for x6 in cand_x6:
                        if x6 in (x3, x5): continue
                        for x7 in cand_x7:
                            if x7 == x6: continue
                            if adj[x6, x7]:
                                return True
    return False

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_edges_count = 0
    best_adj = np.zeros((16, 16), dtype=bool)
    
    # Try multiple random orders
    for _ in range(100): # Adjust based on time
        if _elapsed() > _DEADLINE:
            break
            
        adj = np.zeros((16, 16), dtype=bool)
        edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
        random.shuffle(edges)
        
        current_count = 0
        for u, v in edges:
            if _elapsed() > _DEADLINE:
                break
                
            if not _add_edge_creates_q3(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                current_count += 1
        
        if current_count > best_edges_count:
            best_edges_count = current_count
            best_adj = adj.copy()
            
        # Optional: Local search improvement
        # Try to add any skipped edges
        # This can be expensive, so maybe just do it for the best few?
        
    # Convert best_adj to list of edges
    result = []
    for i in range(16):
        for j in range(i+1, 16):
            if best_adj[i, j]:
                result.append([i, j])
                
    return result
