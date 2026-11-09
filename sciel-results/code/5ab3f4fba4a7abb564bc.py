import time
import random
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing 4-cycles.
    Strategy: Greedy constructive algorithm.
    1. Start with empty graph.
    2. Iteratively add the edge that creates the fewest new 4-cycles.
    3. Tie-break by degree balance to promote regularity.
    """
    n = 31
    m = 91
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Adjacency matrix for O(1) check
    adj = np.zeros((n, n), dtype=int)
    # Common neighbors count matrix
    # common[u, v] = number of vertices w such that adj[u,w] and adj[v,w] are 1
    common = np.zeros((n, n), dtype=int)
    
    degrees = np.zeros(n, dtype=int)
    
    edges = []
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    # We will pick edges one by one
    # To speed up, we can maintain a list of candidate edges and their costs
    # But n=31 is small, so O(n^2) per step is 961 ops. 91 steps * 961 * (update cost)
    # Update cost: when adding (u,v), we need to update common[u][w] and common[v][w] for all w.
    # Also common[w][x] changes if w is common neighbor of u and v? No.
    # common[u][v] is the cost.
    # After adding (u,v):
    # For any w != u,v:
    #   common[u][w] increases by 1 if adj[v][w] is 1.
    #   common[v][w] increases by 1 if adj[u][w] is 1.
    #   common[w][x] for w,x != u,v?
    #   The number of common neighbors of w and x increases by 1 if both u and v are common neighbors?
    #   No, adding edge (u,v) does not change the neighbor set of w or x, unless u or v IS w or x.
    #   So common[w][x] for w,x not in {u,v} remains unchanged?
    #   Wait. common[w][x] = |N(w) intersect N(x)|.
    #   N(w) and N(x) do not change if we add edge (u,v) and w,x != u,v.
    #   So only common[u][*] and common[v][*] change.
    
    for step in range(m):
        if time.time() > deadline:
            break
            
        best_cost = float('inf')
        best_edge = None
        best_tiebreak = float('inf')
        
        # Iterate all non-edges
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 0:
                    c = common[u, v]
                    cost = c * (c - 1) // 2  # C(c, 2)
                    
                    # Tie-breaker: prefer edges that balance degrees
                    # We want to add edges between low-degree vertices
                    # Score: degrees[u] + degrees[v] + abs(degrees[u] - degrees[v])
                    tiebreak = degrees[u] + degrees[v]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (u, v)
                        best_tiebreak = tiebreak
                    elif cost == best_cost:
                        if tiebreak < best_tiebreak:
                            best_tiebreak = tiebreak
                            best_edge = (u, v)
                            
        if best_edge is None:
            break
            
        u, v = best_edge
        edges.append([u, v])
        
        # Update graph state
        adj[u, v] = 1
        adj[v, u] = 1
        degrees[u] += 1
        degrees[v] += 1
        
        # Update common neighbors
        # For all w != u, v:
        # common[u][w] += adj[v][w]
        # common[v][w] += adj[u][w]
        # Also symmetric: common[w][u] += adj[v][w] etc.
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v, w] == 1:
                common[u, w] += 1
                common[w, u] += 1
            if adj[u, w] == 1:
                common[v, w] += 1
                common[w, v] += 1
        # Note: common[u][v] is not updated here, but it won't be used again as a candidate
        # because adj[u][v] is now 1.
        
    # If we didn't reach m edges due to timeout, we return what we have?
    # The task requires EXACTLY 91 edges.
    # If we time out, we must fill the rest arbitrarily to satisfy the contract, 
    # even if it increases the score, to avoid invalid result.
    if len(edges) < m:
        # Add remaining edges arbitrarily
        existing_set = set(map(tuple, edges))
        for u in range(n):
            for v in range(u + 1, n):
                if len(edges) >= m:
                    break
                if (u, v) not in existing_set:
                    edges.append([u, v])
                    existing_set.add((u, v))
            if len(edges) >= m:
                break
                
    return edges[:m]

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    e = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(e)}")
    # Count C4s for verification
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in e:
        adj[u, v] = 1
        adj[v, u] = 1
    # C4 count = 0.5 * sum_{u<v, not edge} C(common(u,v), 2)
    # Or simpler: (trace(A^4) - 2*sum(d_i) - sum(d_i^2) - 4*E) / 8 ?
    # Let's just use the common neighbor formula for non-adjacent pairs
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                c = np.sum(adj[u, :] * adj[v, :])
                c4_count += c * (c - 1) // 2
    # Wait, the formula is:
    # Each C4 has 2 pairs of opposite vertices.
    # For a pair of opposite vertices (u, v) which are NOT adjacent, the number of C4s 
    # with u, v as opposite vertices is C(common_neighbors(u, v), 2).
    # So total C4s = sum_{u<v, adj[u,v]==0} C(common(u,v), 2).
    # My code above does exactly that.
    print(f"C4 Count: {c4_count}")
