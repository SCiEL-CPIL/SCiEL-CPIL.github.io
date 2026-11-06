"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    n = 31
    target_edges = 91
    
    # Helper: count 4-cycles in a graph given adjacency matrix
    def count_c4(adj):
        # A 4-cycle is a set of 4 distinct vertices a,b,c,d with edges ab, bc, cd, da
        # We count each 4-cycle once by ordering: a < b, a < c, and a < d
        # Actually, let's use the standard method: for each pair (u,v) that is NOT an edge,
        # count common neighbors. Each 4-cycle contributes to exactly one non-edge pair?
        # No. Let's use: number of C4 = sum over all pairs {u,v} of C(common_neighbors(u,v), 2)
        # But this counts each C4 exactly once? 
        # A C4 is u-v-w-x-u. The pairs of opposite vertices are (u,w) and (v,x).
        # For pair (u,w), common neighbors are v and x. So C(2,2) = 1.
        # For pair (v,x), common neighbors are u and w. So C(2,2) = 1.
        # So each C4 is counted twice in the sum over ALL pairs.
        # So C4_count = (1/2) * sum_{u<v} C(common(u,v), 2)
        # But this includes pairs (u,v) that ARE edges. If u-v is an edge, common neighbors
        # form triangles with u-v, not 4-cycles. Wait, no.
        # If u and v are adjacent, and w,x are common neighbors, then u-w-v-x-u is a 4-cycle
        # ONLY IF u-x and w-v are edges? No.
        # Let's reconsider. A 4-cycle is a cycle of length 4.
        # The formula sum_{u<v} C(common(u,v), 2) counts the number of "diamonds" or 
        # pairs of triangles sharing an edge? No.
        # Actually, sum_{u<v} C(common(u,v), 2) counts the number of 4-cycles where 
        # u and v are OPPOSITE vertices. Each 4-cycle has exactly 2 pairs of opposite vertices.
        # So total C4 = (1/2) * sum_{u<v} C(common(u,v), 2).
        # This is correct regardless of whether u-v is an edge or not.
        
        # Efficient computation using matrix multiplication:
        # common(u,v) = (A @ A)[u,v] - A[u,v] (if A is 0-1 adjacency, no self-loops)
        # Actually (A^2)[u,v] = number of length-2 paths from u to v.
        # If u != v, this equals common(u,v) + (1 if u~v else 0)? No.
        # (A^2)[u,v] = sum_w A[u,w] A[w,v]. This counts common neighbors w.
        # If u and v are adjacent, w can be u or v? No, A[u,u]=0, A[v,v]=0.
        # So (A^2)[u,v] = number of common neighbors of u and v.
        
        A2 = adj @ adj
        # For u < v, common = A2[u,v]
        # C(common, 2) = common * (common - 1) / 2
        # Sum over u < v
        # We can compute the full sum and divide by 2, but we need to be careful.
        # Let's just sum over upper triangle.
        upper = np.triu_indices(n, k=1)
        commons = A2[upper]
        c4_count = np.sum(commons * (commons - 1)) // 2
        return c4_count
    
    def greedy_c4_free_base():
        """Construct a C4-free graph with 90 edges using greedy randomized search."""
        best_adj = None
        best_c4 = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 1000 + 42)
            adj = np.zeros((n, n), dtype=np.int32)
            edges = []
            
            # Greedy: add edges in random order, skip if it creates a C4
            candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(candidates)
            
            for u, v in candidates:
                if len(edges) >= 90:
                    break
                # Check if adding (u,v) creates a C4
                # A C4 is created if there exist w,x such that u~w, w~v, v~x, x~u
                # i.e., common neighbors of u and v that are connected through a path of length 2?
                # Actually, adding edge (u,v) creates a C4 if there is a path of length 3 from u to v
                # that doesn't use the edge (u,v). I.e., there exist w,x with u~w, w~x, x~v, 
                # and (u,v) not in path. But since we're adding (u,v), we need to check if 
                # there's already a u-w-x-v path.
                # This is equivalent to: are there w,x such that adj[u,w]=1, adj[w,x]=1, adj[x,v]=1?
                # And w != x, w != v, x != u.
                
                # Efficient check: for each common neighbor w of u and v? No.
                # We need a path u-w-x-v. So w is neighbor of u, x is neighbor of v, and w~x.
                # Let's iterate over neighbors of u and neighbors of v.
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                creates_c4 = False
                for w in neighbors_u:
                    if w == v:
                        continue
                    for x in neighbors_v:
                        if x == u or x == w:
                            continue
                        if adj[w, x] == 1:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                
                if not creates_c4:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edges.append((u, v))
            
            if len(edges) == 90:
                c4 = count_c4(adj)
                if c4 < best_c4:
                    best_c4 = c4
                    best_adj = adj.copy()
                    best_edges = list(edges)
            
            if best_c4 == 0:
                break
        
        return best_adj, best_edges
    
    # Step 1: Find a C4-free base with 90 edges
    base_adj, base_edges = greedy_c4_free_base()
    
    if base_adj is None:
        # Fallback: use a simpler construction
        base_adj = np.zeros((n, n), dtype=np.int32)
        base_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if len(base_edges) >= 90:
                    break
                base_adj[i, j] = 1
                base_adj[j, i] = 1
                base_edges.append((i, j))
            if len(base_edges) >= 90:
                break
    
    # Step 2: Add the 91st edge that creates the fewest new 4-cycles
    best_total_c4 = float('inf')
    best_final_edges = None
    
    # Try each possible edge not in base
    existing_edges = set(base_edges)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidate_edges = [e for e in all_possible if e not in existing_edges]
    
    for u, v in candidate_edges:
        if time.time() > deadline - 5:
            break
        
        # Temporarily add edge
        base_adj[u, v] = 1
        base_adj[v, u] = 1
        
        c4 = count_c4(base_adj)
        
        if c4 < best_total_c4:
            best_total_c4 = c4
            best_final_edges = base_edges + [(u, v)]
        
        # Remove edge
        base_adj[u, v] = 0
        base_adj[v, u] = 0
    
    if best_final_edges is None:
        # Fallback: just add the first candidate
        best_final_edges = base_edges + [candidate_edges[0]]
    
    return [[u, v] for u, v in best_final_edges]
# EVOLVE-BLOCK-END
