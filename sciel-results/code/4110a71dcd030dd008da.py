import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s/150s limit
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # The seed program tries 4*n*d times. We replicate this exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def get_conflict_graph(edges, adj):
        n_edges = len(edges)
        # Map edge tuple to index
        edge_index = {e: i for i, e in enumerate(edges)}
        # H[i] will be a set of indices j such that edges[i] and edges[j] conflict
        H = [set() for _ in range(n_edges)]
        
        for i, e in enumerate(edges):
            u, v = e
            # 1. Edges sharing a vertex
            # Neighbors of u and v in G
            # Any edge incident to u or v conflicts with e
            # We can iterate through all edges? O(N^2) is fine for N~240
            # Or use adjacency lists
            
            # Let's just do O(N^2) check for simplicity and correctness first.
            for j, f in enumerate(edges):
                if i == j: continue
                x, y = f
                # Check share vertex
                if u == x or u == y or v == x or v == y:
                    H[i].add(j)
                    H[j].add(i)
                else:
                    # Check if joined by an edge
                    # (u,x) in adj? (u,y) in adj? (v,x) in adj? (v,y) in adj?
                    if adj.get(u, set()).__contains__(x) or \
                       adj.get(u, set()).__contains__(y) or \
                       adj.get(v, set()).__contains__(x) or \
                       adj.get(v, set()).__contains__(y):
                        H[i].add(j)
                        H[j].add(i)
        return H

    def greedy_color(H, n):
        # Simple greedy to get upper bound
        color = [-1] * n
        # Order by degree
        order = sorted(range(n), key=lambda i: len(H[i]), reverse=True)
        for v in order:
            used = set()
            for u in H[v]:
                if color[u] != -1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
        return color, max(color) + 1

    def solve_coloring(H, n, k, timeout_limit):
        # B&B to see if k colors suffice
        # H is list of sets of neighbors
        # We use recursion
        
        # Precompute neighbors lists for faster iteration
        neighbors = [list(s) for s in H]
        
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda i: len(neighbors[i]), reverse=True)
        # Remap indices? No, just iterate in this order.
        # We need to track which vertices are assigned.
        # assigned[v] = color or -1
        assigned = [-1] * n
        # available[v] = bitmask of available colors (1 << c)
        # Initially all k colors available
        full_mask = (1 << k) - 1
        available = [full_mask] * n
        
        # For MRV, we need to pick the unassigned vertex with fewest available bits
        # This is O(N) per step. N is small.
        
        def backtrack(idx):
            if time.time() > timeout_limit:
                return False, None
            
            if idx == n:
                return True, assigned[:]
            
            # Find MRV vertex among unassigned
            min_count = k + 1
            min_v = -1
            for v in range(n):
                if assigned[v] == -1:
                    count = bin(available[v]).count('1')
                    if count < min_count:
                        min_count = count
                        min_v = v
                        if count == 1:
                            break
                        if count == 0:
                            return False, None
            
            if min_v == -1:
                return True, assigned[:]
                
            v = min_v
            mask = available[v]
            
            # Try each available color
            while mask:
                # Pick a bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                # Assign c to v
                assigned[v] = c
                new_avail = available[:]
                new_avail[v] = 0 # Mark as assigned/used
                
                valid = True
                # Update neighbors
                for u in neighbors[v]:
                    if assigned[u] == -1:
                        if new_avail[u] & lsb:
                            new_avail[u] ^= lsb
                            if new_avail[u] == 0:
                                valid = False
                                break
                        else:
                            # Color already forbidden? Should not happen if we maintain correctly
                            pass
                    
                if valid:
                    success, res = backtrack(idx + 1)
                    if success:
                        return True, res
                    # If timeout, propagate? 
                    if time.time() > timeout_limit:
                        return False, None
                            
                # Unassign
                assigned[v] = -1
            
            return False, None

        success, res = backtrack(0)
        return success, res

    out = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    for n, d, seed in specs:
        edges = instance(n, d, seed)
        n_edges = len(edges)
        
        # Build adjacency dict for G
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
        # Build Conflict Graph H
        H = get_conflict_graph(edges, adj)
        
        # Get initial upper bound using Greedy
        greedy_col, k_upper = greedy_color(H, n_edges)
        
        # Lower bound
        max_deg = max(len(s) for s in H) if n_edges > 0 else 0
        k_lower = max(1, max_deg + 1)
        
        best_k = k_upper
        best_coloring = greedy_col
        
        # Try to improve with B&B
        # We try k from k_lower to k_upper - 1
        # If we find a valid coloring with k, we update best_k and best_coloring
        # We can stop if we reach k_lower (optimal)
        
        # To be safe with time, we might not find optimal for all, but we will try.
        # We iterate k from k_lower to best_k - 1
        for k in range(k_lower, best_k):
            if time.time() > deadline - 10: # Leave 10s for finishing
                break
            success, coloring = solve_coloring(H, n_edges, k, deadline - 5)
            if success:
                best_k = k
                best_coloring = coloring
                # If we hit the lower bound, we can't do better
                if k == k_lower:
                    break
            else:
                # If failed, try next k
                pass
        
        # If B&B didn't improve, use greedy result
        # The coloring list must correspond to the sorted edges
        # greedy_col and solve_coloring return colors for indices 0..n_edges-1
        # which correspond to sorted edges
        out.append(best_coloring)
        
    return out
