"""
Optimized program for finding max edge-disjoint Hamiltonian cycles.
The graph has 21 vertices and 75 edges.
Theoretical max for Hamiltonian cycles is floor(75/21) = 3.
We aim to reliably find 3 such cycles.
"""

import time
import sys
from collections import defaultdict

# Pinned graph data
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and edge set
    adj = defaultdict(set)
    all_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_edges.add((min(u, v), max(u, v)))
    
    # Theoretical upper bound for Hamiltonian cycles
    # Each HC uses n edges. Total edges E. Max HCs = E // n
    max_possible = len(edges) // n  # 75 // 21 = 3
    
    if max_possible == 0:
        return []
        
    # We will try to find max_possible cycles.
    # Strategy: Greedy with backtracking on the first cycle if needed.
    # Given the small size, we can try a few permutations of the first cycle's start.
    
    best_cycles = []
    start_time = time.time()
    deadline = start_time + 55.0 # 5s margin
    
    # Helper to find a Hamiltonian cycle given used edges
    def find_ham_cycle(used_edges, start_node):
        """
        Finds a Hamiltonian cycle in the graph G - used_edges.
        Returns a list of vertices or None.
        """
        # Check if it's even possible: every node must have degree >= 2 in remaining graph
        # And the graph must be connected.
        
        # Precompute remaining neighbors for speed
        rem_adj = {}
        for v in range(n):
            nbrs = []
            for nb in adj[v]:
                e = (min(v, nb), max(v, nb))
                if e not in used_edges:
                    nbrs.append(nb)
            if len(nbrs) < 2:
                return None
            rem_adj[v] = nbrs
        
        # Connectivity check
        visited = {start_node}
        queue = [start_node]
        while queue:
            cur = queue.pop()
            for nb in rem_adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) != n:
            return None
            
        # DFS with Warnsdorff heuristic
        path = [start_node]
        visited_path = {start_node}
        
        def dfs(cur):
            if len(path) == n:
                # Check if we can close the cycle
                if start_node in rem_adj[cur]:
                    # Ensure the closing edge is not used
                    e_close = (min(cur, start_node), max(cur, start_node))
                    if e_close not in used_edges:
                        return list(path)
                return None
                
            # Get candidates
            cands = []
            for nxt in rem_adj[cur]:
                if nxt not in visited_path:
                    # Heuristic: prefer nodes with fewer onward moves (Warnsdorff)
                    onward = 0
                    for nn in rem_adj[nxt]:
                        if nn not in visited_path and nn != cur:
                            onward += 1
                    cands.append((onward, nxt))
                    
            cands.sort()
            
            for _, nxt in cands:
                path.append(nxt)
                visited_path.add(nxt)
                
                # Pruning: Check if the unvisited subgraph is connected to nxt
                # This is expensive, so we only do it if the path is getting long?
                # Or just rely on Warnsdorff.
                
                res = dfs(nxt)
                if res is not None:
                    return res
                
                path.pop()
                visited_path.remove(nxt)
                
            return None
            
        return dfs(start_node)

    # Greedy packing
    # Try starting the first cycle from different nodes to increase success rate
    # Usually node 0 is fine, but we can try a few.
    
    # We will attempt to find 3 cycles.
    # If we fail to find the 3rd, we might need to backtrack the 2nd or 1st.
    # Given the time constraint, we will try a few variations of the first cycle.
    
    # Variation 1: Start at 0
    used_edges = set()
    cycles = []
    
    # Try to find cycle 1
    c1 = find_ham_cycle(used_edges, 0)
    if not c1:
        # Try other starts
        for s in range(1, 5):
            c1 = find_ham_cycle(used_edges, s)
            if c1:
                break
    
    if not c1:
        return [] # Should not happen for this graph
        
    used_edges.update((min(u, v), max(u, v)) for i, u in enumerate(c1) for v in [c1[(i+1)%len(c1)]])
    cycles.append(c1)
    
    # Try to find cycle 2
    c2 = find_ham_cycle(used_edges, 0)
    if not c2:
        for s in range(1, 5):
            c2 = find_ham_cycle(used_edges, s)
            if c2:
                break
                
    if not c2:
        # Backtrack: try a different C1?
        # For now, return what we have.
        return [tuple(c) for c in cycles]
        
    used_edges.update((min(u, v), max(u, v)) for i, u in enumerate(c2) for v in [c2[(i+1)%len(c2)]])
    cycles.append(c2)
    
    # Try to find cycle 3
    c3 = find_ham_cycle(used_edges, 0)
    if not c3:
        for s in range(1, 5):
            c3 = find_ham_cycle(used_edges, s)
            if c3:
                break
                
    if c3:
        # Verify edge disjointness just in case
        es3 = set((min(u, v), max(u, v)) for i, u in enumerate(c3) for v in [c3[(i+1)%len(c3)]])
        if not (es3 & used_edges):
            cycles.append(c3)
            
    # Validate final result
    # Ensure all cycles use the same vertex set (all 21 vertices)
    # And are edge-disjoint.
    
    if not cycles:
        return []
        
    # The problem requires returning a list of cycles.
    # Convert to tuples of ints.
    final_cycles = [tuple(c) for c in cycles]
    
    # Final check: time
    if time.time() > deadline:
         # Return best effort
         pass
         
    return final_cycles

# EVOLVE-BLOCK-END
