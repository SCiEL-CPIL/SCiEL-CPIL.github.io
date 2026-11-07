"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    # 1. Identify Candidate Edges
    # To maintain triangle-freeness, we only add edges between opposite parity (bipartite).
    # This ensures the resulting graph remains bipartite and thus triangle-free.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u - v) % 2 == 1 and (u, v) not in base_edges:
                candidates.append((u, v))
                
    # Sort candidates by potential impact (descending distance)
    # Edges that skip more nodes are more likely to be part of the optimal solution.
    candidates.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)
    
    # Precompute base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)

    # Fast diameter check using BFS from all nodes
    # Optimized: Since graph is small (24 nodes), we can use bitsets or simple lists.
    # We'll use a standard BFS but optimized for speed.
    def get_diameter(adj):
        max_dist = 0
        # adj is a list of sets
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist

    # We search for the minimum number of edges k such that there exists a subset of size k 
    # that reduces diameter to <= 4.
    # We iterate k from 1 upwards.
    
    # To make the search efficient, we limit the candidate pool to the top M edges.
    # Why? The optimal solution for diameter reduction on a path almost certainly uses long chords.
    # For n=24, k is likely small (2-5).
    
    M = 60 # Shortlist size. C(60, 4) is ~487k, C(60, 5) is ~5.5M. 
           # 5.5M checks * ~500 ops/check = 2.75B ops. Too slow for pure Python in 50s?
           # Let's refine: We will use a recursive DFS with pruning.
           # Pruning: If current diameter is D, and we have added k edges, 
           # and the "best possible" reduction from remaining edges is R, 
           # if D - R > 4, prune.
           # However, calculating R is hard. 
           # Alternative: Just try small k first.
           
    # Let's try k=1, 2, 3, 4, 5...
    # For each k, we generate combinations from the top M candidates.
    # To speed up, we can pre-filter candidates based on "coverage".
    
    # A smarter approach: 
    # The diameter of a path augmented with chords is determined by the longest shortest path.
    # We can model this as: Can we connect the path such that every node is within distance 4 of every other?
    
    # Let's use a recursive backtracking search that builds the set of edges incrementally.
    # State: (current_edges_set, current_adj, current_diam, index_in_candidates)
    # We want to find the smallest set.
    
    # We will iterate k from 1 to 10.
    # For each k, we do a DFS.
    
    # Optimization: 
    # Instead of checking diameter from scratch every time (O(N^2)), 
    # we can update distances incrementally? Hard with BFS.
    # But N=24 is small. O(N^2) = 576 operations.
    # If we check 100,000 combinations, that's 57M operations. Feasible in ~5-10s in Python if optimized.
    # C(60, 4) is 487,635. 487k * 576 = 280M ops. ~20-30s. Risky.
    # Let's reduce M to 40. C(40, 4) = 91,390. 91k * 576 = 52M ops. ~3-5s. Safe.
    # C(40, 5) = 658,008. 658k * 576 = 379M ops. ~20-30s. Risky for k=5.
    # So we can exhaustively search k=1,2,3,4 on top 40 candidates.
    # If no solution found, we try k=5 on top 30 candidates. C(30,5)=142,506. Safe.
    
    def search_for_k(k, candidate_pool):
        # Generate combinations of size k from candidate_pool
        # Use itertools.combinations
        for combo in itertools.combinations(candidate_pool, k):
            if time.time() > deadline:
                return None
            
            # Build adjacency
            adj = [set(base_adj[i]) for i in range(n)]
            for u, v in combo:
                adj[u].add(v)
                adj[v].add(u)
            
            # Check diameter
            if get_diameter(adj) <= 4:
                return list(combo)
                
        return None

    # Try increasing k
    # Start with a larger pool for small k, shrink for larger k
    pools = [
        (1, 80),
        (2, 80),
        (3, 60),
        (4, 45),
        (5, 35),
        (6, 30)
    ]
    
    for k, m in pools:
        if time.time() > deadline - 5:
            break
        pool = candidates[:m]
        sol = search_for_k(k, pool)
        if sol:
            # Format output
            final_added = []
            seen = set()
            for u, v in sol:
                if u < v:
                    key = (u, v)
                else:
                    key = (v, u)
                if key not in seen:
                    final_added.append([int(key[0]), int(key[1])])
                    seen.add(key)
            return final_added

    # Fallback: If no small solution found, use a greedy heuristic to get *a* valid solution
    # This should not happen for n=24, but just in case.
    adj = [set(base_adj[i]) for i in range(n)]
    current_edges = []
    current_diam = get_diameter(adj)
    
    while current_diam > 4:
        if time.time() > deadline:
            break
        best_edge = None
        best_diam = current_diam + 1
        for u, v in candidates:
            if (u, v) in [(a,b) for a,b in current_edges] or (v, u) in [(a,b) for a,b in current_edges]:
                continue
            adj[u].add(v)
            adj[v].add(u)
            new_diam = get_diameter(adj)
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = (u, v)
                
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        current_edges.append((u, v))
        current_diam = best_diam
        
    if current_diam <= 4:
        final_added = []
        seen = set()
        for u, v in current_edges:
            if u < v:
                key = (u, v)
            else:
                key = (v, u)
            if key not in seen:
                final_added.append([int(key[0]), int(key[1])])
                seen.add(key)
        return final_added

    return []

# EVOLVE-BLOCK-END
