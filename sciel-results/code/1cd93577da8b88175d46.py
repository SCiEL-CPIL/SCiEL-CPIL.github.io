"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _bfs(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # 1. Identify candidate edges
    # An edge (u, v) is a candidate if:
    # a) It is not in base_edges
    # b) It does not create a triangle in the base graph.
    #    Since base graph is a path, a triangle is formed if there is a common neighbor.
    #    In a path, common neighbors only exist if dist(u,v) == 2.
    #    So candidates are pairs (u, v) such that |u - v| > 2.
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges and v - u > 2:
                candidates.append((u, v))
                
    # 2. Identify the universe of "violations"
    # Pairs (i, j) in the base graph with distance > 4.
    # In a path 0-1-...-23, dist(i, j) = |i - j|.
    # So violations are pairs with |i - j| > 4.
    universe = []
    for i in range(n):
        for j in range(i + 5, n):
            universe.append((i, j))
            
    # 3. Determine coverage sets
    # For each candidate edge e, which violations does it cover?
    # A violation (i, j) is covered by edge (u, v) if adding (u, v) makes dist(i, j) <= 4.
    # This is equivalent to: is there a path from i to j of length <= 4 in G + {(u, v)}?
    # We can precompute this for all candidates.
    
    # Optimization: 
    # dist_G(i, j) is |i-j|.
    # If we add (u, v), the new distance is min(|i-j|, dist(i, u) + 1 + dist(v, j), dist(i, v) + 1 + dist(u, j)).
    # Note: dist(i, u) in the path is |i-u|.
    # So a violation (i, j) is covered by (u, v) if:
    # min(|i-j|, |i-u| + 1 + |v-j|, |i-v| + 1 + |u-j|) <= 4
    
    coverage = []
    for idx, (u, v) in enumerate(candidates):
        covered = set()
        for (i, j) in universe:
            # Since i < j and |i-j| > 4, the first term |i-j| is > 4.
            # We only need to check if the new path via (u, v) is <= 4.
            d1 = abs(i - u) + 1 + abs(v - j)
            d2 = abs(i - v) + 1 + abs(u - j)
            if d1 <= 4 or d2 <= 4:
                covered.add((i, j))
        coverage.append(covered)
        
    # 4. Branch and Bound Set Cover
    # We want to find the minimum number of edges to cover all violations.
    # Also, we must ensure the resulting graph is triangle-free.
    # Triangle-free constraint:
    # The base graph is a path. Adding an edge (u, v) creates a triangle if and only if there is a common neighbor.
    # In the final graph, a triangle can be formed by:
    # a) Two added edges and one base edge?
    #    Let added edges be (a,b) and (c,d).
    #    If they share a vertex, say b=c, then (a,b) and (b,d) are added.
    #    Is there an edge (a,d) in the base graph? Only if |a-d| == 1.
    #    If (a,d) is in base, then (a,b), (b,d), (a,d) form a triangle.
    #    So we cannot have two added edges (a,b) and (b,d) such that |a-d|==1.
    #    Actually, more generally, if we add (a,b) and (c,d), they form a triangle with a base edge (x,y) if {a,b,c,d} contains x,y and the other two are connected to them.
    #    Wait, simpler: 
    #    Triangle vertices x, y, z.
    #    Edges (x,y), (y,z), (z,x).
    #    At most one of these can be a base edge (since base is a path, it has no triangles, and any two base edges sharing a vertex are adjacent in the path, i.e., distance 2 apart, so the third edge would be a "jump" of 2, which is forbidden as a candidate).
    #    So a triangle must consist of either:
    #    - 1 base edge and 2 added edges.
    #    - 3 added edges.
    
    # Let's check compatibility of adding a set of edges.
    # Two added edges (u1, v1) and (u2, v2) are incompatible if they form a triangle with a base edge or each other.
    # Case 1: Triangle with base edge.
    # Base edge is (k, k+1).
    # Added edges must connect to k and k+1.
    # So we have added edges (a, k) and (b, k+1) or (a, k+1) and (b, k).
    # For (a, k), (b, k+1) and (k, k+1) to form a triangle, we need (a, b) to be an edge.
    # (a, b) is either base or added.
    # If (a, b) is base, then |a-b|=1.
    # If (a, b) is added, then we have a triangle of 3 added edges? No, (a,b) is the third edge.
    # So if we add (a, k) and (b, k+1), and |a-b|=1 (base edge), then triangle (a, k, k+1) is NOT formed unless (a, k+1) or (b, k) is an edge.
    # Wait. The edges are (a,k), (b,k+1), (k,k+1).
    # The vertices are a, k, k+1, b.
    # For a triangle, we need 3 vertices.
    # If a=b, then we have (a,k) and (a,k+1) and (k,k+1). This is a triangle (a, k, k+1).
    # So adding (a, k) and (a, k+1) is forbidden.
    # This means: we cannot add two edges that share a vertex 'a' and are both connected to adjacent vertices 'k' and 'k+1'.
    # i.e., if we add (u, v), we cannot add (u, v+1) or (u, v-1) or (u+1, v) or (u-1, v).
    # Basically, if (u, v) is added, no other added edge can be (u, v') where |v-v'|=1, or (u', v) where |u-u'|=1.
    
    # Case 2: Triangle of 3 added edges.
    # (a,b), (b,c), (c,a).
    # This is a clique K3 in the subgraph of added edges.
    # Since we are adding edges to a path, can we form a K3 with added edges?
    # Yes. e.g. add (0,2), (2,4), (0,4).
    # (0,2) and (2,4) share 2. (0,4) connects 0 and 4.
    # Are these valid candidates?
    # (0,2): dist 2. Forbidden (creates triangle 0-1-2 with base).
    # So candidates have dist >= 3.
    # If we have (a,b) with b-a >= 3, (b,c) with c-b >= 3, and (a,c).
    # Then c-a >= 6.
    # This forms a triangle.
    # So we must ensure no 3 added edges form a triangle.
    
    # Let's precompute incompatibility between candidates.
    # incompat[i][j] is True if candidate i and candidate j cannot both be in the solution.
    incompat = [[False] * len(candidates) for _ in range(len(candidates))]
    for i in range(len(candidates)):
        u1, v1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            u2, v2 = candidates[j]
            # Check if (u1,v1) and (u2,v2) form a triangle with a base edge.
            # This happens if they share a vertex, say x, and the other endpoints y, z are adjacent in base (|y-z|=1).
            # Or if they don't share a vertex, but their endpoints form a triangle with a base edge?
            # No, a triangle has 3 edges. If the base edge is (y,z), the other two edges must be (x,y) and (x,z).
            # So the added edges must share vertex x.
            # So:
            if u1 == u2:
                if abs(v1 - v2) == 1:
                    incompat[i][j] = True
            elif u1 == v2:
                if abs(v1 - u2) == 1:
                    incompat[i][j] = True
            elif v1 == u2:
                if abs(u1 - v2) == 1:
                    incompat[i][j] = True
            elif v1 == v2:
                if abs(u1 - u2) == 1:
                    incompat[i][j] = True
            
            # Check if they form a triangle with each other and a base edge?
            # No, that's covered above.
            
            # Check if they form a triangle of 3 added edges?
            # This requires a third edge. We only check pairwise incompatibility for pruning.
            # If we pick i, j, and later k, we check if {i,j,k} is a triangle.
            # To make B&B efficient, we can just check if the current set + new edge is triangle-free.
            
    # We will use a simple B&B that maintains the set of chosen edges.
    # State: (index in candidates, current_set, current_coverage, current_cost)
    # To speed up, we sort candidates by coverage size descending.
    # And we use a greedy upper bound.
    
    # Sort candidates
    cand_indices = list(range(len(candidates)))
    cand_indices.sort(key=lambda i: len(coverage[i]), reverse=True)
    
    # Map original index to new index? Or just use the sorted list.
    sorted_cands = [candidates[i] for i in cand_indices]
    sorted_cov = [coverage[i] for i in cand_indices]
    
    # Re-compute incompatibility for sorted candidates?
    # Or just keep a map.
    # Let's keep the original indices in the sorted list for easier incompat check.
    # sorted_idx[i] is the original index of the i-th sorted candidate.
    
    best_cost = float('inf')
    best_set = []
    
    # Greedy initialization for upper bound
    def greedy_cover():
        remaining = set(universe)
        chosen = []
        avail = list(range(len(sorted_cov)))
        while remaining:
            if not avail:
                return None
            # Find candidate with max coverage of remaining
            best_i = -1
            best_cov_count = -1
            for i in avail:
                cov_count = len(remaining & sorted_cov[i])
                if cov_count > best_cov_count:
                    best_cov_count = cov_count
                    best_i = i
            if best_i == -1:
                return None
            chosen.append(sorted_idx[best_i])
            remaining -= sorted_cov[best_i]
            # Remove incompatible candidates?
            # For a simple greedy, we might just pick the next best regardless, 
            # but we must ensure triangle-free.
            # Let's refine: only pick from avail if compatible with all chosen.
            # This makes greedy slower but safer.
            new_avail = []
            for j in avail:
                if j == best_i: continue
                compatible = True
                for k in chosen:
                    if incompat[k][sorted_idx[j]]:
                        compatible = False
                        break
                if compatible:
                    new_avail.append(j)
            avail = new_avail
        return chosen
        
    init_set = greedy_cover()
    if init_set:
        best_cost = len(init_set)
        best_set = init_set[:]
        
    # B&B
    # We search for a set of size < best_cost.
    # We can prune if current_depth + lower_bound >= best_cost.
    # Lower bound: ceil(remaining_violations / max_coverage_per_edge)
    
    def dfs(depth, current_set, current_cov, remaining, idx_start):
        nonlocal best_cost, best_set
        
        if time.time() > deadline:
            return
        
        if not remaining:
            if depth < best_cost:
                best_cost = depth
                best_set = current_set[:]
            return
            
        # Pruning
        if depth >= best_cost:
            return
            
        # Lower bound
        # Max coverage of any single edge for remaining
        max_cov = 0
        for i in range(idx_start, len(sorted_cov)):
            # Check compatibility
            compatible = True
            for k in current_set:
                if incompat[k][sorted_idx[i]]:
                    compatible = False
                    break
            if not compatible:
                continue
            c = len(remaining & sorted_cov[i])
            if c > max_cov:
                max_cov = c
            if max_cov == 0:
                break # No compatible edge covers anything?
                
        if max_cov == 0:
            return # Dead end
            
        lb = (len(remaining) + max_cov - 1) // max_cov
        if depth + lb >= best_cost:
            return
            
        # Branch
        # We can iterate through candidates from idx_start to end.
        # To avoid duplicates, we usually pick the first candidate that covers a specific element.
        # Let's pick the first element in 'remaining' (arbitrary order, e.g. sorted).
        # And iterate through candidates that cover it.
        
        target = min(remaining) # Pick a violation to cover
        
        for i in range(idx_start, len(sorted_cov)):
            if target not in sorted_cov[i]:
                continue
                
            # Check compatibility
            compatible = True
            for k in current_set:
                if incompat[k][sorted_idx[i]]:
                    compatible = False
                    break
            if not compatible:
                continue
                
            # Include i
            new_remaining = remaining - sorted_cov[i]
            new_set = current_set + [sorted_idx[i]]
            dfs(depth + 1, new_set, new_remaining, new_remaining, i + 1)
            
            # Exclude i?
            # Standard set cover B&B:
            # We branch on "include i" or "exclude i".
            # But if we exclude i, we must still cover 'target'.
            # So we can just continue the loop to try other candidates for 'target'.
            # This is equivalent to excluding i for the purpose of covering 'target' right now.
            # However, we might need i later for other elements?
            # No, if we don't pick i now to cover 'target', we pick some other j to cover 'target'.
            # Then i might be picked later for other elements.
            # So the loop structure "for i covering target: include i" handles the "pick one to cover target" logic.
            # This is correct for set cover.
            
            if time.time() > deadline:
                return

    # Start DFS
    dfs(0, [], set(), set(universe), 0)
    
    if best_set:
        # Convert to required format
        result = [[int(candidates[u][0]), int(candidates[u][1])] for u in best_set]
        return result
    else:
        # Fallback: should not happen if greedy works, but just in case
        return []

# EVOLVE-BLOCK-END
