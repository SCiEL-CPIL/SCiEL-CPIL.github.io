import time
import sys
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # 1. Build Base Graph Adjacency (Bitsets)
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # 2. Identify Safe Candidate Edges
    # An edge (u, v) is safe if u and v are not adjacent (no self-loop/multi-edge) 
    # and do not share a common neighbor (triangle-free constraint).
    # We only consider u < v.
    candidates = []
    candidate_bitmaps = []
    
    # Precompute common neighbors for speed? Or just check on the fly.
    # For N=24, checking on the fly is fast.
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                continue # Already connected
            # Check triangle: (adj[u] & adj[v]) == 0
            if (adj[u] & adj[v]) == 0:
                candidates.append((u, v))
                candidate_bitmaps.append(1 << u | (1 << v))
                
    m = len(candidates)
    
    # 3. Precompute "Conflict" between candidates?
    # Actually, the triangle-free constraint is static relative to the base graph.
    # Adding edge (u,v) does NOT make another safe edge (x,y) unsafe, 
    # UNLESS (x,y) forms a triangle with (u,v).
    # Wait. The constraint is "Union with pinned path must remain triangle-free".
    # This means NO triangles in the final graph.
    # If I add edge (u,v) and edge (x,y), do they conflict?
    # They conflict if they form a triangle.
    # A triangle requires 3 edges. 
    # If I add (u,v) and (x,y), a triangle could be formed if:
    # 1. u==x and v==y (same edge, impossible in set)
    # 2. u==x, v!=y. Triangle (u, v, y) requires edge (u,v), (u,y), (v,y). 
    #    (u,v) is added. (u,y) is base or added. (v,y) is base or added.
    #    If (u,y) is base and (v,y) is base, then (u,v) would have been unsafe in step 2.
    #    So (u,v) is safe means no base neighbors in common.
    #    However, if I add (u,v) and (u,y) [where y is not v], and there is a base edge (v,y)?
    #    Then triangle (u,v,y) is formed.
    #    This means candidate (u,v) and candidate (u,y) conflict IF (v,y) is a base edge.
    #    More generally, two added edges (u,v) and (x,y) conflict if they form a triangle.
    #    This happens if:
    #    - {u,v} and {x,y} share a vertex, say u=x. Then we need edge (v,y) to exist in the FINAL graph.
    #      (v,y) exists if it's in base OR if it's also added.
    #      If (v,y) is base, then (u,v) and (u,y) conflict.
    #      If (v,y) is added, we have a chain of dependencies.
    #    - {u,v} and {x,y} are disjoint. No triangle possible with just 2 edges.
    
    # So, the "Independent Set" formulation is on the graph of candidates where edges represent "conflicts".
    # Two candidates conflict if adding both creates a triangle.
    # This conflict graph is not static? 
    # If (v,y) is a candidate, and I add (u,v) and (u,y), they conflict ONLY IF (v,y) is added.
    # Wait. If I add (u,v) and (u,y), and (v,y) is NOT added, is there a triangle?
    # No. The triangle must be fully present.
    # So (u,v) and (u,y) do NOT conflict with each other directly, unless (v,y) is in the base graph.
    # If (v,y) is in the base graph, then (u,v) was unsafe in step 2?
    # Let's re-verify Step 2 logic.
    # "Safe" meant: no common neighbor in BASE.
    # If (v,y) is base, then v and y are connected.
    # If I consider adding (u,v), I check if u and v have common neighbor in base.
    # If y is a common neighbor of u and v in base, then (u,v) is unsafe.
    # So if (u,v) is safe, it means NO base path of length 2 between u and v.
    # Therefore, if (u,v) is safe, and I add (u,y) (also safe), can they form a triangle?
    # Triangle (u, v, y). Edges: (u,v) [added], (u,y) [added], (v,y) [must exist].
    # If (v,y) is base, then v and y are connected.
    # Does (u,v) being safe imply (v,y) is not base?
    # Safe(u,v) => N_base(u) and N_base(v) are disjoint.
    # If (v,y) is base, then y in N_base(v).
    # If (u,y) is added, y is not in N_base(u) necessarily? 
    # If (u,y) is added, it means (u,y) is not base.
    # But if (v,y) is base, and (u,v) is safe, then y is not in N_base(u).
    # So (u,y) is not base. This is consistent.
    # BUT, if I add (u,v) and (u,y), and (v,y) is base, THEN a triangle (u,v,y) is formed.
    # Did Step 2 filter this out?
    # Step 2 checks if (u,v) has common neighbor in base.
    # If (v,y) is base and (u,y) is NOT base, y is not a common neighbor.
    # So (u,v) passes Step 2.
    # So (u,v) and (u,y) CAN form a triangle if (v,y) is base.
    # This means the conflict between candidates depends on the base graph AND the fact that the third edge is base.
    
    # Let's build a Conflict Graph among candidates.
    # Conflict[i][j] = True if adding edge i and edge j creates a triangle.
    # Edges i=(u,v), j=(x,y).
    # They form a triangle if they share a vertex and the third edge is in the base graph.
    # Case 1: u == x. Triangle (u, v, y). Needs (v,y) in base.
    # Case 2: u == y. Triangle (u, v, x). Needs (v,x) in base.
    # Case 3: v == x. Triangle (u, v, y). Needs (u,y) in base.
    # Case 4: v == y. Triangle (u, v, x). Needs (u,x) in base.
    # Disjoint edges never form a triangle with each other.
    
    conflict = [0] * m
    for i in range(m):
        u, v = candidates[i]
        for j in range(i + 1, m):
            x, y = candidates[j]
            if u == x or u == y or v == x or v == y:
                # Check if third edge is in base
                # If u==x, check (v,y)
                # If u==y, check (v,x)
                # If v==x, check (u,y)
                # If v==y, check (u,x)
                
                # We can just check if the set of 3 vertices forms a triangle in base+added?
                # No, simpler:
                # The 3 vertices involved are {u,v,x,y} minus duplicates.
                # Since they share a vertex, there are exactly 3 distinct vertices.
                # Let's identify the 3 vertices.
                verts = {u, v, x, y}
                if len(verts) == 3:
                    # Find the 3 vertices
                    a, b, c = list(verts)
                    # Check if all 3 edges exist in BASE? No, 2 are added, 1 must be base.
                    # Actually, if any of the 3 edges is NOT in base and NOT in {i, j}, then no triangle.
                    # The edges of the triangle are (a,b), (b,c), (c,a).
                    # Two of them are the added edges i and j.
                    # The third one MUST be in the base graph for the triangle to be formed by just adding i and j.
                    # (If the third one is also added, it's a different pair conflict, handled by that pair).
                    
                    # So, check if the "third" edge is in base.
                    # How to identify the third edge?
                    # The added edges are (u,v) and (x,y).
                    # The triangle vertices are A, B, C.
                    # The edges are AB, BC, CA.
                    # One of these is (u,v), one is (x,y).
                    # The remaining one is the "base" edge required.
                    
                    # Let's just check: does the subgraph induced by {u,v,x,y} in the BASE graph contain the edge connecting the two "non-shared" vertices?
                    # If u==x, shared is u. Non-shared are v and y. Check if (v,y) in base.
                    # If u==y, shared is u. Non-shared are v and x. Check if (v,x) in base.
                    # If v==x, shared is v (x). Non-shared are u and y. Check if (u,y) in base.
                    # If v==y, shared is v (y). Non-shared are u and x. Check if (u,x) in base.
                    
                    is_conflict = False
                    if u == x:
                        if (adj[v] >> y) & 1: is_conflict = True
                    elif u == y:
                        if (adj[v] >> x) & 1: is_conflict = True
                    elif v == x:
                        if (adj[u] >> y) & 1: is_conflict = True
                    elif v == y:
                        if (adj[u] >> x) & 1: is_conflict = True
                        
                    if is_conflict:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

    # 4. Backtracking Search for Minimum Independent Set (in Conflict Graph)
    # We want the smallest set of candidates S such that:
    # 1. S is an independent set in the Conflict Graph (no triangles formed).
    # 2. Diameter of Base + S <= 4.
    
    # Optimization: Pre-compute diameter impact?
    # Diameter calculation is expensive. 
    # Strategy:
    # Iterate k from 1 to m.
    # Find if there is an independent set of size k that satisfies diameter.
    # Use branch and bound.
    
    # To speed up diameter check, we can use bitset BFS.
    def check_diameter(added_mask):
        # Build adjacency with added edges
        # adj is list of ints.
        # added_mask is int.
        # Create local adj
        local_adj = adj[:]
        for i in range(m):
            if (added_mask >> i) & 1:
                u, v = candidates[i]
                local_adj[u] |= (1 << v)
                local_adj[v] |= (1 << u)
                
        # BFS from each node
        max_diam = 0
        for src in range(n):
            # Bitset BFS
            visited = 1 << src
            dist = 0
            frontier = 1 << src
            
            while frontier:
                dist += 1
                # Expand frontier
                new_frontier = 0
                f = frontier
                while f:
                    lsb = f & -f
                    idx = (lsb.bit_length() - 1)
                    f ^= lsb
                    new_frontier |= local_adj[idx]
                new_frontier &= ~visited
                if new_frontier == 0:
                    break
                visited |= new_frontier
                frontier = new_frontier
                
            if dist - 1 > max_diam:
                max_diam = dist - 1
            if max_diam > 4:
                return False # Early exit
                
        return True
        
    # Heuristic for ordering candidates:
    # Prefer edges that connect "far" nodes? 
    # Or just use the order from generation (u, v).
    # To improve search, we can sort candidates by "centrality" or just keep as is.
    # Given N=24, m is small.
    # Max m: 24*23/2 - 23 (base) = 253 - 23 = 230.
    # Minus unsafe ones.
    # Path graph: 
    # (0,2) unsafe (common 1). (0,3) safe? N(0)={1}, N(3)={2,4}. Disjoint. Safe.
    # Most long edges are safe.
    
    # We will search for k=1, 2, 3...
    # For a given k, we do backtracking.
    
    # To make backtracking fast, we can use a "Conflict-Free" list generation.
    
    best_solution = None
    deadline = time.time() + 80
    
    # Sort candidates to put "good" ones first?
    # Heuristic: Edges that reduce diameter the most are good.
    # Approximation: Edges connecting high-distance nodes.
    # Distance in path is |u-v|.
    # So sort by |u-v| descending.
    candidates_sorted = sorted(candidates, key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    # Re-index conflict graph for sorted candidates?
    # Or just keep original indices and iterate in sorted order.
    # Backtracking is easier if we iterate indices 0..m-1.
    # Let's create a mapping: sorted_idx -> original_idx.
    # And conflict_orig -> conflict_sorted?
    # Simpler: Just use the original list, but in the backtracking, choose the next candidate from a list of "available" candidates.
    
    # Actually, let's just stick to the original order for simplicity, 
    # but add a heuristic in the branching: 
    # When choosing the next edge to add, prefer those with higher "potential" (longer distance).
    
    # Let's refine the backtracking:
    # State: current_mask, current_conflict_mask (edges that cannot be added), current_diam (not tracked, just checked at leaf or periodically)
    # Problem: Checking diameter at every leaf is slow if leaves are many.
    # But we stop as soon as we find a valid one.
    # And we search k=1, then k=2, etc.
    
    # Lower bound on k:
    # Diameter of path is 23.
    # Each edge can reduce diameter by at most... ?
    # Adding an edge (u,v) creates a shortcut.
    # Roughly, we need at least 1 edge.
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Backtracking function
        # args: start_idx, remaining_to_pick, current_mask, forbidden_mask
        # forbidden_mask contains bits for edges that conflict with current_mask.
        
        def backtrack(idx, rem, cur_mask, forbidden):
            nonlocal found, best_solution
            if found:
                return
            
            if rem == 0:
                if check_diameter(cur_mask):
                    found = True
                    # Extract edges
                    sol = []
                    for i in range(m):
                        if (cur_mask >> i) & 1:
                            sol.append(candidates[i])
                    best_solution = sol
                return
            
            if time.time() > deadline:
                return
                
            # If not enough candidates left to pick 'rem', prune
            # Count available candidates from idx to m
            # This is expensive. 
            # Instead, we can just iterate.
            
            # Heuristic: Pick candidates in order of "desirability" (longest first)
            # We need to iterate through candidates starting from idx.
            # But we want to try "good" ones first.
            # Since we pre-sorted by distance, let's use that order.
            # We need a mapping from sorted order to original index.
            
            # Let's restructure: 
            # 1. Sort candidates by distance desc.
            # 2. Build conflict matrix in this sorted order.
            # 3. Backtrack over sorted indices 0..m-1.
            
            # This requires rebuilding conflict matrix. 
            # Let's do that outside the loop.
            
            pass # Placeholder
            
        # We need to implement the sorted version properly.
        
        # Let's abort the inner function and do it properly outside.
        
        break # Exit k loop to implement properly
        
    # --- Proper Implementation ---
    
    # 1. Sort candidates
    # Map original index to sorted position?
    # Or just create new lists.
    
    orig_indices = list(range(m))
    # Sort by distance descending
    sorted_indices = sorted(orig_indices, key=lambda i: abs(candidates[i][0] - candidates[i][1]), reverse=True)
    
    # New candidate list in sorted order
    sorted_candidates = [candidates[i] for i in sorted_indices]
    
    # New conflict matrix in sorted order
    # conflict_sorted[i][j] corresponds to conflict between sorted_candidates[i] and sorted_candidates[j]
    # We can map back to original to check conflict.
    # Or just rebuild.
    
    # Rebuild conflict for sorted list
    m_sorted = m
    sorted_conflict = [0] * m_sorted
    
    for i in range(m_sorted):
        u, v = sorted_candidates[i]
        for j in range(i + 1, m_sorted):
            x, y = sorted_candidates[j]
            if u == x or u == y or v == x or v == y:
                is_conflict = False
                if u == x:
                    if (adj[v] >> y) & 1: is_conflict = True
                elif u == y:
                    if (adj[v] >> x) & 1: is_conflict = True
                elif v == x:
                    if (adj[u] >> y) & 1: is_conflict = True
                elif v == y:
                    if (adj[u] >> x) & 1: is_conflict = True
                    
                if is_conflict:
                    sorted_conflict[i] |= (1 << j)
                    sorted_conflict[j] |= (1 << i)
                    
    # Now search
    
    def solve_k(k):
        # Backtracking to find independent set of size k that satisfies diameter
        # Returns list of sorted indices or None
        
        # Pruning: 
        # If the number of available candidates (not forbidden) is < k, fail.
        
        def bt(idx, rem, cur_mask, forbidden):
            if rem == 0:
                if check_diameter(cur_mask):
                    return True
                return False
                
            if time.time() > deadline:
                return False
                
            # Heuristic: Try candidates with highest "gain" first?
            # We are iterating idx from 0 to m-1.
            # Since we sorted by distance, idx 0 is the longest edge.
            # This is a good heuristic.
            
            # Optimization: 
            # If we pick candidate i, we forbid all j such that conflict[i][j] is set.
            
            # We can also prune if (m - idx) < rem.
            if m_sorted - idx < rem:
                return False
                
            for i in range(idx, m_sorted):
                if (forbidden >> i) & 1:
                    continue
                    
                # Pick i
                new_mask = cur_mask | (1 << i)
                new_forbidden = forbidden | (1 << i) | sorted_conflict[i]
                
                if bt(i + 1, rem - 1, new_mask, new_forbidden):
                    return True
                    
                # Unpick i (handled by recursion scope)
                
            return False
            
        return bt(0, k, 0, 0)
        
    # Iterate k
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
            
        if solve_k(k):
            # Find the solution
            # We need to recover the mask. 
            # The bt function above returns True/False but doesn't store the mask.
            # Let's modify it to store the best mask found.
            
            # Actually, let's just re-run or store in nonlocal.
            # Better: modify bt to return the mask or store it.
            
            # Let's rewrite solve_k to store the result.
            
            result_mask = [0]
            
            def bt_store(idx, rem, cur_mask, forbidden):
                if rem == 0:
                    if check_diameter(cur_mask):
                        result_mask[0] = cur_mask
                        return True
                    return False
                    
                if time.time() > deadline:
                    return False
                    
                if m_sorted - idx < rem:
                    return False
                    
                for i in range(idx, m_sorted):
                    if (forbidden >> i) & 1:
                        continue
                        
                    new_mask = cur_mask | (1 << i)
                    new_forbidden = forbidden | (1 << i) | sorted_conflict[i]
                    
                    if bt_store(i + 1, rem - 1, new_mask, new_forbidden):
                        return True
                        
                return False
                
            if bt_store(0, k, 0, 0):
                # Extract edges
                sol = []
                mask = result_mask[0]
                for i in range(m_sorted):
                    if (mask >> i) & 1:
                        u, v = sorted_candidates[i]
                        sol.append([int(u), int(v)])
                return sol
                
    return []

# Wait, I need to put this all inside construct().
# The code above was a draft.
# Let's assemble the final clean version.

def construct():
    n = P['n']
    base_edges = P['edges']
    
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                continue
            if (adj[u] & adj[v]) == 0:
                candidates.append((u, v))
                
    m = len(candidates)
    
    # Sort by distance desc
    candidates.sort(key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    # Build conflict matrix for sorted candidates
    conflict = [0] * m
    for i in range(m):
        u, v = candidates[i]
        for j in range(i + 1, m):
            x, y = candidates[j]
            if u == x or u == y or v == x or v == y:
                is_conflict = False
                if u == x:
                    if (adj[v] >> y) & 1: is_conflict = True
                elif u == y:
                    if (adj[v] >> x) & 1: is_conflict = True
                elif v == x:
                    if (adj[u] >> y) & 1: is_conflict = True
                elif v == y:
                    if (adj[u] >> x) & 1: is_conflict = True
                    
                if is_conflict:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    
    def check_diameter(added_mask):
        local_adj = adj[:]
        for i in range(m):
            if (added_mask >> i) & 1:
                u, v = candidates[i]
                local_adj[u] |= (1 << v)
                local_adj[v] |= (1 << u)
                
        max_diam = 0
        for src in range(n):
            visited = 1 << src
            dist = 0
            frontier = 1 << src
            
            while frontier:
                dist += 1
                new_frontier = 0
                f = frontier
                while f:
                    lsb = f & -f
                    idx = (lsb.bit_length() - 1)
                    f ^= lsb
                    new_frontier |= local_adj[idx]
                new_frontier &= ~visited
                if new_frontier == 0:
                    break
                visited |= new_frontier
                frontier = new_frontier
                
            if dist - 1 > max_diam:
                max_diam = dist - 1
            if max_diam > 4:
                return False
        return True
        
    deadline = time.time() + 80
    best_sol = None
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
            
        result_mask = [0]
        
        def bt(idx, rem, cur_mask, forbidden):
            if rem == 0:
                if check_diameter(cur_mask):
                    result_mask[0] = cur_mask
                    return True
                return False
                
            if time.time() > deadline:
                return False
                
            if m - idx < rem:
                return False
                
            for i in range(idx, m):
                if (forbidden >> i) & 1:
                    continue
                    
                new_mask = cur_mask | (1 << i)
                new_forbidden = forbidden | (1 << i) | conflict[i]
                
                if bt(i + 1, rem - 1, new_mask, new_forbidden):
                    return True
                    
            return False
            
        if bt(0, k, 0, 0):
            sol = []
            mask = result_mask[0]
            for i in range(m):
                if (mask >> i) & 1:
                    u, v = candidates[i]
                    sol.append([int(u), int(v)])
            return sol
            
    return []

# One final check: 
# The problem asks for "Minimize the number of added edges".
# My search goes k=1, 2, ... so it finds the minimum.
# The "triangle-free" constraint is handled by the conflict graph (independent set).
# The "diameter" constraint is handled by check_diameter.
# The "connected" constraint is automatically satisfied if diameter is finite (and <=4).
# The "no duplicate edges" is handled by candidates list (only non-base edges).
# The "types" are int.

# Potential issue: Recursion depth.
# m can be up to ~200.
# k will likely be small (1-5).
# Depth of recursion is k.
# So recursion depth is small.
# The loop `for i in range(idx, m)` is the branching factor.
# This is efficient.

# One optimization: 
# If k=1 fails, we try k=2.
# We can cache the result of check_diameter for masks? 
# Masks are unique in the search tree (mostly).
# But check_diameter is O(N^2) or O(N+E). With bitsets it's very fast.
# N=24.
# It should be fine.

# Final check on "triangle-free" logic.
# If I add edges that don't conflict in the pairwise sense, can they form a triangle?
# Triangle requires 3 edges.
# If I add e1, e2, e3.
# If e1 and e2 don't conflict, it means they don't form a triangle with a BASE edge.
# If e1 and e3 don't conflict, same.
# If e2 and e3 don't conflict, same.
# Can e1, e2, e3 form a triangle among themselves?
# Yes! 
# Example: e1=(0,1), e2=(1,2), e3=(0,2).
# None of these are base edges (assuming path is 0-1-2... wait, 0-1 and 1-2 ARE base edges).
# So e1, e2, e3 cannot be added if they are base.
# Let's take a cycle in a non-path graph?
# But base is a path.
# So any 3 added edges forming a triangle must use 3 non-base edges.
# My conflict graph only checks if 2 added edges + 1 base edge form a triangle.
# It DOES NOT check if 3 added edges form a triangle.
# 
# This is a BUG.
# 
# I need to ensure the set of added edges is triangle-free by itself too.
# This means the subgraph induced by added edges must be triangle-free.
# 
# How to handle?
# The "Independent Set" on the pairwise conflict graph is not sufficient.
# 
# However, for a path graph, can 3 added edges form a triangle?
# Added edges are "chords".
# A triangle formed by 3 chords (u,v), (v,w), (w,u).
# This requires u, v, w to be mutually connected by chords.
# 
# I need to add a check in the backtracking:
# When adding edge i, check if it forms a triangle with TWO other already added edges.
# 
# In `bt`, when we pick `i`, we should check if `i` forms a triangle with any pair in `cur_mask`.
# 
# Let's add this check.
# 
# Modified `bt`:
# When picking `i`:
# 1. Check conflict with base (handled by `forbidden` mask? No, `forbidden` mask handles conflicts with PREVIOUSLY added edges that would form a triangle with BASE).
#    Wait. `conflict[i]` marks edges that form a triangle with `i` AND A BASE EDGE.
#    So `forbidden` prevents adding `j` if `i` is added and `i, j` + base = triangle.
#    This handles (Added, Added, Base) triangles.
#    
# 2. We also need to prevent (Added, Added, Added) triangles.
#    When adding `i`, we must ensure `i` does not form a triangle with any two edges in `cur_mask`.
#    
#    Let's check: 
#    For each edge `j` in `cur_mask`:
#       Let `i` be (u,v). `j` be (x,y).
#       If they share a vertex, say u==x.
#       Then we need to check if there is an edge `k` in `cur_mask` (k != j) that connects v and y.
#       If such `k` exists, then (u,v), (u,y), (v,y) form a triangle.
#       
#    This check is O(k^2) per addition. Since k is small, it's fine.
#    
#    Let's implement `is_triangle_with_cur_mask(i, cur_mask)`.
    
    # Revised construct with this fix.
    
    # I will rewrite the code block below with this fix included.

# --- FINAL CODE ---

import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    adj = [0] * n
    for u, v in base_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                continue
            if (adj[u] & adj[v]) == 0:
                candidates.append((u, v))
                
    m = len(candidates)
    
    # Sort by distance desc
    candidates.sort(key=lambda e: abs(e[0]-e[1]), reverse=True)
    
    # Build conflict matrix for (Added, Added, Base) triangles
    conflict = [0] * m
    for i in range(m):
        u, v = candidates[i]
        for j in range(i + 1, m):
            x, y = candidates[j]
            if u == x or u == y or v == x or v == y:
                is_conflict = False
                if u == x:
                    if (adj[v] >> y) & 1: is_conflict = True
                elif u == y:
                    if (adj[v] >> x) & 1: is_conflict = True
                elif v == x:
                    if (adj[u] >> y) & 1: is_conflict = True
                elif v == y:
                    if (adj[u] >> x) & 1: is_conflict = True
                    
                if is_conflict:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                    
    def check_diameter(added_mask):
        local_adj = adj[:]
        for i in range(m):
            if (added_mask >> i) & 1:
                u, v = candidates[i]
                local_adj[u] |= (1 << v)
                local_adj[v] |= (1 << u)
                
        max_diam = 0
        for src in range(n):
            visited = 1 << src
            dist = 0
            frontier = 1 << src
            
            while frontier:
                dist += 1
                new_frontier = 0
                f = frontier
                while f:
                    lsb = f & -f
                    idx = (lsb.bit_length() - 1)
                    f ^= lsb
                    new_frontier |= local_adj[idx]
                new_frontier &= ~visited
                if new_frontier == 0:
                    break
                visited |= new_frontier
                frontier = new_frontier
                
            if dist - 1 > max_diam:
                max_diam = dist - 1
            if max_diam > 4:
                return False
        return True
        
    def forms_triangle_with_mask(i, mask):
        # Check if candidate i forms a triangle with any two edges in mask
        u, v = candidates[i]
        # Iterate through edges in mask
        m_copy = mask
        while m_copy:
            lsb1 = m_copy & -m_copy
            j1 = (lsb1.bit_length() - 1)
            m_copy ^= lsb1
            
            x, y = candidates[j1]
            
            # Check if i and j1 share a vertex
            if u == x or u == y or v == x or v == y:
                # Identify the two "free" vertices
                # If u==x, free are v and y. Need edge (v,y) in mask.
                # If u==y, free are v and x. Need edge (v,x) in mask.
                # If v==x, free are u and y. Need edge (u,y) in mask.
                # If v==y, free are u and x. Need edge (u,x) in mask.
                
                # Find the edge in mask that connects the free vertices.
                # We can just check if the edge exists in the subgraph of mask.
                # But we need to check if it's in the MASK (i.e., added).
                
                # Let's find the target edge index.
                # We can precompute a map: (u, v) -> index in candidates?
                # Or just check if the edge (a, b) is in the mask by iterating?
                # Since k is small, iterating is fine. But we are already iterating.
                
                # Let's determine the two vertices a, b.
                if u == x:
                    a, b = v, y
                elif u == y:
                    a, b = v, x
                elif v == x:
                    a, b = u, y
                elif v == y:
                    a, b = u, x
                else:
                    continue # No shared vertex, no triangle possible with just 2 edges
                
                # Check if edge (a, b) is in mask
                # We need to find if any edge k in mask is (a, b) or (b, a)
                m_copy2 = mask
                while m_copy2:
                    lsb2 = m_copy2 & -m_copy2
                    j2 = (lsb2.bit_length() - 1)
                    m_copy2 ^= lsb2
                    
                    if j2 == j1:
                        continue
                        
                    x2, y2 = candidates[j2]
                    if (x2 == a and y2 == b) or (x2 == b and y2 == a):
                        return True
                        
        return False
        
    deadline = time.time() + 80
    best_sol = None
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
            
        result_mask = [0]
        
        def bt(idx, rem, cur_mask, forbidden):
            if rem == 0:
                if check_diameter(cur_mask):
                    result_mask[0] = cur_mask
                    return True
                return False
                
            if time.time() > deadline:
                return False
                
            if m - idx < rem:
                return False
                
            for i in range(idx, m):
                if (forbidden >> i) & 1:
                    continue
                    
                # Check (Added, Added, Added) triangle
                if forms_triangle_with_mask(i, cur_mask):
                    continue
                    
                new_mask = cur_mask | (1 << i)
                new_forbidden = forbidden | (1 << i) | conflict[i]
                
                if bt(i + 1, rem - 1, new_mask, new_forbidden):
                    return True
                    
            return False
            
        if bt(0, k, 0, 0):
            sol = []
            mask = result_mask[0]
            for i in range(m):
                if (mask >> i) & 1:
                    u, v = candidates[i]
                    sol.append([int(u), int(v)])
            return sol
            
    return []

