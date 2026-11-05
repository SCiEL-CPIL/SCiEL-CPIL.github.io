import numpy as np

N = 16

def construct():
    # Map (u, v) to index
    # 0 <= u < v < N
    # Index = u * (N - 1 - u) + (v - u - 1) ? 
    # Let's just use a dictionary or a 2D array for speed.
    # 2D array edge_color[u][v] = color, -1 if uncolored.
    # We also need to map index to (u, v) for the return value.
    
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edges.append((u, v))
            
    num_edges = len(edges)
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i # Symmetric access
        
    colors = [-1] * num_edges
    next_color = 0
    
    # Precompute for each edge, the list of other 3 edges for each of the 3 cycles
    # This might be slow to precompute for all, but N=16 is small.
    # Let's compute on the fly or precompute a list of "conflict sets".
    
    # For each edge index i (u, v), find all pairs (x, y) such that x, y not in {u, v}
    # For each pair, there are 3 cycles.
    # We need the indices of the other 3 edges.
    
    # Let's build a list for each edge i: 
    # cycles_for_edge[i] = list of tuples (idx1, idx2, idx3)
    # where (idx1, idx2, idx3) are the other 3 edges in a 4-cycle containing edge i.
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    
    for i, (u, v) in enumerate(edges):
        # Find all other vertices
        others = [k for k in range(N) if k != u and k != v]
        # For each pair of others
        for x in others:
            for y in others:
                if x >= y: continue
                # Vertices are u, v, x, y
                # Cycle 1: u-v-x-y-u -> Edges: (u,v), (v,x), (x,y), (y,u)
                # Other edges: (v,x), (x,y), (y,u)
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                # Cycle 2: u-v-y-x-u -> Edges: (u,v), (v,y), (y,x), (x,u)
                # Other edges: (v,y), (y,x), (x,u)
                e1 = edge_to_idx[(v, y)]
                e2 = edge_to_idx[(y, x)]
                e3 = edge_to_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                # Cycle 3: u-x-v-y-u -> Edges: (u,x), (x,v), (v,y), (y,u)
                # Other edges: (u,x), (x,v), (v,y), (y,u)
                e1 = edge_to_idx[(u, x)]
                e2 = edge_to_idx[(x, v)]
                e3 = edge_to_idx[(v, y)]
                # Wait, the 4th edge is (y,u) which is (u,y).
                # Is (u,y) already in the set? 
                # The cycle is u-x-v-y-u. Edges: ux, xv, vy, yu.
                # The edge being colored is (u,v)? No, (u,v) is NOT in this cycle.
                # My previous assumption was that edge i=(u,v) is in the cycle.
                # Let's re-verify the cycles containing (u,v).
                # Vertices {u, v, x, y}.
                # Edges incident to u: ux, uy, uv
                # Edges incident to v: vx, vy, uv
                # Edge between x,y: xy
                # 
                # Cycles containing uv:
                # 1. u-v-x-y-u : uv, vx, xy, yu. (Contains uv)
                # 2. u-v-y-x-u : uv, vy, yx, xu. (Contains uv)
                # 3. u-x-v-y-u : ux, xv, vy, yu. (Does NOT contain uv)
                # 4. u-y-v-x-u : uy, yv, vx, xu. (Does NOT contain uv)
                # 
                # Wait, a 4-cycle on 4 vertices has 3 distinct cycle structures?
                # No, for a set of 4 vertices, there are 3 perfect matchings of the complement?
                # Let's list all 4-cycles on {u,v,x,y}:
                # 1. u-v-x-y-u (Edges: uv, vx, xy, yu)
                # 2. u-v-y-x-u (Edges: uv, vy, yx, xu)
                # 3. u-x-v-y-u (Edges: ux, xv, vy, yu)
                # 4. u-x-y-v-u (Edges: ux, xy, yv, vu) -> Same as 2 reversed? u-x-y-v-u uses ux, xy, yv, vu.
                # 5. u-y-x-v-u (Edges: uy, yx, xv, vu)
                # 6. u-y-v-x-u (Edges: uy, yv, vx, xu)
                #
                # Distinct sets of edges:
                # S1: {uv, vx, xy, yu}
                # S2: {uv, vy, yx, xu}
                # S3: {ux, xv, vy, yu}
                # S4: {ux, xy, yv, vu} -> {ux, xy, vy, uv}
                # S5: {uy, yx, xv, vu} -> {uy, xy, vx, uv}
                # S6: {uy, yv, vx, xu}
                #
                # Which of these contain uv?
                # S1: Yes
                # S2: Yes
                # S3: No
                # S4: Yes (vu is uv)
                # S5: Yes (vu is uv)
                # S6: No
                #
                # So there are 4 cycles containing edge uv?
                # Let's check the number of 4-cycles containing a specific edge in K4.
                # In K4, there are 3 distinct 4-cycles (up to rotation/reflection)?
                # No, there are 3 distinct Hamiltonian cycles in K4?
                # Number of Hamiltonian cycles in K4 is (4-1)! / 2 = 3.
                # They are:
                # 1. 0-1-2-3-0 (Edges: 01, 12, 23, 30)
                # 2. 0-1-3-2-0 (Edges: 01, 13, 32, 20)
                # 3. 0-2-1-3-0 (Edges: 02, 21, 13, 30)
                #
                # Edge 01 is in Cycle 1 and Cycle 2.
                # Edge 01 is NOT in Cycle 3.
                #
                # So for any edge, there are exactly 2 cycles containing it in a K4?
                # Let's re-verify.
                # Vertices {0,1,2,3}. Edge (0,1).
                # Cycle 1: 0-1-2-3-0. Contains 01.
                # Cycle 2: 0-1-3-2-0. Contains 01.
                # Cycle 3: 0-2-1-3-0. Contains 21 (which is 12), not 01.
                #
                # So only 2 cycles per edge per vertex pair?
                # But we have 14 other vertices.
                # For each pair (x,y), there are 2 cycles containing (u,v).
                # Total cycles to check per edge: 91 pairs * 2 cycles/pair = 182 cycles.
                
                # Let's correct the precomputation.
                pass
                
    # Re-do precomputation correctly
    cycles_for_edge = [[] for _ in range(num_edges)]
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for idx_x in range(len(others)):
            x = others[idx_x]
            for idx_y in range(idx_x + 1, len(others)):
                y = others[idx_y]
                # Cycle 1: u-v-x-y-u
                # Edges: (u,v), (v,x), (x,y), (y,u)
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                # Cycle 2: u-v-y-x-u
                # Edges: (u,v), (v,y), (y,x), (x,u)
                e1 = edge_to_idx[(v, y)]
                e2 = edge_to_idx[(y, x)]
                e3 = edge_to_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
    # Greedy Coloring
    for i in range(num_edges):
        u, v = edges[i]
        
        # Determine if a new color is allowed
        # A new color is allowed if for ALL cycles containing edge i,
        # the other 3 edges do NOT have 3 distinct colors.
        # If any cycle has 3 distinct colors, the new color would make it 4.
        
        can_use_new = True
        # Also, if we can't use new, we need to find a valid existing color.
        # The valid colors must be in the intersection of the color sets of all "critical" cycles.
        # A cycle is critical if its other 3 edges are colored and have 3 distinct colors.
        
        candidate_colors = set()
        has_critical_cycle = False
        
        for (e1, e2, e3) in cycles_for_edge[i]:
            # Check if other edges are colored
            if colors[e1] == -1 or colors[e2] == -1 or colors[e3] == -1:
                # Not fully colored yet, no constraint imposed by this cycle on the new color
                # because if we pick a new color, and later the other edges get colored,
                # we might violate it?
                # NO. The constraint is on the final state.
                # If we assign a color to edge i now, and the other edges are uncolored,
                # we don't know their colors yet.
                # However, if we assign a NEW color to edge i, and later the other 3 edges
                # get 3 distinct colors different from ours, we violate the constraint.
                # So we MUST ensure that if we pick a new color, it is "safe".
                # But we can't know the future colors.
                # This implies the greedy "new color if possible" strategy is flawed if it doesn't account for future constraints.
                # 
                # Alternative: Only check constraints on ALREADY COLORED edges.
                # If a cycle has uncolored edges, we assume we can fix it later?
                # No, if we color edge i with C_new, and later color e1=C1, e2=C2, e3=C3 (distinct from C_new and each other), we fail.
                # To be safe, we should only use a NEW color if it is guaranteed not to cause a violation.
                # But we can't guarantee that.
                #
                # However, note that the other edges e1, e2, e3 have indices > i or < i?
                # We are processing in lexicographical order.
                # If e1, e2, e3 are all < i, they are colored.
                # If any is > i, it is uncolored.
                #
                # If any is uncolored, can we use a new color?
                # If we use a new color, and later the uncolored edges are assigned colors,
                # we might end up with 4 distinct colors.
                # To maximize colors, we *want* to use new colors.
                # But we must ensure validity.
                # 
                # Strategy:
                # 1. If ALL other edges in ALL cycles are colored:
                #    - Check if any cycle has 3 distinct colors. If yes, new color is FORBIDDEN.
                #    - If no cycle has 3 distinct colors, new color is ALLOWED.
                # 2. If SOME other edges are uncolored:
                #    - We cannot be sure. 
                #    - However, if we use a new color, we risk violation.
                #    - If we use an existing color, we are safer?
                #    - Actually, if we use an existing color, we might still violate if the uncolored edges get distinct colors.
                #    - Example: Cycle edges: i (new), e1 (uncolored), e2 (uncolored), e3 (uncolored).
                #    - If we pick new color for i. Later e1=1, e2=2, e3=3. Violation.
                #    - If we pick existing color 1 for i. Later e1=2, e2=3, e3=4. Violation (1,2,3,4).
                #    
                # It seems we should only assign a color if it doesn't create an *immediate* violation with colored edges, and hope that the greedy order prevents future violations?
                # No, that's not guaranteed.
                #
                # Let's look at the "smallest available color" hint.
                # If we always pick the smallest valid color (where valid means: does not create 4 distinct colors in any cycle where the other 3 are colored), we might get stuck with few colors.
                # But the task is to MAXIMIZE.
                #
                # Let's try the "New Color if Safe" strategy.
                # "Safe" means: For every cycle containing i, if the other 3 edges are colored, they must NOT have 3 distinct colors.
                # If any cycle has 3 distinct colors (and is fully colored), we MUST pick a color from that set.
                # If multiple such cycles exist, we must pick a color in the intersection.
                # If the intersection is empty, we have a problem. (Backtracking needed?)
                # Given N=16, maybe greedy works without backtracking?
                #
                # What if a cycle has uncolored edges?
                # If we pick a new color, we don't know if it will conflict later.
                # But if we pick an existing color, we also don't know.
                # However, picking a NEW color increases the global count.
                # Let's assume that if no *fully colored* cycle forbids a new color, we should use a new color.
                # This is a heuristic.
                
                pass # Placeholder for logic refinement

        # Refined Logic:
        # 1. Identify all cycles containing edge i where the other 3 edges are ALREADY COLORED.
        # 2. For each such cycle, get the set of colors S.
        # 3. If |S| == 3, then the color of edge i MUST be in S.
        # 4. If |S| < 3, no constraint from this cycle.
        # 5. Let Allowed = Intersection of all S where |S| == 3.
        # 6. If Allowed is not empty:
        #    - We MUST pick a color from Allowed.
        #    - To maximize diversity, we should pick a color that is "least used" or just any? 
        #    - Actually, if we are forced to pick from Allowed, we can't use a new color.
        # 7. If Allowed is empty (i.e., no fully colored cycle has 3 distinct colors):
        #    - We CAN use a new color.
        #    - Should we? Yes, to maximize count.
        #    - But what if using a new color leads to a dead end later?
        #    - Given the symmetry and small N, let's try "New Color" if allowed.
        
        allowed_set = None
        for (e1, e2, e3) in cycles_for_edge[i]:
            if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                s = {colors[e1], colors[e2], colors[e3]}
                if len(s) == 3:
                    if allowed_set is None:
                        allowed_set = s
                    else:
                        allowed_set = allowed_set.intersection(s)
        
        if allowed_set is not None:
            # Must pick from allowed_set
            # Pick the smallest one? Or one that minimizes future conflicts?
            # Let's pick the smallest for determinism.
            c = min(allowed_set)
            colors[i] = c
        else:
            # Can use new color
            colors[i] = next_color
            next_color += 1
            
    return colors

# Wait, the logic above assumes that if no fully colored cycle has 3 distinct colors, a new color is safe.
# Is it possible that a new color causes a violation in a cycle with uncolored edges?
# Yes. But we can't prevent that without backtracking.
# However, if we always use new colors when possible, we might create many "critical" cycles later.
# Let's test this logic mentally.
# If it fails (returns invalid coloring), the score is 0.
# We need to be sure it's valid.
# The condition "No four-cycle may have four distinct edge colors" is a global constraint.
# The greedy check only looks at past edges.
# If we assign a new color to edge i, and later assign colors to e1, e2, e3 such that they are distinct from each other and from i, we violate.
# Does the greedy algorithm prevent this?
# When we color e1, we check cycles containing e1. One such cycle is the one with i, e2, e3.
# If i is colored, and e2, e3 are uncolored, the cycle is not "fully colored" from e1's perspective?
# No, e1 sees i (colored), e2 (uncolored), e3 (uncolored).
# So e1 can also pick a new color.
# Then e2 sees i, e1 (colored), e3 (uncolored). e2 picks new color.
# Then e3 sees i, e1, e2 (all colored).
# If i, e1, e2 have 3 distinct colors, e3 MUST pick one of them.
# So the last edge in the cycle is forced to reuse a color!
# This guarantees validity!
# 
# Key Insight: In any 4-cycle, the edge with the LARGEST index (processed last) will see the other 3 edges as colored.
# If those 3 edges have 3 distinct colors, the last edge MUST pick one of them.
# If those 3 edges have < 3 distinct colors, the last edge can pick a new color (and the cycle will have <= 3 distinct colors).
# Therefore, the greedy algorithm that checks constraints against *already colored* edges is GUARANTEED to produce a valid coloring.
# 
# So the logic is:
# 1. For edge i, look at all cycles where i is the max-index edge?
# No, we just look at all cycles where the OTHER 3 edges are colored.
# Since we process in increasing index, the other 3 edges are colored IF AND ONLY IF their indices are < i.
# So we only need to check cycles where the other 3 edges have indices < i.
# 
# My previous precomputation just listed all cycles.
# In the check, I verify `colors[e1] != -1`. This effectively filters for edges with index < i.
# 
# So the algorithm is correct and guaranteed valid.
# 
# To maximize colors:
# - If forced (intersection of S is non-empty), pick from it.
# - If not forced, pick a NEW color.
# 
# This should maximize the number of colors.
