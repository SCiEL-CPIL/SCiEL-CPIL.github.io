import random
import sys
import time

N = 16
sys.setrecursionlimit(10000)

def construct():
    # Generate all edges in lexicographic order
    edges = []
    edge_index = {}
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_index[(i, j)] = len(edges) - 1
            edge_index[(j, i)] = len(edges) - 1
    
    num_edges = len(edges)
    
    # Precompute 4-cycles
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x.
    # The cycles are (u,v,w,x), (u,v,x,w), (u,w,v,x) in terms of edge sets.
    # Edges in cycle 1: (u,v), (v,w), (w,x), (x,u)
    # Edges in cycle 2: (u,v), (v,x), (x,w), (w,u)
    # Edges in cycle 3: (u,w), (w,v), (v,x), (x,u)
    
    # For each edge, store the list of other edges that form a 4-cycle with it.
    # cycle_neighbors[e_idx] = list of (e1, e2, e3) such that {e_idx, e1, e2, e3} is a 4-cycle.
    cycle_neighbors = [[] for _ in range(num_edges)]
    
    for u in range(N):
        for v in range(u + 1, N):
            for w in range(v + 1, N):
                for x in range(w + 1, N):
                    # Vertices u, v, w, x are distinct.
                    # Cycle 1: u-v-w-x-u
                    e_uv = edge_index[(u, v)]
                    e_vw = edge_index[(v, w)]
                    e_wx = edge_index[(w, x)]
                    e_xu = edge_index[(x, u)]
                    
                    cycle_neighbors[e_uv].append((e_vw, e_wx, e_xu))
                    cycle_neighbors[e_vw].append((e_wx, e_xu, e_uv))
                    cycle_neighbors[e_wx].append((e_xu, e_uv, e_vw))
                    cycle_neighbors[e_xu].append((e_uv, e_vw, e_wx))
                    
                    # Cycle 2: u-v-x-w-u
                    e_vx = edge_index[(v, x)]
                    e_xw = edge_index[(w, x)] # Same as e_wx
                    e_wu = edge_index[(u, w)]
                    
                    # Note: e_wx is (w,x). e_xw is same edge.
                    # Cycle edges: (u,v), (v,x), (x,w), (w,u)
                    cycle_neighbors[e_uv].append((e_vx, e_wx, e_wu))
                    cycle_neighbors[e_vx].append((e_wx, e_wu, e_uv))
                    cycle_neighbors[e_wx].append((e_wu, e_uv, e_vx))
                    cycle_neighbors[e_wu].append((e_uv, e_vx, e_wx))
                    
                    # Cycle 3: u-w-v-x-u
                    e_uv2 = edge_index[(u, v)]
                    e_wv = edge_index[(v, w)] # Same as e_vw
                    e_vx2 = edge_index[(v, x)] # Same as e_vx
                    e_xu2 = edge_index[(u, x)] # Same as e_xu
                    
                    # Cycle edges: (u,w), (w,v), (v,x), (x,u)
                    cycle_neighbors[e_wu].append((e_vw, e_vx, e_xu))
                    cycle_neighbors[e_vw].append((e_vx, e_xu, e_wu))
                    cycle_neighbors[e_vx].append((e_xu, e_wu, e_vw))
                    cycle_neighbors[e_xu].append((e_wu, e_vw, e_vx))

    best_count = 0
    best_assignment = [0] * num_edges
    colors = [-1] * num_edges
    used_colors = set()
    
    # Heuristic: Sort edges? 
    # Lexicographic order is fine.
    # Value Ordering: 
    # To maximize distinct colors, we should try to assign new colors.
    # However, if we assign a new color, it might block many future edges.
    # A good strategy is to try new colors first, but if that leads to a dead end, backtrack.
    
    # We will use a randomized backtracking search.
    # To ensure we finish within 60s, we need to be efficient.
    # We will limit the depth of search or the number of nodes if it gets too slow, 
    # but given the structure, a good heuristic should find a solution quickly.
    
    # Actually, for "Maximize number of colors", a greedy approach with randomization and local search might be better, 
    # but the directive says "randomized backtracking search".
    
    # Let's implement a DFS that tries to maximize colors.
    
    def dfs(idx):
        nonlocal best_count, best_assignment
        
        if idx == num_edges:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_assignment = colors[:]
            return True
        
        # Get valid colors for edges[idx]
        # A color c is valid if for all (e1, e2, e3) in cycle_neighbors[idx]:
        # if colors[e1], colors[e2], colors[e3] are all assigned:
        #    len({c, colors[e1], colors[e2], colors[e3]}) <= 3
        
        # Optimization: Pre-compute the set of "forbidden" colors for each edge based on current assignment?
        # That's expensive.
        
        # Instead, iterate through candidates.
        # Candidates: 
        # 1. New colors (not in used_colors)
        # 2. Old colors (in used_colors)
        
        # To maximize colors, try new colors first.
        # But which new color? Any new color is symmetric in terms of "newness".
        # So we can just pick one new color, say max(used_colors)+1 or 0 if empty.
        # Wait, the domain is 0..119.
        # If we just pick the smallest available new color, we are effectively using colors 0, 1, 2...
        # This is a valid strategy.
        
        # However, we might need to use specific colors to satisfy constraints?
        # No, the constraint is only on distinctness.
        
        # Strategy:
        # 1. Try a new color (smallest unused).
        # 2. If that fails, try existing colors.
        
        # But what if the smallest unused color causes a conflict, but a larger unused color doesn't?
        # Since the colors are just labels, if color X causes a conflict, any other new color Y (not used) 
        # will cause the same conflict pattern relative to the *set* of used colors?
        # No. The conflict depends on the specific values.
        # Example: Used {1, 2}. Cycle has edges with colors 1, 2. 
        # If we assign 3, set is {1,2,3}, size 3. OK.
        # If we assign 1, set is {1,2}, size 2. OK.
        # If we assign 2, set is {1,2}, size 2. OK.
        # If we assign 4, set is {1,2,4}, size 3. OK.
        
        # The only time a new color fails is if it completes a set of 4 distinct colors with 3 existing distinct colors.
        # So, if the 3 existing colors in a cycle are distinct, say {1, 2, 3}, then we CANNOT assign any new color 
        # that is not in {1, 2, 3}. But 1, 2, 3 are already used.
        # So if a cycle has 3 distinct existing colors, we MUST use one of those 3 colors for the 4th edge.
        # This means we CANNOT introduce a new color in that position.
        
        # So the logic is:
        # For edge e, check all cycles containing e.
        # If any cycle has 3 other edges assigned with 3 distinct colors {c1, c2, c3}, 
        # then e MUST be colored with c1, c2, or c3.
        # This restricts the domain of e to {c1, c2, c3}.
        
        # If no such cycle exists, e can take any color (new or old).
        
        valid_colors = None
        for e1, e2, e3 in cycle_neighbors[idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            if c1 != -1 and c2 != -1 and c3 != -1:
                s = {c1, c2, c3}
                if len(s) == 3:
                    # e must be in s
                    if valid_colors is None:
                        valid_colors = set(s)
                    else:
                        valid_colors.intersection_update(s)
                        if not valid_colors:
                            return False
        
        if valid_colors is None:
            # No restriction from "3 distinct" cycles.
            # We can try new colors or old colors.
            # To maximize count, try a new color first.
            # Pick the smallest unused color in 0..119.
            # Or just try all? No, try new first.
            
            # Find smallest unused color
            next_new_color = None
            for c in range(120):
                if c not in used_colors:
                    next_new_color = c
                    break
            
            candidates = []
            if next_new_color is not None:
                candidates.append(next_new_color)
            
            # Then try old colors
            # To break symmetry, we can just pick one old color? 
            # Or try all?
            # If we try all old colors, branching factor is high.
            # But if we have a valid solution with K colors, we can always map colors to 0..K-1.
            # So the specific values of old colors don't matter for the *count*, only the structure.
            # However, the constraints depend on equality.
            # If we are free to choose, picking any old color is symmetric?
            # Not exactly. Different old colors might have different "conflict histories".
            # But if we just want to find *a* solution with max colors, 
            # we can prioritize new colors. If new colors fail, we must use old ones.
            
            # Let's try the new color first.
            # If it fails, try old colors.
            # To speed up, if we have many old colors, we can just pick one randomly?
            # No, we need to be sure.
            
            # Let's construct the list of candidates:
            # 1. New color
            # 2. Old colors (shuffled)
            
            old_list = list(used_colors)
            random.shuffle(old_list)
            candidates.extend(old_list)
            
        else:
            # Must use one of valid_colors
            candidates = list(valid_colors)
            random.shuffle(candidates)
        
        for c in candidates:
            colors[idx] = c
            if c not in used_colors:
                used_colors.add(c)
            
            if dfs(idx + 1):
                return True
            
            colors[idx] = -1
            if c not in used_colors:
                used_colors.remove(c)
                
        return False

    # Run search
    # To handle potential timeouts or getting stuck in bad branches, 
    # we can add a node limit and restart.
    
    nodes_visited = 0
    MAX_NODES = 5000000 # 5 million nodes
    
    def dfs_limited(idx):
        nonlocal best_count, best_assignment, nodes_visited
        
        if nodes_visited > MAX_NODES:
            return False
        
        nodes_visited += 1
        
        if idx == num_edges:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_assignment = colors[:]
            return True
        
        valid_colors = None
        for e1, e2, e3 in cycle_neighbors[idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            
            if c1 != -1 and c2 != -1 and c3 != -1:
                s = {c1, c2, c3}
                if len(s) == 3:
                    if valid_colors is None:
                        valid_colors = set(s)
                    else:
                        valid_colors.intersection_update(s)
                        if not valid_colors:
                            return False
        
        if valid_colors is None:
            next_new_color = None
            for c in range(120):
                if c not in used_colors:
                    next_new_color = c
                    break
            
            candidates = []
            if next_new_color is not None:
                candidates.append(next_new_color)
            
            old_list = list(used_colors)
            random.shuffle(old_list)
            candidates.extend(old_list)
            
        else:
            candidates = list(valid_colors)
            random.shuffle(candidates)
        
        for c in candidates:
            colors[idx] = c
            if c not in used_colors:
                used_colors.add(c)
            
            if dfs_limited(idx + 1):
                return True
            
            colors[idx] = -1
            if c not in used_colors:
                used_colors.remove(c)
                
        return False

    # Try multiple times with different seeds to escape local minima (if any)
    # Although this is exact search, the order of old colors is randomized.
    
    best_found = 0
    for attempt in range(10):
        random.seed(attempt)
        colors = [-1] * num_edges
        used_colors = set()
        nodes_visited = 0
        dfs_limited(0)
        
        if best_count > best_found:
            best_found = best_count
            
        if best_found >= 110: # High enough?
            break
            
    if best_assignment is None or all(x == 0 for x in best_assignment):
        return [0] * num_edges
        
    return best_assignment
