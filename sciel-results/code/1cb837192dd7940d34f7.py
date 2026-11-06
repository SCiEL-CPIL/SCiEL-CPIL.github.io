"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    n = 20
    colors = list(range(20))
    
    # Represent graph as edge_color dict: (u,v) -> color, with u < v
    edge_color = {}
    
    # Helper: check if adding edge (u,v) with color c creates a bad 4-cycle
    # A 4-cycle containing (u,v): for any w != u,v, if (u,w) and (v,w) exist,
    # and there's a 4th vertex... Actually a 4-cycle is u-a-v-b-u or u-a-b-v-u etc.
    # Simpler: enumerate all 4-cycles in the current graph that would include new edge.
    
    def find_bad_4cycles_with_edge(u, v, c):
        """Check if adding (u,v) with color c creates any non-rainbow 4-cycle."""
        # A 4-cycle containing edge (u,v) has form u-x-v-y-u where x,y are other vertices
        # and edges (u,x),(x,v),(v,y),(y,u) all exist (or will exist with new edge)
        # Actually 4-cycle: u - x - v - y - u, needs edges (u,x),(x,v),(v,y),(y,u)
        # The new edge is (u,v). So the 4-cycle must use (u,v) plus 3 other edges.
        # Form: u - v - x - y - u: edges (u,v),(v,x),(x,y),(y,u)
        # So for each x,y: if (v,x),(x,y),(y,u) exist, check colors
        bad = []
        adj = defaultdict(set)
        for (a,b), col in edge_color.items():
            adj[a].add(b)
            adj[b].add(a)
        
        for x in range(n):
            if x == u or x == v:
                continue
            if v not in adj[x]:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if x not in adj[y] or y not in adj[u]:
                    continue
                # 4-cycle: u-v-x-y-u
                # Edges: (u,v) new with color c, (v,x), (x,y), (y,u)
                c_vx = edge_color.get((min(v,x), max(v,x)))
                c_xy = edge_color.get((min(x,y), max(x,y)))
                c_yu = edge_color.get((min(y,u), max(y,u)))
                if c_vx is None or c_xy is None or c_yu is None:
                    continue
                col_set = [c, c_vx, c_xy, c_yu]
                if len(set(col_set)) < 4:
                    bad.append((x, y, col_set))
        return bad
    
    def is_valid_full():
        """Check all 4-cycles are rainbow."""
        adj = defaultdict(set)
        for (a,b), col in edge_color.items():
            adj[a].add(b)
            adj[b].add(a)
        
        # Enumerate all 4-cycles: for each pair of vertices (u,v) as a diagonal
        # 4-cycle u-a-v-b: edges (u,a),(a,v),(v,b),(b,u)
        # To avoid duplicates, consider u < v as diagonal, and a < b
        for u in range(n):
            for v in range(u+1, n):
                # common neighbors of u and v
                common = adj[u] & adj[v]
                common_list = list(common)
                for i in range(len(common_list)):
                    for j in range(i+1, len(common_list)):
                        a = common_list[i]
                        b = common_list[j]
                        # 4-cycle: u-a-v-b-u
                        c_ua = edge_color.get((min(u,a), max(u,a)))
                        c_av = edge_color.get((min(a,v), max(a,v)))
                        c_vb = edge_color.get((min(v,b), max(v,b)))
                        c_bu = edge_color.get((min(b,u), max(b,u)))
                        if None in (c_ua, c_av, c_vb, c_bu):
                            continue
                        if len(set([c_ua, c_av, c_vb, c_bu])) < 4:
                            return False
        return True
    
    # Start with a base structure: complete bipartite K_{10,10}
    # Left: 0-9, Right: 10-19
    # Color (i, j) where i in 0-9, j in 10-19
    # Use color = (i * 2 + (j - 10)) mod 20 ... let's try a systematic approach
    
    # Actually, let's use a known good structure and then optimize
    # Start with a cycle or sparse graph, then add edges greedily
    
    # Phase 1: Build a dense base using complete bipartite with careful coloring
    # K_{10,10}: vertices 0-9 (left), 10-19 (right)
    # Color edge (i, j) = (i + (j-10)*2) mod 20
    for i in range(10):
        for j in range(10, 20):
            c = (i + (j-10)*2) % 20
            edge_color[(i, j)] = c
    
    # Verify base is valid
    if not is_valid_full():
        # Fallback: use a simpler structure
        edge_color = {}
        # Just use a star
        for v in range(1, 20):
            edge_color[(0, v)] = 0
        # Try to add more
        pass
    
    # Phase 2: Try to add intra-side edges and optimize via color reassignment
    # For each potential edge not in graph, try to add it with some color
    # If it creates conflicts, try reassigning colors of existing edges in conflicting 4-cycles
    
    def try_add_edge(u, v):
        """Try to add edge (u,v) with some color, possibly reassigning existing colors."""
        if u == v:
            return False
        key = (min(u,v), max(u,v))
        if key in edge_color:
            return True
        
        # Try each color
        for c in colors:
            bad = find_bad_4cycles_with_edge(u, v, c)
            if not bad:
                edge_color[key] = c
                return True
            # If there are bad cycles, try to fix by reassigning colors
            # For each bad cycle, we need to change one of the 4 edge colors
            # This is complex; skip for now in greedy phase
        return False
    
    # Add all possible edges greedily
    for u in range(n):
        for v in range(u+1, n):
            if time.time() > deadline - 20:
                break
            try_add_edge(u, v)
        if time.time() > deadline - 20:
            break
    
    # Phase 3: Local search - try to improve by reassigning colors to allow more edges
    # or by swapping colors to resolve conflicts
    best_edge_count = len(edge_color)
    best_state = dict(edge_color)
    
    # Local search: try reassigning colors of existing edges to free up color slots
    # for adding new edges
    import copy
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        current_count = len(edge_color)
        if current_count >= best_edge_count:
            best_edge_count = current_count
            best_state = dict(edge_color)
        
        # Try to add a missing edge
        missing = [(u,v) for u in range(n) for v in range(u+1,n) if (u,v) not in edge_color]
        if not missing:
            break
        
        random.shuffle(missing)
        improved = False
        for (u,v) in missing[:50]:
            if time.time() > deadline - 10:
                break
            # Try to add with color reassignment
            saved = dict(edge_color)
            added = try_add_edge(u, v)
            if added and len(edge_color) > len(saved):
                improved = True
                break
            else:
                edge_color.clear()
                edge_color.update(saved)
        
        if not improved:
            # Try color swaps to resolve conflicts
            # Pick a random edge and try changing its color
            if edge_color:
                edges_list = list(edge_color.keys())
                random.shuffle(edges_list)
                for (u,v) in edges_list[:20]:
                    if time.time() > deadline - 10:
                        break
                    old_c = edge_color[(u,v)]
                    for new_c in colors:
                        if new_c == old_c:
                            continue
                        saved = dict(edge_color)
                        edge_color[(u,v)] = new_c
                        if is_valid_full():
                            # Try to add an edge now
                            for (a,b) in missing[:10]:
                                if try_add_edge(a, b):
                                    improved = True
                                    break
                        if not improved:
                            edge_color.clear()
                            edge_color.update(saved)
                    if improved:
                        break
    
    # Return best state
    if len(best_state) > len(edge_color):
        edge_color = best_state
    
    result = []
    for (u,v), c in edge_color.items():
        result.append([u, v, c])
    return result
# EVOLVE-BLOCK-END
