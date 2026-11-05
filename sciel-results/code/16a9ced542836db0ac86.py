"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def get_conflicts(edge, adj, n):
        """Get all vertices that conflict with this edge (endpoints + their neighbors)."""
        u, v = edge
        near = {u, v}
        near.update(adj[u])
        near.update(adj[v])
        return near

    def can_assign_edge(idx, c, colour, edges, adj, n, color_vertices, color_edges):
        """Check if edge idx can be assigned color c."""
        u, v = edges[idx]
        near = get_conflicts((u, v), adj, n)
        
        # Check vertex conflicts
        for vert in color_vertices[c]:
            if vert in near:
                return False
        
        # Check edge-joining conflicts
        for (a, b) in color_edges[c]:
            if a == u or a == v or b == u or b == v:
                continue  # shares a vertex, already handled above
            if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                return False
        
        return True

    def greedy_color(edges, adj, n, color_order=None):
        """Greedy edge coloring with induced matching constraint."""
        if color_order is None:
            color_order = list(range(len(edges)))
        
        colour = {}
        color_vertices = defaultdict(set)  # color -> set of vertices used
        color_edges = defaultdict(set)  # color -> set of edges
        
        for idx in color_order:
            u, v = edges[idx]
            
            best_color = None
            for c in range(len(color_vertices)):
                if can_assign_edge(idx, c, colour, edges, adj, n, color_vertices, color_edges):
                    best_color = c
                    break
            
            if best_color is None:
                best_color = len(color_vertices)
            
            colour[idx] = best_color
            color_vertices[best_color].add(u)
            color_vertices[best_color].add(v)
            color_edges[best_color].add((u, v))
        
        return colour

    def local_improve(colour, edges, adj, n, max_iters=100):
        """Try to reduce number of colors by recoloring edges."""
        m = len(edges)
        num_colors = max(colour.values()) + 1 if colour else 0
        
        # Precompute conflict info for each edge
        edge_near = [get_conflicts(edges[i], adj, n) for i in range(m)]
        
        # Build color -> list of edge indices
        color_to_edges = defaultdict(list)
        for idx in range(m):
            color_to_edges[colour[idx]].append(idx)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            order = list(range(m))
            random.shuffle(order)
            
            for idx in order:
                u, v = edges[idx]
                current_color = colour[idx]
                
                # Try all colors from 0 to current_color - 1
                for c in range(current_color):
                    if not color_to_edges[c]:
                        continue
                    
                    # Check if we can assign color c
                    can = True
                    for other_idx in color_to_edges[c]:
                        if other_idx == idx:
                            continue
                        ou, ov = edges[other_idx]
                        # Vertex conflict
                        if ou in edge_near[idx] or ov in edge_near[idx]:
                            can = False
                            break
                        # Edge-joining conflict
                        if ou == u or ou == v or ov == u or ov == v:
                            continue
                        if ou in adj[u] or ou in adj[v] or ov in adj[u] or ov in adj[v]:
                            can = False
                            break
                    
                    if can:
                        colour[idx] = c
                        color_to_edges[current_color].remove(idx)
                        color_to_edges[c].append(idx)
                        improved = True
                        break
            
            if not improved:
                break
        
        num_colors = max(colour.values()) + 1 if colour else 0
        return num_colors

    def try_merge_colors(colour, edges, adj, n, max_attempts=30):
        """Try to merge color classes by recoloring conflicting edges."""
        m = len(edges)
        if not colour:
            return 0
        
        num_colors = max(colour.values()) + 1
        edge_near = [get_conflicts(edges[i], adj, n) for i in range(m)]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            if num_colors <= 1:
                break
            
            # Try to merge the highest color into a lower one
            src_color = num_colors - 1
            src_edges = [i for i in range(m) if colour[i] == src_color]
            
            if not src_edges:
                num_colors -= 1
                continue
            
            # Try each target color
            merged = False
            for tgt_color in range(src_color):
                tgt_edges = [i for i in range(m) if colour[i] == tgt_color]
                
                # Count how many src_edges conflict with tgt_color
                movable = 0
                for idx in src_edges:
                    u, v = edges[idx]
                    can_move = True
                    for other_idx in tgt_edges:
                        if other_idx == idx:
                            continue
                        ou, ov = edges[other_idx]
                        if ou in edge_near[idx] or ov in edge_near[idx]:
                            can_move = False
                            break
                        if ou == u or ou == v or ov == u or ov == v:
                            continue
                        if ou in adj[u] or ou in adj[v] or ov in adj[u] or ov in adj[v]:
                            can_move = False
                            break
                    if can_move:
                        movable += 1
                
                # Only merge if most edges can be moved directly
                if movable >= len(src_edges) * 0.7:
                    # Move all src edges to tgt_color
                    for idx in src_edges:
                        colour[idx] = tgt_color
                    num_colors -= 1
                    merged = True
                    break
            
            if not merged:
                # Try to move individual edges from src to any lower color
                moved_any = False
                for idx in src_edges:
                    u, v = edges[idx]
                    for c in range(src_color):
                        # Check if idx can be assigned to color c
                        can = True
                        for other_idx in range(m):
                            if other_idx == idx or colour[other_idx] != c:
                                continue
                            ou, ov = edges[other_idx]
                            if ou in edge_near[idx] or ov in edge_near[idx]:
                                can = False
                                break
                            if ou == u or ou == v or ov == u or ov == v:
                                continue
                            if ou in adj[u] or ou in adj[v] or ov in adj[u] or ov in adj[v]:
                                can = False
                                break
                        if can:
                            colour[idx] = c
                            moved_any = True
                            break
                
                if not moved_any:
                    break
        
        num_colors = max(colour.values()) + 1 if colour else 0
        return num_colors

    def polish(colour, edges, adj, n, max_iters=200):
        """High-precision polish: repeatedly try to reduce colors."""
        m = len(edges)
        if not colour:
            return 0
        
        num_colors = max(colour.values()) + 1
        edge_near = [get_conflicts(edges[i], adj, n) for i in range(m)]
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            order = list(range(m))
            random.shuffle(order)
            
            for idx in order:
                u, v = edges[idx]
                current_color = colour[idx]
                
                for c in range(current_color):
                    can = True
                    for other_idx in range(m):
                        if other_idx == idx or colour[other_idx] != c:
                            continue
                        ou, ov = edges[other_idx]
                        if ou in edge_near[idx] or ov in edge_near[idx]:
                            can = False
                            break
                        if ou == u or ou == v or ov == u or ov == v:
                            continue
                        if ou in adj[u] or ou in adj[v] or ov in adj[u] or ov in adj[v]:
                            can = False
                            break
                    
                    if can:
                        colour[idx] = c
                        improved = True
                        break
            
            if not improved:
                break
        
        num_colors = max(colour.values()) + 1 if colour else 0
        return num_colors

    def color_instance(edges, adj, n):
        """Color a single instance with multiple random orders, local improvement, merging, and polish."""
        m = len(edges)
        best_colour = None
        best_num_colors = float('inf')
        
        # Try multiple random orderings
        num_tries = 15
        rng = random.Random(42)
        
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            
            # Random order
            order = list(range(m))
            rng.shuffle(order)
            
            colour = greedy_color(edges, adj, n, order)
            num_colors = max(colour.values()) + 1 if colour else 0
            
            # Local improvement
            num_colors = local_improve(colour, edges, adj, n)
            
            # Try merging color classes
            num_colors = try_merge_colors(colour, edges, adj, n)
            
            # Final polish
            num_colors = polish(colour, edges, adj, n)
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colour = dict(colour)
        
        # Return colors in sorted edge order
        if best_colour is None:
            best_colour = greedy_color(edges, adj, n)
            best_num_colors = max(best_colour.values()) + 1 if best_colour else 0
        
        return [best_colour[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colours = color_instance(edges, adj, n)
        out.append(colours)
    
    return out
# EVOLVE-BLOCK-END
