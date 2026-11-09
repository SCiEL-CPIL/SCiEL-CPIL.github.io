"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def edge_conflicts(e1, e2, adj):
        """Check if two edges conflict (share vertex or are joined by an edge)."""
        u1, v1 = e1
        u2, v2 = e2
        # Share a vertex
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return True
        # Joined by an edge
        if v2 in adj[u1] or u2 in adj[v1] or v1 in adj[u2] or u1 in adj[v2]:
            return True
        return False
    
    def greedy_coloring(edges, adj, order=None):
        """Greedy coloring: assign each edge the smallest available color."""
        if order is None:
            order = list(range(len(edges)))
        
        color_of_edge = {}
        color_classes = []  # list of sets of edge indices per color
        
        for idx in order:
            e = edges[idx]
            # Find which colors are taken by conflicting edges
            taken = set()
            u, v = e
            # All edges that conflict with e
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            
            for other_idx, other_e in enumerate(edges):
                if other_idx == idx:
                    continue
                ou, ov = other_e
                # Check if other_e conflicts with e
                if ou in near or ov in near:
                    if other_idx in color_of_edge:
                        taken.add(color_of_edge[other_idx])
            
            # Assign smallest available color
            c = 0
            while c in taken:
                c += 1
            color_of_edge[idx] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(idx)
        
        return color_of_edge, color_classes
    
    def is_valid_coloring(edges, adj, color_of_edge):
        """Verify each color class is an induced matching."""
        color_classes = defaultdict(list)
        for idx, c in color_of_edge.items():
            color_classes[c].append(idx)
        
        for c, idxs in color_classes.items():
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    if edge_conflicts(edges[idxs[i]], edges[idxs[j]], adj):
                        return False
        return True
    
    def simulated_annealing(edges, adj, max_iters, start_temp, end_temp, seed):
        """Simulated annealing to minimize number of colors."""
        rng = random.Random(seed)
        n_edges = len(edges)
        
        # Start with greedy coloring
        color_of_edge, _ = greedy_coloring(edges, adj)
        best_coloring = dict(color_of_edge)
        best_num_colors = max(color_of_edge.values()) + 1 if color_of_edge else 0
        
        current_num_colors = best_num_colors
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Pick a random edge and try to change its color
            idx = rng.randrange(n_edges)
            old_color = color_of_edge[idx]
            
            # Try a new random color
            new_color = rng.randint(0, current_num_colors)  # allow up to current_num_colors
            
            if new_color == old_color:
                continue
            
            # Check if new_color is valid for this edge
            e = edges[idx]
            u, v = e
            near = {u, v}
            for x in (u, v):
                near.update(adj[x])
            
            valid = True
            for other_idx in range(n_edges):
                if other_idx == idx:
                    continue
                if color_of_edge[other_idx] == new_color:
                    other_e = edges[other_idx]
                    if edge_conflicts(e, other_e, adj):
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Compute delta in number of colors
            color_of_edge[idx] = new_color
            
            # Check if old color is now empty
            old_color_still_used = any(color_of_edge[i] == old_color for i in range(n_edges) if i != idx)
            new_color_newly_used = new_color > current_num_colors - 1
            
            if old_color_still_used and not new_color_newly_used:
                delta = 0
            elif old_color_still_used and new_color_newly_used:
                delta = 1
            elif not old_color_still_used and not new_color_newly_used:
                delta = -1
            else:  # old color freed, new color added
                delta = 0
            
            new_num_colors = current_num_colors + delta
            
            # Acceptance criterion
            if delta < 0 or (temp > 0 and rng.random() < min(1.0, delta / temp if delta > 0 else 1.0)):
                current_num_colors = new_num_colors
                if current_num_colors < best_num_colors:
                    best_num_colors = current_num_colors
                    best_coloring = dict(color_of_edge)
            else:
                # Reject
                color_of_edge[idx] = old_color
        
        return best_coloring, best_num_colors
    
    def local_search(edges, adj, color_of_edge, max_iters, seed):
        """Local search: try to reduce colors by moving edges."""
        rng = random.Random(seed)
        n_edges = len(edges)
        
        # Build color classes
        def get_color_classes(col):
            classes = defaultdict(list)
            for i, c in col.items():
                classes[c].append(i)
            return classes
        
        classes = get_color_classes(color_of_edge)
        num_colors = max(color_of_edge.values()) + 1 if color_of_edge else 0
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try to eliminate each color class
            for c in sorted(classes.keys(), reverse=True):
                if not classes[c]:
                    continue
                
                # Try to move each edge in this color class to another color
                moved_any = False
                for idx in classes[c][:]:  # iterate over copy
                    e = edges[idx]
                    u, v = e
                    near = {u, v}
                    for x in (u, v):
                        near.update(adj[x])
                    
                    # Try to move to an existing lower color
                    for new_c in range(c):
                        valid = True
                        for other_idx in classes.get(new_c, []):
                            if edge_conflicts(e, edges[other_idx], adj):
                                valid = False
                                break
                        if valid:
                            # Move it
                            color_of_edge[idx] = new_c
                            classes[c].remove(idx)
                            if new_c not in classes:
                                classes[new_c] = []
                            classes[new_c].append(idx)
                            moved_any = True
                            improved = True
                            break
                    if moved_any:
                        break
                
                # If color class is now empty, we've reduced colors
                if not classes[c]:
                    num_colors -= 1
        
        return color_of_edge, num_colors
    
    results = []
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        adj = build_adj(edges)
        n_edges = len(edges)
        
        # Get initial greedy coloring
        color_of_edge, _ = greedy_coloring(edges, adj)
        best_coloring = dict(color_of_edge)
        best_num_colors = max(color_of_edge.values()) + 1
        
        # Try multiple random orderings for greedy
        for trial in range(20):
            if time.time() > deadline:
                break
            order = list(range(n_edges))
            random.Random(trial * 1000 + spec[2]).shuffle(order)
            color_of_edge, _ = greedy_coloring(edges, adj, order)
            num_colors = max(color_of_edge.values()) + 1
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = dict(color_of_edge)
        
        # Simulated annealing with multiple restarts
        sa_start_temp = 5.0
        sa_end_temp = 0.01
        sa_max_iters = 5000
        
        for restart in range(10):
            if time.time() > deadline:
                break
            
            color_of_edge, num_colors = simulated_annealing(
                edges, adj, sa_max_iters, sa_start_temp, sa_end_temp, 
                seed=spec[2] * 100 + restart
            )
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = dict(color_of_edge)
        
        # Local search polish
        color_of_edge, num_colors = local_search(edges, adj, dict(best_coloring), max_iters=1000, seed=spec[2])
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = dict(color_of_edge)
        
        # Verify validity
        if not is_valid_coloring(edges, adj, best_coloring):
            # Fallback to simple greedy
            color_of_edge, _ = greedy_coloring(edges, adj)
            best_coloring = dict(color_of_edge)
        
        # Build output in sorted edge order
        result = [best_coloring[i] for i in range(n_edges)]
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
