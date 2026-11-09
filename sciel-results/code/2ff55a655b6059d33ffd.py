import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety

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

    def solve_instance(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        # Build incompatibility graph
        # Two edges are incompatible if they share a vertex OR are joined by an edge.
        # We want to color such that each color class is a set of mutually compatible edges.
        # This is equivalent to coloring the incompatibility graph.
        
        # For efficiency, we'll build a list of lists of incompatible edges.
        # Since m is small (~200-500), O(m^2) is fine.
        incompatible = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # Check if they form an induced matching
                # 1. Disjoint vertices
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
                    continue
                # 2. No edge between {u1, v1} and {u2, v2}
                if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                    incompatible[i].add(j)
                    incompatible[j].add(i)
        
        # DSATUR-based initial coloring
        colors = [-1] * m
        color_classes = [[] for _ in range(m)]  # color_classes[c] = list of vertices with color c
        
        def dsatur_color():
            nonlocal colors, color_classes
            colors = [-1] * m
            color_classes = [[] for _ in range(m)]
            
            # DSATUR: at each step, pick uncolored vertex with max saturation degree,
            # break ties by degree in incompatibility graph
            uncolored = set(range(m))
            for _ in range(m):
                if not uncolored:
                    break
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    # Saturation degree: number of distinct colors among neighbors
                    sat_colors = set()
                    for u in incompatible[v]:
                        if colors[u] != -1:
                            sat_colors.add(colors[u])
                    sat = len(sat_colors)
                    deg = len(incompatible[v])
                    if sat > best_sat or (sat == best_sat and deg > best_deg):
                        best_sat = sat
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                # Assign smallest available color
                used = set()
                for u in incompatible[best_v]:
                    if colors[u] != -1:
                        used.add(colors[u])
                
                c = 0
                while c in used:
                    c += 1
                colors[best_v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(best_v)
                uncolored.discard(best_v)
        
        dsatur_color()
        
        current_color = len(color_classes)
        
        # Local search: try to reduce number of colors via move and swap operations
        def can_move(v, c):
            """Check if vertex v can be moved to color c"""
            for u in incompatible[v]:
                if colors[u] == c:
                    return False
            return True
        
        def can_swap(v1, v2, c1, c2):
            """Check if swapping colors of v1 and v2 is valid"""
            # After swap: v1 gets c2, v2 gets c1
            # Check v1's neighbors with color c2 (excluding v2)
            for u in incompatible[v1]:
                if colors[u] == c2 and u != v2:
                    return False
            # Check v2's neighbors with color c1 (excluding v1)
            for u in incompatible[v2]:
                if colors[u] == c1 and u != v1:
                    return False
            return True
        
        improved = True
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to remove a color by moving all its vertices to other colors
            for c in range(current_color - 1, -1, -1):
                if len(color_classes[c]) == 0:
                    continue
                
                # Try to move each vertex in color c to another color
                moved_any = False
                for v in list(color_classes[c]):
                    moved = False
                    for c2 in range(c):
                        if can_move(v, c2):
                            # Move v from c to c2
                            color_classes[c].remove(v)
                            color_classes[c2].append(v)
                            colors[v] = c2
                            moved = True
                            moved_any = True
                            break
                    if not moved:
                        # Can't move this vertex, so we can't eliminate color c
                        break
                
                if moved_any and len(color_classes[c]) == 0:
                    current_color -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap operations to enable future color removals
            for c1 in range(current_color):
                for c2 in range(c1 + 1, current_color):
                    if not color_classes[c1] or not color_classes[c2]:
                        continue
                    for v1 in color_classes[c1]:
                        for v2 in color_classes[c2]:
                            if can_swap(v1, v2, c1, c2):
                                # Perform swap
                                colors[v1], colors[v2] = c2, c1
                                color_classes[c1].remove(v1)
                                color_classes[c2].remove(v2)
                                color_classes[c2].append(v1)
                                color_classes[c1].append(v2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Since we haven't processed this instance, we can't return a valid coloring for it yet.
            # But the loop structure allows us to check before starting an instance.
            # If we are in the middle of an instance, we'd need to handle it inside solve_instance.
            # Given the speed of the new algorithm, this is unlikely.
            pass
            
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colors = solve_instance(edges, adj)
        out.append(colors)
        
    return out
