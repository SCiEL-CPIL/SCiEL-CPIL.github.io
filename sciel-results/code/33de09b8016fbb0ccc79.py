"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import itertools
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20

    def is_valid_4cycle(c1, c2, c3, c4):
        """Check if 4 colors form a rainbow (all distinct)."""
        return len({c1, c2, c3, c4}) == 4

    def try_random_construction(seed_val):
        """Try to build a valid edge-colored graph using random greedy with backtracking."""
        random.seed(seed_val)
        
        # Order edges randomly
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        color_map = {}
        adj = {i: set() for i in range(n)}
        
        # For fast checking: for each pair of vertices (x,y), store the set of colors used
        # on edges that could form a 4-cycle with a new edge.
        # Actually, let's just check directly but efficiently.
        
        for u, v in all_edges:
            # Try each color
            found = False
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for c in colors:
                # Tentatively add edge
                color_map[(u, v)] = c
                adj[u].add(v)
                adj[v].add(u)
                
                # Check if this creates any non-rainbow 4-cycle
                # A 4-cycle containing edge (u,v) has form u-v-x-y-u
                # where v-x, x-y, y-u are existing edges.
                valid = True
                
                # Iterate over all possible x, y
                # x must be adjacent to v
                # y must be adjacent to u
                # y must be adjacent to x
                # u, v, x, y must be distinct
                
                for x in adj[v]:
                    if x == u:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == x:
                            continue
                        if y in adj[u]:
                            # Check 4-cycle u-v-x-y-u
                            c1 = color_map[(min(u,v), max(u,v))]
                            c2 = color_map[(min(v,x), max(v,x))]
                            c3 = color_map[(min(x,y), max(x,y))]
                            c4 = color_map[(min(y,u), max(y,u))]
                            if not is_valid_4cycle(c1, c2, c3, c4):
                                valid = False
                                break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    break
                else:
                    # Remove edge
                    del color_map[(u, v)]
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            # If no color works, skip this edge (it remains absent)
            
        # Convert to list format
        result = []
        for (u, v), c in color_map.items():
            result.append([u, v, c])
        return result

    # Try multiple random constructions
    best_edges = []
    best_count = 0
    
    # Use a range of seeds
    seed_counter = 0
    while time.time() < deadline:
        edges = try_random_construction(seed_counter)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            # If we get a very dense graph, we might be done
            if best_count > 180:
                break
        seed_counter += 1

    return best_edges
# EVOLVE-BLOCK-END
