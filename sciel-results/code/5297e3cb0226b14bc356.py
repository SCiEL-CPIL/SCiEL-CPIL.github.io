"""Strong edge colourings via Maximum Induced Matching extraction + local refinement."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict, deque
from math import exp

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def build_conflict_graph(edges, n):
        m = len(edges)
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        conflict_adj = [set() for _ in range(m)]
        
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        return conflict_adj

    def find_mis_greedy(conflict_adj, candidates, rng):
        """
        Find a Maximum Independent Set (Induced Matching) in the subgraph induced by candidates.
        Uses a greedy heuristic: repeatedly pick a vertex with minimum degree in the remaining graph.
        """
        if not candidates:
            return []
        
        # candidates is a set of edge indices
        remaining = set(candidates)
        mis = []
        
        # To make it faster, we can just iterate.
        # Since m is small (< 200), O(m^2) or O(m^3) is fine.
        
        # We maintain a set of available vertices.
        # A vertex is available if it is in remaining.
        
        while remaining:
            # Find vertex with min degree in the induced subgraph of remaining
            min_deg = float('inf')
            chosen = -1
            for v in remaining:
                deg = len(conflict_adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
                    # Tie-breaker: random
                    if deg == min_deg and rng.random() < 0.5:
                        pass # Keep first found or randomize? 
                        # Let's randomize among min-degree vertices to allow diversity
                        # Actually, just picking one is fine.
            
            # Add chosen to MIS
            mis.append(chosen)
            # Remove chosen and its neighbors from remaining
            remaining.discard(chosen)
            for nb in conflict_adj[chosen]:
                if nb in remaining:
                    remaining.discard(nb)
        
        return mis

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
            
            colours = [0] * m
            used_colors = 0
            remaining = set(range(m))
            
            # 1. Greedy MIM Packing
            while remaining:
                # Find MIS in remaining
                matching = find_mis_greedy(conflict_adj, remaining, rng)
                
                # Assign color
                for e in matching:
                    colours[e] = used_colors
                
                # Remove from remaining
                remaining.difference_update(matching)
                used_colors += 1
                
            current_count = used_colors
            
            # 2. Refinement: Try to remove the highest color
            # This is similar to local search but operates on the set of colors
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                max_c = max(colours) if m > 0 else -1
                # Try to remove max_c
                edges_with_max = [i for i in range(m) if colours[i] == max_c]
                
                # Check if we can reassign all edges_with_max to colors < max_c
                # We do this by trying to find a valid assignment for each edge sequentially
                # or using a simple backtracking if the set is small.
                # Given the structure, sequential greedy reassignment might fail if order matters.
                # Let's try a simple heuristic: shuffle edges, try to move each to a lower color.
                
                old_colours = colours[:]
                success = True
                # Sort by degree in conflict graph? Or random?
                # Random is safer for diversity.
                rng.shuffle(edges_with_max)
                
                for e in edges_with_max:
                    blocked = set()
                    for nb in conflict_adj[e]:
                        if colours[nb] != max_c: # Only consider colors that are NOT max_c (since we are removing max_c)
                            # Actually, other edges with max_c are also being moved, so their current color is max_c.
                            # If we move e, it can't conflict with any edge currently having a color < max_c.
                            # It also can't conflict with edges that have been moved to < max_c already.
                            # It CAN conflict with edges that still have max_c (not yet moved).
                            # So we must ensure e's new color is not used by ANY neighbor.
                            # But neighbors with max_c are "unavailable" as targets for now? No, they are targets for other edges.
                            # The constraint is: new_color(e) not in { colours[nb] for nb in neighbors }.
                            # If nb has max_c, and we haven't moved it yet, colours[nb] is max_c.
                            # We are trying to assign colors < max_c. So max_c is not a candidate.
                            # So we just need new_color not in { colours[nb] for nb in neighbors if colours[nb] < max_c } 
                            # AND new_color not in { colours[nb] for nb in neighbors if colours[nb] == max_c }?
                            # No, if nb has max_c, and we assign e a color c < max_c, there is no conflict because c != max_c.
                            # Wait, the conflict is about SHARING a color.
                            # If e gets color c, and nb has color c, conflict.
                            # If nb has color max_c, and e gets color c < max_c, no conflict.
                            # So we only care about neighbors with color < max_c.
                            if colours[nb] < max_c:
                                blocked.add(colours[nb])
                    
                    # Find a valid color in [0, max_c)
                    valid = False
                    for c in range(max_c):
                        if c not in blocked:
                            colours[e] = c
                            valid = True
                            break
                    
                    if not valid:
                        success = False
                        break
                
                if success:
                    # We successfully removed max_c
                    # Need to recompute max or just mark improved
                    improved = True
                    # Recalculate current count
                    # We removed one color, so count decreases by 1
                    # But we might have introduced gaps? No, we used existing colors.
                    # The number of distinct colors decreases by 1.
                    current_count -= 1
                    # We should break the inner loop or continue?
                    # Continue to see if we can remove more.
                    # But we modified colours. The next iteration will check the new max.
                    # We need to update the "edges_with_max" logic for the next loop.
                    # The while loop recalculates max_c.
                    pass
                else:
                    # Restore
                    colours = old_colours
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            # Early stop if we hit the bound
            lb = 5 * d * d // 4
            if current_count <= lb:
                break
        
        if best_colours is None:
            # Fallback: simple greedy coloring
            colours = [0] * m
            coloured_set = set()
            for idx in range(m):
                blocked = set()
                for nb in conflict_adj[idx]:
                    if nb in coloured_set:
                        blocked.add(colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                colours[idx] = c
                coloured_set.add(idx)
            best_colours = colours
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        inst_start = time.time()
        inst_limit = min(inst_start + 45.0, deadline)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        if time.time() > deadline:
            break

    return results
# EVOLVE-BLOCK-END
