"""Strong edge colourings via Maximum Induced Matching extraction + local refinement."""
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
                    e_i, e_j = inc_edges[i], inc_edges[j]
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

    def find_max_induced_matching(conflict_adj, m, time_limit, rng):
        """
        Find a maximum induced matching in the conflict graph using a heuristic 
        that prioritizes edges with low conflict degree, then refines with local search.
        """
        if m == 0:
            return []
        
        # Compute conflict degrees
        degs = [len(neigh) for neigh in conflict_adj]
        
        # Greedy selection: pick edge with min degree, remove it and all its neighbors
        selected = []
        used = [False] * m
        remaining = list(range(m))
        remaining.sort(key=lambda x: degs[x])
        
        # We'll use a priority queue approach for efficiency, but m is small enough for list
        # Actually, let's just iterate.
        # A better greedy: repeatedly pick the edge with the smallest number of remaining neighbors.
        
        # To speed up, we can use a simple heuristic:
        # 1. Sort by degree
        # 2. Pick first available, mark it and all its neighbors as used.
        
        # But this is not necessarily maximum. Let's try multiple random restarts of this greedy.
        
        best_match = []
        
        for restart in range(50):
            if time.time() > time_limit:
                break
            
            current_match = []
            used = [False] * m
            
            # Randomize the order slightly
            candidates = list(range(m))
            if restart == 0:
                candidates.sort(key=lambda x: degs[x])
            else:
                # Sort by degree + random noise
                candidates.sort(key=lambda x: degs[x] + rng.random() * 10)
            
            for e in candidates:
                if used[e]:
                    continue
                current_match.append(e)
                used[e] = True
                for nb in conflict_adj[e]:
                    used[nb] = True
            
            if len(current_match) > len(best_match):
                best_match = current_match[:]
        
        # Local search on the matching: try to swap edges to increase size
        # This is a maximum matching in a general graph, which is hard.
        # But induced matching is even harder.
        # Given time constraints, we'll stick with the greedy result.
        
        return best_match

    def greedy_color_with_order(order, conflict_adj, m):
        coloured_set = set()
        colours = [0] * m
        for idx in order:
            blocked = set()
            for nb in conflict_adj[idx]:
                if nb in coloured_set:
                    blocked.add(colours[nb])
            c = 0
            while c in blocked:
                c += 1
            colours[idx] = c
            coloured_set.add(idx)
        return colours

    def try_remove_color(colours, c_remove, conflict_adj, m, rng):
        edges_with_c = [i for i in range(m) if colours[i] == c_remove]
        if not edges_with_c:
            return True, colours[:]
        
        old_colours = colours[:]
        rng.shuffle(edges_with_c)
        
        for idx in edges_with_c:
            available = []
            for c_new in range(c_remove):
                valid = True
                for nb in conflict_adj[idx]:
                    if colours[nb] == c_new:
                        valid = False
                        break
                if valid:
                    available.append(c_new)
            if not available:
                colours = old_colours
                return False, old_colours
            colours[idx] = rng.choice(available)
        
        return True, colours

    def local_search_improve(colours, conflict_adj, m, rng, time_limit):
        improved = True
        iters = 0
        while improved and iters < 200 and time.time() < time_limit:
            improved = False
            iters += 1
            max_c = max(colours) if m > 0 else -1
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                success, new_colours = try_remove_color(colours, c_remove, conflict_adj, m, rng)
                if success:
                    colours = new_colours
                    improved = True
                    break
        return colours

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        # Strategy 1: Greedy + Local Search
        # Strategy 2: Maximum Induced Matching based coloring
        
        # We will interleave both strategies.
        
        # First, get a baseline from greedy
        order = list(range(m))
        order.sort(key=lambda x: len(conflict_adj[x]), reverse=True)
        colours = greedy_color_with_order(order, conflict_adj, m)
        colours = local_search_improve(colours, conflict_adj, m, rng, time_limit)
        current_count = max(colours) + 1 if m > 0 else 0
        best_colours = colours[:]
        best_count = current_count
        
        # Now try the induced matching approach
        # This is harder to implement as a direct coloring, but we can use it to guide the search.
        # Alternatively, we can use a different heuristic:
        # Color by repeatedly finding a maximum induced matching in the remaining graph.
        
        # Let's implement that.
        
        def coloring_by_max_induced_matching(conflict_adj, m, rng, time_limit):
            colours = [-1] * m
            used = [False] * m
            color_idx = 0
            
            while True:
                if time.time() > time_limit:
                    break
                
                # Find a maximum induced matching in the subgraph induced by unused edges
                # This is expensive. We'll approximate it.
                
                # Get list of unused edges
                unused = [i for i in range(m) if not used[i]]
                if not unused:
                    break
                
                # Build a local conflict graph for unused edges
                # Map original index to local index
                local_to_orig = {}
                orig_to_local = {}
                for li, oi in enumerate(unused):
                    local_to_orig[li] = oi
                    orig_to_local[oi] = li
                
                l_m = len(unused)
                if l_m == 0:
                    break
                
                # Build local conflict adj
                local_conflict = [set() for _ in range(l_m)]
                for li, oi in enumerate(unused):
                    for nb in conflict_adj[oi]:
                        if nb in orig_to_local:
                            local_conflict[li].add(orig_to_local[nb])
                
                # Find max induced matching in local_conflict
                # We'll use a simple greedy with random restarts
                best_local_match = []
                
                for restart in range(20):
                    if time.time() > time_limit:
                        break
                    
                    current_match = []
                    used_local = [False] * l_m
                    
                    candidates = list(range(l_m))
                    if restart == 0:
                        candidates.sort(key=lambda x: len(local_conflict[x]))
                    else:
                        candidates.sort(key=lambda x: len(local_conflict[x]) + rng.random() * 5)
                    
                    for e in candidates:
                        if used_local[e]:
                            continue
                        current_match.append(e)
                        used_local[e] = True
                        for nb in local_conflict[e]:
                            used_local[nb] = True
                    
                    if len(current_match) > len(best_local_match):
                        best_local_match = current_match[:]
                
                if not best_local_match:
                    # Fallback: just pick one edge
                    best_local_match = [0]
                
                # Assign color_idx to these edges
                for li in best_local_match:
                    oi = local_to_orig[li]
                    colours[oi] = color_idx
                    used[oi] = True
                
                color_idx += 1
            
            return colours
        
        # Try the induced matching coloring
        im_colours = coloring_by_max_induced_matching(conflict_adj, m, rng, time_limit)
        if -1 not in im_colours:
            im_colours = local_search_improve(im_colours, conflict_adj, m, rng, time_limit)
            im_count = max(im_colours) + 1 if m > 0 else 0
            if im_count < best_count:
                best_count = im_count
                best_colours = im_colours[:]
        
        # Try a few more random restarts of greedy + local search
        for restart in range(10):
            if time.time() > time_limit:
                break
            
            order = list(range(m))
            rng.shuffle(order)
            colours = greedy_color_with_order(order, conflict_adj, m)
            colours = local_search_improve(colours, conflict_adj, m, rng, time_limit)
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
        
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
