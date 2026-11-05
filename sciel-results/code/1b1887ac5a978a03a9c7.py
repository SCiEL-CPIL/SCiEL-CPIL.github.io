"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_conflicts(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]
        
        for i, (u, v) in enumerate(edges):
            # Edges sharing vertex u or v
            for nb in adj[u]:
                e2 = (min(u, nb), max(u, nb))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for nb in adj[v]:
                e2 = (min(v, nb), max(v, nb))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Edges joined by an edge: (u,w) where w in adj[u], w != v
            # Actually need: edge (a,b) conflicts with (c,d) if a=c or a=d or b=c or b=d
            # OR if (a,c) in E or (a,d) in E or (b,c) in E or (b,d) in E
            # The shared vertex case is handled above. Now handle "joined by an edge":
            # For edge (u,v), find all edges (x,y) such that (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E
            # and {x,y} & {u,v} == empty
            for nb in adj[u]:
                if nb == v:
                    continue
                for nb2 in adj[nb]:
                    if nb2 == u or nb2 == v:
                        continue
                    e2 = (min(nb, nb2), max(nb, nb2))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for nb in adj[v]:
                if nb == u:
                    continue
                for nb2 in adj[nb]:
                    if nb2 == u or nb2 == v:
                        continue
                    e2 = (min(nb, nb2), max(nb, nb2))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
        
        return conflicts

    def greedy_color(conflicts, n_edges, order):
        """Greedy colouring following the given order of edge indices."""
        colour = [-1] * n_edges
        used_colours = set()
        
        for idx in order:
            blocked = set()
            for j in conflicts[idx]:
                if colour[j] != -1:
                    blocked.add(colour[j])
            c = 0
            while c in blocked:
                c += 1
            colour[idx] = c
            used_colours.add(c)
        
        return colour, len(used_colours)

    def try_k_colors(conflicts, n_edges, k, order, time_limit):
        """
        Try to color with exactly k colors using backtracking with forward checking.
        Returns a valid coloring or None.
        """
        colour = [-1] * n_edges
        # For each color, track which edges have that color
        color_classes = [set() for _ in range(k)]
        
        # Domain: for each edge, the set of available colors
        domains = [list(range(k)) for _ in range(n_edges)]
        
        # Process edges in the given order
        def backtrack(pos):
            if time.time() > time_limit:
                return False
            if pos == n_edges:
                return True
            
            idx = order[pos]
            
            # Try each available color
            for c in domains[idx]:
                # Check if color c is available (no conflicting edge has color c)
                valid = True
                for j in conflicts[idx]:
                    if colour[j] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color c to edge idx
                colour[idx] = c
                color_classes[c].add(idx)
                
                # Forward checking: update domains of unassigned conflicting edges
                saved_domains = []
                consistent = True
                for j in conflicts[idx]:
                    if colour[j] == -1:  # unassigned
                        if c in domains[j]:
                            saved_domains.append((j, c, list(domains[j])))
                            domains[j].remove(c)
                            if not domains[j]:
                                consistent = False
                                break
                
                if consistent and backtrack(pos + 1):
                    return True
                
                # Undo
                colour[idx] = -1
                color_classes[c].remove(idx)
                for j, removed_c, old_domain in saved_domains:
                    domains[j] = old_domain
            
            return False
        
        return backtrack(0)

    def merge_color_classes(conflicts, n_edges, colour, k):
        """
        Try to merge color classes to reduce the number of colors used.
        Returns improved coloring and new k.
        """
        current_k = k
        improved = True
        while improved:
            improved = False
            # Try to merge each pair of color classes
            for c1 in range(current_k):
                if time.time() > deadline - 10:
                    return colour, current_k
                if not any(colour[i] == c1 for i in range(n_edges)):
                    continue
                for c2 in range(c1 + 1, current_k):
                    if not any(colour[i] == c2 for i in range(n_edges)):
                        continue
                    # Check if we can recolor all edges of c2 to c1
                    # i.e., no two edges in c1 ∪ c2 conflict
                    can_merge = True
                    edges_c1 = [i for i in range(n_edges) if colour[i] == c1]
                    edges_c2 = [i for i in range(n_edges) if colour[i] == c2]
                    for i in edges_c1:
                        for j in edges_c2:
                            if i in conflicts[j]:
                                can_merge = False
                                break
                        if not can_merge:
                            break
                    if can_merge:
                        for i in edges_c2:
                            colour[i] = c1
                        # Renumber colors to keep them contiguous
                        new_colour = [0] * n_edges
                        next_c = 0
                        for i in range(n_edges):
                            if colour[i] == c2:
                                new_colour[i] = c1
                            elif colour[i] > c2:
                                new_colour[i] = colour[i] - 1
                            else:
                                new_colour[i] = colour[i]
                        colour = new_colour
                        current_k -= 1
                        improved = True
                        break
                if improved:
                    break
        return colour, current_k

    def color_instance(n, d, seed):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflicts(edges, adj)
        n_edges = len(edges)
        
        if n_edges == 0:
            return []
        
        # Get a greedy coloring first to establish an upper bound
        rng = random.Random(42)
        
        # Try multiple greedy orderings to get a good upper bound
        best_greedy = None
        best_greedy_k = n_edges
        
        # Descending degree in conflict graph
        deg_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]), reverse=True)
        colour, k = greedy_color(conflicts, n_edges, deg_order)
        if k < best_greedy_k:
            best_greedy_k = k
            best_greedy = colour[:]
        
        # Ascending degree
        asc_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]))
        colour, k = greedy_color(conflicts, n_edges, asc_order)
        if k < best_greedy_k:
            best_greedy_k = k
            best_greedy = colour[:]
        
        # A few random orderings
        for _ in range(50):
            if time.time() > deadline - 30:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            colour, k = greedy_color(conflicts, n_edges, order)
            if k < best_greedy_k:
                best_greedy_k = k
                best_greedy = colour[:]
        
        # Now try to improve using k-color feasibility search
        # Start from a lower bound
        # Lower bound: max degree in conflict graph + 1
        max_conf_deg = max(len(conflicts[i]) for i in range(n_edges)) if n_edges > 0 else 0
        lower_bound = max_conf_deg + 1
        
        # Also use the Erdos-Nesetril bound as an upper bound hint
        en_bound = int(5 * d * d / 4) + 1
        
        # Try k from lower_bound up to best_greedy_k
        best_colour = best_greedy
        best_k = best_greedy_k
        
        # Try k-color search for k from lower_bound to best_greedy_k - 1
        for k_try in range(lower_bound, best_greedy_k):
            if time.time() > deadline - 20:
                break
            
            # Try a few different orderings for backtracking
            found = False
            for order_idx in range(3):
                if time.time() > deadline - 20:
                    break
                
                if order_idx == 0:
                    order = deg_order
                elif order_idx == 1:
                    order = asc_order
                else:
                    order = list(range(n_edges))
                    rng.shuffle(order)
                
                time_limit = deadline - 15
                result = try_k_colors(conflicts, n_edges, k_try, order, time_limit)
                if result is not None:
                    # Valid k-color coloring found
                    best_k = k_try
                    best_colour = result
                    found = True
                    break
            
            if found:
                # Try to merge to reduce further
                best_colour, best_k = merge_color_classes(conflicts, n_edges, best_colour, best_k)
                # If we reduced further, we could try even lower k, but let's keep it simple
                break
        
        return [best_colour[i] for i in range(n_edges)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 20:
            # Not enough time, use simple greedy for remaining
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            conflicts = build_conflicts(edges, adj)
            n_edges = len(edges)
            if n_edges == 0:
                out.append([])
                continue
            deg_order = sorted(range(n_edges), key=lambda i: len(conflicts[i]), reverse=True)
            colour, _ = greedy_color(conflicts, n_edges, deg_order)
            out.append([colour[i] for i in range(n_edges)])
        else:
            out.append(color_instance(*spec))
    
    return out
# EVOLVE-BLOCK-END
