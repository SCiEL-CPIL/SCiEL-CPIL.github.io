"""Strong edge colourings via improved local search with swap-based moves."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def greedy_color(colouring, conflict_adj, order, max_color=None):
        """Greedy color edges in given order, optionally limiting to max_color."""
        m = len(colouring)
        blocked = [set() for _ in range(m)]
        
        for idx in order:
            if max_color is None:
                c = 0
                while c in blocked[idx]:
                    c += 1
                colouring[idx] = c
            else:
                # Find smallest available color <= max_color
                found = False
                for c in range(max_color + 1):
                    if c not in blocked[idx]:
                        colouring[idx] = c
                        found = True
                        break
                if not found:
                    return False
            # Update blocked sets for neighbors
            for nb in conflict_adj[idx]:
                blocked[nb].add(colouring[idx])
        return True

    def try_remove_color(colouring, conflict_adj, c_remove, rng, m, max_iter=100):
        """Try to remove color c_remove by reassigning its edges to other colors."""
        edges_with_c = [i for i in range(m) if colouring[i] == c_remove]
        if not edges_with_c:
            return True
        
        old_colours = colouring[:]
        rng.shuffle(edges_with_c)
        
        for idx in edges_with_c:
            available = []
            for c_new in range(c_remove):
                valid = True
                for nb in conflict_adj[idx]:
                    if colouring[nb] == c_new:
                        valid = False
                        break
                if valid:
                    available.append(c_new)
            
            if not available:
                colouring[:] = old_colours
                return False
            else:
                colouring[idx] = rng.choice(available)
        
        return True

    def swap_local_search(colouring, conflict_adj, m, rng, time_limit, max_rounds=50):
        """Local search using swap moves: try to move an edge from a higher color to a lower color."""
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds and time.time() < time_limit:
            improved = False
            rounds += 1
            
            # Get current max color
            max_c = max(colouring) if m > 0 else -1
            if max_c <= 0:
                break
            
            # Try to reduce colors from high to low
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    return colouring
                
                edges_with_c = [i for i in range(m) if colouring[i] == c_remove]
                if not edges_with_c:
                    continue
                
                # Try to reassign these edges to lower colors
                old_colours = colouring[:]
                rng.shuffle(edges_with_c)
                
                success = True
                for idx in edges_with_c:
                    available = []
                    for c_new in range(c_remove):
                        valid = True
                        for nb in conflict_adj[idx]:
                            if colouring[nb] == c_new:
                                valid = False
                                break
                        if valid:
                            available.append(c_new)
                    
                    if not available:
                        success = False
                        break
                    else:
                        colouring[idx] = rng.choice(available)
                
                if success:
                    improved = True
                    break
                else:
                    colouring[:] = old_colours
            
            # If we couldn't remove a color, try swap moves
            if not improved and time.time() < time_limit:
                # Try random swap moves
                for _ in range(1000):
                    if time.time() > time_limit:
                        break
                    i = rng.randrange(m)
                    j = rng.randrange(m)
                    if i == j or colouring[i] == colouring[j]:
                        continue
                    if colouring[i] > colouring[j]:
                        i, j = j, i
                    # Try to swap: move i to j's color and j to i's color
                    # Actually, let's just try to move i to a lower color
                    c_i = colouring[i]
                    c_j = colouring[j]
                    if c_i <= c_j:
                        continue
                    # Can i take color c_j?
                    valid_i = True
                    for nb in conflict_adj[i]:
                        if colouring[nb] == c_j:
                            valid_i = False
                            break
                    if not valid_i:
                        continue
                    # Can j take color c_i?
                    valid_j = True
                    for nb in conflict_adj[j]:
                        if colouring[nb] == c_i:
                            valid_j = False
                            break
                    if not valid_j:
                        continue
                    # Do the swap
                    colouring[i], colouring[j] = c_j, c_i
                    improved = True
                    break
        
        return colouring

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        # Try multiple strategies
        strategies = [
            'conflict_degree',
            'random',
            'weighted'
        ]
        
        max_restarts = 30
        restart_count = 0
        
        while restart_count < max_restarts and time.time() < time_limit:
            strategy = strategies[restart_count % len(strategies)]
            
            if strategy == 'conflict_degree':
                # Sort by conflict degree (highest first)
                degrees = [len(conflict_adj[i]) for i in range(m)]
                order = list(range(m))
                order.sort(key=lambda i: -degrees[i])
            elif strategy == 'random':
                order = list(range(m))
                rng.shuffle(order)
            else:  # weighted
                order = list(range(m))
                weights = [len(conflict_adj[i]) + 1 for i in range(m)]
                rng.shuffle(order)
                # Weighted shuffle: edges with higher conflict degree more likely to be colored first
                # Simple approach: just shuffle but bias by sorting partially
                order = list(range(m))
                rng.shuffle(order)
            
            colours = [-1] * m
            
            # Greedy coloring
            if not greedy_color(colours, conflict_adj, order):
                # Fallback to simple greedy
                colours = [0] * m
                for i in range(m):
                    blocked = set()
                    for nb in conflict_adj[i]:
                        if colours[nb] != -1:
                            blocked.add(colours[nb])
                    c = 0
                    while c in blocked:
                        c += 1
                    colours[i] = c
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            # Local search
            colours = swap_local_search(colours, conflict_adj, m, rng, time_limit, max_rounds=20)
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
        
        if best_colours is None:
            # Fallback
            best_colours = [0] * m
            for i in range(m):
                blocked = set()
                for nb in conflict_adj[i]:
                    if best_colours[nb] != -1:
                        blocked.add(best_colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                best_colours[i] = c
        
        return best_colours

    results = []
    # Allocate time: more time to harder instances
    time_alloc = [(40, 4, 101, 40.0), (60, 5, 102, 50.0), (80, 6, 103, 55.0)]
    
    for n, d, seed, t_alloc in time_alloc:
        if time.time() > deadline - 5:
            # Not enough time, return best effort
            edges = instance(n, d, seed)
            m = len(edges)
            # Quick greedy
            conflict_adj = build_conflict_graph(edges, n)
            colours = [0] * m
            for i in range(m):
                blocked = set()
                for nb in conflict_adj[i]:
                    if colours[nb] != -1:
                        blocked.add(colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                colours[i] = c
            results.append(colours)
            continue
        
        edges = instance(n, d, seed)
        inst_start = time.time()
        inst_limit = min(inst_start + t_alloc, deadline - 2.0)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)

    return results
# EVOLVE-BLOCK-END
