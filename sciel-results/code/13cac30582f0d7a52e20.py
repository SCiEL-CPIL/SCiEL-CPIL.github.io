"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def decompose_graph(edges, time_limit):
        """
        Decompose edges into minimum pieces (cycles + single edges).
        Strategy: 
        1. Repeatedly find long cycles using DFS with heuristics.
        2. Use edge-coloring / matching ideas to find disjoint cycles.
        3. Any remaining edges become single-edge pieces.
        """
        remaining = set()
        for e in edges:
            ek = edge_key(e[0], e[1])
            remaining.add(ek)
        
        if not remaining:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        used_edges = set()
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                used_edges.add(ek)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
        
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            if ek in used_edges:
                return
            used_edges.add(ek)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
        
        # Find cycles using DFS with multiple strategies
        def find_cycle_dfs(start, max_depth=20):
            """Find a cycle starting and ending at start using DFS."""
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                if time.time() > time_limit:
                    return None
                if depth >= max_depth:
                    return None
                for nb in sorted(adj[current], key=lambda x: -len(adj[x])):
                    if nb == start and len(path) >= 3:
                        return path[:]
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        result = dfs(nb, depth + 1)
                        if result is not None:
                            return result
                        path.pop()
                        visited.discard(nb)
                return None
            
            return dfs(start, 0)
        
        def find_cycle_greedy(start):
            """Find a cycle by greedily following high-degree neighbors."""
            path = [start]
            visited = {start}
            current = start
            
            for _ in range(30):
                if time.time() > time_limit:
                    return None
                neighbors = [nb for nb in adj[current] if nb not in visited or nb == start]
                if not neighbors:
                    return None
                # Prefer neighbors with high degree
                neighbors.sort(key=lambda x: -len(adj[x]))
                next_v = neighbors[0]
                if next_v == start and len(path) >= 3:
                    return path[:]
                if next_v in visited and next_v != start:
                    return None
                visited.add(next_v)
                path.append(next_v)
                current = next_v
            return None
        
        def find_longest_cycle(start, max_depth=15):
            """Try to find a longer cycle using backtracking with pruning."""
            best = None
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                nonlocal best
                if time.time() > time_limit:
                    return
                if depth >= max_depth:
                    if current == start and len(path) >= 3:
                        if best is None or len(path) > len(best):
                            best = path[:]
                    return
                if current == start and len(path) >= 3:
                    if best is None or len(path) > len(best):
                        best = path[:]
                for nb in sorted(adj[current], key=lambda x: -len(adj[x])):
                    if nb == start and len(path) >= 3:
                        if best is None or len(path) + 1 > len(best):
                            best = path[:]
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.discard(nb)
            
            dfs(start, 0)
            return best
        
        # Main loop: repeatedly extract cycles
        progress = True
        while progress and remaining:
            if time.time() > time_limit:
                break
            progress = False
            
            best_cycle = None
            best_len = 0
            
            # Try multiple starting vertices
            candidates = [v for v in adj if len(adj[v]) >= 2]
            random.shuffle(candidates)
            
            for start in candidates[:30]:
                if time.time() > time_limit:
                    break
                
                # Try different cycle-finding strategies
                for strategy in [find_longest_cycle, find_cycle_dfs, find_cycle_greedy]:
                    cycle = strategy(start)
                    if cycle is not None and len(cycle) > best_len:
                        best_cycle = cycle
                        best_len = len(cycle)
                        if best_len >= 10:
                            break
            
            if best_cycle is not None:
                commit_cycle(best_cycle)
                progress = True
            else:
                # No more cycles found, take remaining edges as singles
                break
        
        # Remaining edges become single-edge pieces
        for u, v in list(remaining):
            if (u, v) not in used_edges:
                commit_single_edge(u, v)
        
        return pieces
    
    def decompose_with_improvement(edges, time_limit):
        """
        Decompose and then try to improve by replacing multiple single edges
        with a cycle if possible.
        """
        pieces = decompose_graph(edges, time_limit)
        
        # Try to improve: look for sets of single edges that form a cycle
        # and replace them with a single cycle piece
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            
            # Build a graph from single-edge pieces
            single_edges = []
            cycle_pieces = []
            for p in pieces:
                if len(p) == 2:
                    single_edges.append((p[0], p[1]))
                else:
                    cycle_pieces.append(p)
            
            if len(single_edges) < 3:
                break
            
            # Build adjacency from single edges
            adj_single = defaultdict(set)
            for u, v in single_edges:
                adj_single[u].add(v)
                adj_single[v].add(u)
            
            # Try to find a cycle in the single-edge graph
            found_cycle = None
            for start in adj_single:
                if len(adj_single[start]) < 2:
                    continue
                cycle = find_cycle_dfs(start, max_depth=10)
                if cycle is not None:
                    found_cycle = cycle
                    break
            
            if found_cycle is not None:
                # Remove the edges in the cycle from single_edges
                cycle_edges = set()
                for i in range(len(found_cycle)):
                    u = found_cycle[i]
                    v = found_cycle[(i + 1) % len(found_cycle)]
                    cycle_edges.add(edge_key(u, v))
                
                new_single_edges = []
                for u, v in single_edges:
                    if edge_key(u, v) not in cycle_edges:
                        new_single_edges.append((u, v))
                
                # Replace pieces
                pieces = cycle_pieces + [[list(e) for e in new_single_edges]] + [found_cycle]
                # Actually, we need to keep them as separate pieces
                pieces = cycle_pieces + [found_cycle]
                for u, v in new_single_edges:
                    pieces.append([u, v])
                improved = True
        
        return pieces
    
    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        time_for_this = min(35, deadline - time.time() - 5)
        if time_for_this < 3:
            time_for_this = 3
        
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            pieces = decompose_with_improvement(edges, time.time() + 3)
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
