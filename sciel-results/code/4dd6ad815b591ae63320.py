import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_cycles_containing_edge(adj, edge, max_cycles=50, min_len=3):
        """
        Find cycles in the graph (defined by adj) that contain the specific edge.
        Uses DFS from one endpoint to the other, avoiding the edge itself initially,
        then closing the loop.
        """
        u, v = edge
        cycles = []
        
        # To find cycles containing (u,v), we can look for paths from u to v 
        # that do not use the edge (u,v) directly, and have length >= 2 (so total cycle len >= 3).
        # Actually, a cycle containing (u,v) is (u, v, ..., u).
        # So we need a path from v to u of length >= 2 in the graph with (u,v) removed.
        
        # Create a copy of adj without the edge (u,v) for the search
        adj_no_edge = {k: set(v) for k, v in adj.items()}
        if v in adj_no_edge[u]:
            adj_no_edge[u].remove(v)
        if u in adj_no_edge[v]:
            adj_no_edge[v].remove(u)
            
        # DFS from v to u
        stack = [(v, [v])]
        visited = {v}
        
        while stack:
            node, path = stack.pop()
            if node == u and len(path) >= 2:
                # Found a path v -> ... -> u
                # Cycle is u -> v -> ... -> u
                # The path is [v, ..., u]. 
                # Cycle vertices: u, v, ..., u (last is u, first is u? No)
                # Path: v, n1, ..., u.
                # Cycle: u, v, n1, ..., u.
                # Vertices: [u] + path
                cycle = [u] + path
                # Ensure simple cycle (path is simple by DFS)
                cycles.append(cycle)
                if len(cycles) >= max_cycles:
                    break
            else:
                for neighbor in adj_no_edge[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, path + [neighbor]))
                        # Prune if path is too long? 
                        # We want long cycles, but for B&B we might want variety.
                        # Limit depth to prevent explosion
                        if len(path) > 20: 
                            break
                            
        # Sort cycles by length descending (longer cycles save more pieces)
        cycles.sort(key=lambda c: len(c), reverse=True)
        return cycles

    def decompose_bb(edges, time_limit=10.0):
        """
        Branch and Bound to minimize number of pieces.
        """
        if not edges:
            return []
        
        # State: set of remaining edges
        # We use a recursive search with memoization if possible, but state space is large.
        # We use a priority queue (Best-First Search) or DFS with pruning.
        # Given the size, DFS with pruning is likely sufficient.
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_pieces = None
        best_count = float('inf')
        start_search = time.time()
        
        # Pre-sort edges to branch on "important" edges first (high degree)
        edge_list = list(edges)
        edge_list.sort(key=lambda e: len(adj[e[0]]) + len(adj[e[1]]), reverse=True)
        
        def solve(remaining_edges_set, current_pieces, current_count):
            nonlocal best_pieces, best_count, start_search
            
            if time.time() - start_search > time_limit:
                return
                
            if not remaining_edges_set:
                if current_count < best_count:
                    best_count = current_count
                    best_pieces = list(current_pieces)
                return
            
            # Lower Bound Pruning
            # Each piece covers at most 3 edges? No, can be more.
            # But a single edge covers 1. A triangle covers 3.
            # A safe lower bound: ceil(len(remaining) / 3) is too weak if no triangles.
            # Better: The number of connected components with odd number of edges? No.
            # Let's use: current_count + ceil(len(remaining_edges_set) / 3) 
            # This assumes max 3 edges per piece on average.
            lb = current_count + (len(remaining_edges_set) + 2) // 3
            if lb >= best_count:
                return

            # Pick an edge to branch on
            # Heuristic: pick the edge with the highest degree in the remaining graph
            # to constrain the search early.
            rem_adj = defaultdict(set)
            for u, v in remaining_edges_set:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                
            # Find edge with max degree sum
            max_deg = -1
            chosen_edge = None
            for e in remaining_edges_set:
                deg = len(rem_adj[e[0]]) + len(rem_adj[e[1]])
                if deg > max_deg:
                    max_deg = deg
                    chosen_edge = e
                    
            if chosen_edge is None:
                return

            u, v = chosen_edge
            
            # Option 1: chosen_edge is a single piece
            new_rem = set(remaining_edges_set)
            new_rem.remove(chosen_edge)
            new_pieces = current_pieces + [[u, v]]
            solve(new_rem, new_pieces, current_count + 1)
            
            # Option 2: chosen_edge is part of a cycle
            # Find cycles containing chosen_edge in the remaining graph
            cycles = get_cycles_containing_edge(rem_adj, chosen_edge, max_cycles=10)
            
            # Try the best (longest) cycles first
            for cycle in cycles:
                if time.time() - start_search > time_limit:
                    return
                    
                # Check if cycle edges are all in remaining_edges_set
                valid = True
                cycle_edges = []
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    if e not in remaining_edges_set:
                        valid = False
                        break
                    cycle_edges.append(e)
                    
                if not valid:
                    continue
                
                new_rem = set(remaining_edges_set)
                for e in cycle_edges:
                    new_rem.remove(e)
                
                new_pieces = current_pieces + [list(cycle)]
                solve(new_rem, new_pieces, current_count + 1)

        solve(set(edges), [], 0)
        
        if best_pieces is None:
            # Fallback: all single edges
            return [[u, v] for u, v in edges]
            
        return best_pieces

    def decompose_greedy_with_ls(edges, time_limit=5.0):
        """
        Greedy decomposition followed by Local Search.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        current_adj = {u: set(v) for u, v in adj.items()}
        pieces = []
        
        while True:
            if time.time() - start_time > DEADLINE:
                break
            if not any(len(neigh) > 0 for neigh in current_adj.values()):
                break
                
            # Find longest cycle
            best_cycle = None
            best_len = 0
            vertices = [u for u in current_adj if current_adj[u]]
            
            for start in vertices[:5]: # Limit starts
                # DFS for longest path/cycle
                # Simple heuristic: try to close a cycle
                stack = [(start, [start], {start})]
                while stack:
                    node, path, visited = stack.pop()
                    # Check if we can close cycle
                    if len(path) >= 3 and start in current_adj[node]:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                    for neighbor in current_adj[node]:
                        if neighbor not in visited:
                            new_visited = set(visited)
                            new_visited.add(neighbor)
                            stack.append((neighbor, path + [neighbor], new_visited))
                            if len(path) > 15: # Limit depth
                                break
            if best_cycle is None:
                # No cycle found, take a single edge
                # Pick edge with lowest degree to preserve high-degree nodes for cycles?
                # Or highest? Let's pick arbitrary or lowest to break components.
                rem_edges = []
                for u in current_adj:
                    for v in current_adj[u]:
                        if u < v:
                            rem_edges.append((u, v))
                if not rem_edges:
                    break
                e = rem_edges[0]
                pieces.append([e[0], e[1]])
                current_adj[e[0]].remove(e[1])
                current_adj[e[1]].remove(e[0])
            else:
                pieces.append(list(best_cycle))
                for i in range(len(best_cycle)):
                    a, b = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    if b in current_adj[a]:
                        current_adj[a].remove(b)
                    if a in current_adj[b]:
                        current_adj[b].remove(a)
                        
        # Add remaining edges
        for u in current_adj:
            for v in current_adj[u]:
                if u < v:
                    pieces.append([u, v])
                    # Remove to avoid duplicates if we iterate both ways, 
                    # but we just append. We need to clean up.
        # Clean up remaining edges properly
        final_pieces = []
        remaining = set(edges)
        for p in pieces:
            if len(p) == 2:
                e = (min(p[0], p[1]), max(p[0], p[1]))
                if e in remaining:
                    final_pieces.append(p)
                    remaining.remove(e)
            else:
                # Verify cycle edges are in remaining
                valid = True
                for i in range(len(p)):
                    a, b = p[i], p[(i + 1) % len(p)]
                    e = (min(a, b), max(a, b))
                    if e not in remaining:
                        valid = False
                        break
                if valid:
                    final_pieces.append(p)
                    for i in range(len(p)):
                        a, b = p[i], p[(i + 1) % len(p)]
                        e = (min(a, b), max(a, b))
                        remaining.remove(e)
        
        # Add any truly remaining edges
        for e in remaining:
            final_pieces.append([e[0], e[1]])
            
        return final_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        
        # Try B&B first with a time limit
        # If it finds a solution quickly, great.
        # Otherwise, fall back to Greedy.
        
        # Allocate time budget
        # 3 graphs. ~40s each.
        t_start = time.time()
        bb_result = decompose_bb(edges, time_limit=30.0)
        t_end = time.time()
        
        # Also run Greedy to compare
        greedy_result = decompose_greedy_with_ls(edges, time_limit=5.0)
        
        # Choose the better one
        if len(bb_result) < len(greedy_result):
            results.append(bb_result)
        else:
            results.append(greedy_result)
            
        # Ensure we don't exceed total deadline
        if time.time() - start_time > DEADLINE:
             # If we are out of time, just return whatever we have or simple edges
             # But we should have finished.
             pass

    return results
