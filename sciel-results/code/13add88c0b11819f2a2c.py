"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict, deque

_DEADLINE_SECONDS = 148.0


def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_greedy(edges, n, time_limit):
    """
    Decompose graph by repeatedly finding long cycles and removing them.
    Uses iterative DFS to avoid recursion limits.
    """
    if not edges:
        return []
    
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    start_time = time.time()
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        if time.time() - start_time > time_limit:
            break
        
        # Find a cycle in the current graph
        # Try to find a long cycle for better savings
        best_cycle = None
        best_len = 0
        
        # Try finding cycles from each vertex
        for start in range(n):
            if time.time() - start_time > time_limit:
                break
            if not adj[start] or len(adj[start]) < 2:
                continue
            
            # Iterative DFS to find a cycle starting and ending at 'start'
            # where 'start' is the smallest vertex in the cycle
            found = _find_one_cycle_iterative(adj, start, n, remaining_edges)
            if found and len(found) > best_len:
                best_len = len(found)
                best_cycle = found
                if best_len >= n:  # Can't do better than Hamiltonian
                    break
        
        if best_cycle:
            # Verify all edges in cycle still exist
            valid = True
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    valid = False
                    break
            
            if valid:
                pieces.append(best_cycle)
                # Remove cycle edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                # Cycle has invalid edges, try next
                continue
        else:
            # No cycle found, take a single edge
            edge = next(iter(remaining_edges))
            remaining_edges.discard(edge)
            u, v = edge
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
    
    # Add any remaining edges
    for edge in remaining_edges:
        pieces.append(list(edge))
    
    return pieces


def _find_one_cycle_iterative(adj, start, n, remaining_edges):
    """
    Find one simple cycle containing 'start' where start is the smallest vertex.
    Uses iterative DFS to avoid recursion limits.
    Returns the cycle as a list of vertices, or None.
    """
    if not adj[start] or len(adj[start]) < 2:
        return None
    
    # We want to find a cycle that returns to 'start'
    # Use iterative DFS with explicit stack
    # Stack entries: (node, iterator_state)
    # We track the path
    
    path = [start]
    on_path = {start}
    
    # For each node, we need to track which neighbors we've tried
    # Use a dict: node -> list of neighbors to try, and an index into that list
    neighbor_lists = {}
    for u in range(n):
        if u in adj:
            # Filter neighbors: must be >= start, and edge must be in remaining_edges
            valid_neighbors = []
            for v in sorted(adj[u]):
                if v < start:
                    continue
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    continue
                valid_neighbors.append(v)
            neighbor_lists[u] = valid_neighbors
    
    if not neighbor_lists.get(start, []):
        return None
    
    # Iterative DFS
    # Stack of (node, next_neighbor_index)
    stack = [(start, 0)]
    
    while stack:
        node, next_idx = stack[-1]
        
        if next_idx >= len(neighbor_lists.get(node, [])):
            # No more neighbors to try, backtrack
            stack.pop()
            if len(stack) > 0:
                # Remove node from path
                path.pop()
                on_path.discard(node)
            continue
        
        # Get the next neighbor to try
        v = neighbor_lists[node][next_idx]
        stack[-1] = (node, next_idx + 1)  # Update next index
        
        if v == start:
            # Found a cycle!
            if len(path) >= 3:
                return path[:]
            continue
        
        if v not in on_path:
            on_path.add(v)
            path.append(v)
            stack.append((v, 0))
    
    return None


def _decompose_eulerian(edges, n):
    """
    Decompose an undirected graph into cycles and edges using the Eulerian trail approach.
    """
    if not edges:
        return []

    # Build adjacency for the directed Eulerian multigraph
    adj = defaultdict(list)
    edge_id_counter = 0
    
    for u, v in edges:
        adj[u].append((v, edge_id_counter))
        edge_id_counter += 1
        adj[v].append((u, edge_id_counter))
        edge_id_counter += 1

    # Find connected components in the undirected version
    undirected_adj = defaultdict(set)
    for u, v in edges:
        undirected_adj[u].add(v)
        undirected_adj[v].add(u)
    
    visited_nodes = set()
    components = []
    for node in range(n):
        if node in visited_nodes or not undirected_adj[node]:
            continue
        comp = set()
        stack = [node]
        visited_nodes.add(node)
        while stack:
            curr = stack.pop()
            comp.add(curr)
            for neighbor in undirected_adj[curr]:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)
    
    all_cycles = []
    
    for comp in components:
        if not comp:
            continue
        
        start = None
        for node in sorted(comp):
            if adj[node]:
                start = node
                break
        if start is None:
            continue
        
        comp_arcs_used = set()
        
        def hierholzer(start_node):
            stack = [start_node]
            path = []
            
            while stack:
                node = stack[-1]
                found = False
                while adj[node]:
                    v, arc_id = adj[node].pop()
                    if arc_id not in comp_arcs_used:
                        comp_arcs_used.add(arc_id)
                        stack.append(v)
                        found = True
                        break
                if not found:
                    path.append(stack.pop())
            
            path.reverse()
            return path
        
        euler_path = hierholzer(start)
        
        if len(euler_path) < 2:
            continue
            
        remaining_path = list(euler_path)
        comp_cycles = []
        
        while len(remaining_path) > 1:
            pos = {}
            cycle_found = None
            for i in range(len(remaining_path)):
                v = remaining_path[i]
                if v in pos:
                    start_idx = pos[v]
                    end_idx = i
                    cycle = remaining_path[start_idx:end_idx+1]
                    if len(cycle) >= 3:
                        inner = cycle[1:-1]
                        if len(set(inner)) == len(inner) and cycle[0] not in inner:
                            comp_cycles.append(cycle)
                            remaining_path = remaining_path[:start_idx] + remaining_path[end_idx:]
                            break
                else:
                    pos[v] = i
            
            if not cycle_found and len(remaining_path) <= 1:
                break
            if not cycle_found:
                if len(remaining_path) <= 1:
                    break
                break
        
        all_cycles.extend(comp_cycles)
    
    pieces = []
    used_edges = set()
    
    for cycle in all_cycles:
        if len(cycle) < 3:
            continue
        if cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        
        if len(cycle) < 3:
            continue
            
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e not in set((min(a,b), max(a,b)) for a,b in edges):
                valid = False
                break
            if e in used_edges:
                valid = False
                break
        
        if valid:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            pieces.append(cycle)
    
    all_edges_set = set((min(u, v), max(u, v)) for u, v in edges)
    for e in all_edges_set:
        if e not in used_edges:
            pieces.append(list(e))
            
    return pieces


def construct():
    """
    Decompose three fixed graphs into cycles and edges, minimizing piece count.
    Returns three lists of pieces.
    """
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)

        graph_time = min(remaining_time * 0.9, 40.0)
        
        pieces = _decompose_greedy(edges, n, graph_time)
        
        used_edges = set()
        valid = True
        for piece in pieces:
            if len(piece) == 2:
                u, v = piece
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
            
        if not valid or used_edges != edges_set:
            pieces = _decompose_eulerian(edges, n)
            used_edges = set()
            valid = True
            for piece in pieces:
                if len(piece) == 2:
                    u, v = piece
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                elif len(piece) >= 3:
                    for i in range(len(piece)):
                        u = piece[i]
                        v = piece[(i + 1) % len(piece)]
                        e = (min(u, v), max(u, v))
                        if e in used_edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break
            
            if not valid or used_edges != edges_set:
                pieces = [[u, v] for u, v in edges]

        results.append(pieces)

    return results
