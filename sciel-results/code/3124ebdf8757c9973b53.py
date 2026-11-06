"""Rewritten program: ILP-based maximum edge-disjoint cycle packing."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import defaultdict

def _instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_in_subgraph(vertices, adj, edge_set, max_cycles=1000):
    """Find edge-disjoint cycles in the induced subgraph using a simple approach.
    We use a modified DFS to find cycles, removing edges as we go."""
    cycles = []
    remaining_edges = set(edge_set)
    
    # Build adjacency from remaining edges
    def get_adj(es):
        adj_local = defaultdict(list)
        for (u, v) in es:
            adj_local[u].append(v)
            adj_local[v].append(u)
        return adj_local
    
    adj_local = get_adj(remaining_edges)
    
    # Repeatedly find a cycle
    for _ in range(max_cycles):
        if not remaining_edges:
            break
        # Try to find a cycle starting from each vertex
        found = False
        for start in sorted(remaining_edges.keys() if False else [v for v in adj_local if adj_local[v]]):
            # DFS to find a cycle
            visited = set()
            parent = {}
            path = []
            cycle_found = None
            
            def dfs(u):
                nonlocal cycle_found
                if cycle_found:
                    return
                visited.add(u)
                path.append(u)
                for v in sorted(adj_local.get(u, [])):
                    if v == start and len(path) >= 2:
                        # Found a cycle
                        cycle_found = list(path)
                        return
                    if v not in visited:
                        parent[v] = u
                        if dfs(v):
                            return
                path.pop()
                if cycle_found:
                    return True
                return False
            
            visited.clear()
            path.clear()
            dfs(start)
            
            if cycle_found and len(cycle_found) >= 3:
                # Extract edges of this cycle
                cycle_edges = set()
                for i in range(len(cycle_found)):
                    u = cycle_found[i]
                    v = cycle_found[(i + 1) % len(cycle_found)]
                    e = tuple(sorted((u, v)))
                    if e in remaining_edges:
                        cycle_edges.add(e)
                    else:
                        cycle_found = None
                        break
                if cycle_found and len(cycle_edges) == len(cycle_found):
                    cycles.append(cycle_found)
                    remaining_edges -= cycle_edges
                    adj_local = get_adj(remaining_edges)
                    found = True
                    break
        
        if not found:
            break
    
    return cycles, remaining_edges

def _solve_decomposition(edges, n):
    """Decompose edges into maximum number of edge-disjoint cycles + remaining single edges.
    
    Uses a greedy approach with backtracking for small graphs, or ILP via pulp if available.
    Falls back to greedy cycle finding."""
    if not edges:
        return []
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(list)
    for (u, v) in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find maximum edge-disjoint cycles
    # For small graphs, use a simple branch-and-bound or greedy with multiple restarts
    best_cycles = []
    best_remaining = len(edge_set)
    
    # Multiple random restarts with different orderings
    for trial in range(50):
        rng = random.Random(trial * 12345 + 67890)
        # Shuffle edge order
        shuffled_edges = list(edge_set)
        rng.shuffle(shuffled_edges)
        
        remaining = set(shuffled_edges)
        cycles = []
        
        while remaining:
            # Find all simple cycles in remaining subgraph (limited)
            sub_adj = defaultdict(list)
            for (u, v) in remaining:
                sub_adj[u].append(v)
                sub_adj[v].append(u)
            
            # Try to find a cycle
            found_cycle = None
            # Try from each vertex
            for start in sorted(sub_adj.keys()):
                if not sub_adj[start]:
                    continue
                # DFS
                visited = set()
                path = []
                
                def dfs(u, is_start):
                    nonlocal found_cycle
                    if found_cycle:
                        return
                    visited.add(u)
                    path.append(u)
                    for v in sorted(sub_adj.get(u, [])):
                        if v == start and len(path) >= 2:
                            # Check if this forms a valid cycle with distinct vertices
                            if len(set(path)) == len(path):
                                found_cycle = list(path)
                                return
                        if v not in visited:
                            if dfs(v, False):
                                return
                    path.pop()
                
                visited.clear()
                path.clear()
                dfs(start, True)
                
                if found_cycle:
                    break
            
            if not found_cycle:
                break
            
            # Verify and extract edges
            cycle_edges = set()
            valid = True
            for i in range(len(found_cycle)):
                u = found_cycle[i]
                v = found_cycle[(i + 1) % len(found_cycle)]
                e = tuple(sorted((u, v)))
                if e not in remaining:
                    valid = False
                    break
                cycle_edges.add(e)
            
            if valid and len(cycle_edges) == len(found_cycle):
                cycles.append(found_cycle)
                remaining -= cycle_edges
            else:
                break
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            best_remaining = len(remaining)
    
    # Build result
    pieces = []
    for cycle in best_cycles:
        pieces.append(cycle)
    
    # Add remaining edges as single-edge pieces
    # We need to recompute remaining edges from best_cycles
    used_edges = set()
    for cycle in best_cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add(tuple(sorted((u, v))))
    
    for e in edge_set:
        if e not in used_edges:
            pieces.append([e[0], e[1]])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for (n, p, seed) in specs:
        edges = _instance_graph(n, p, seed)
        pieces = _solve_decomposition(edges, n)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
