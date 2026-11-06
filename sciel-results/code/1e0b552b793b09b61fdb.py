"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import defaultdict

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycles_in_graph(adj, deadline):
    """
    Find a set of edge-disjoint cycles in the graph given by adjacency lists.
    Uses a heuristic: repeatedly find the longest cycle possible from a random start,
    remove its edges, and repeat.
    """
    cycles = []
    # Work on a copy of the adjacency structure (sets of neighbors)
    # We need to track which edges are still available.
    # Since the graph is small (n<=30), we can represent the graph as a set of edges.
    # Let's reconstruct the edge set from adj for easy removal.
    edges = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
            else:
                edges.add((v, u))
    
    # We need a way to quickly check if an edge exists and remove it.
    # Also, we need to update adj.
    # Let's use a dictionary of sets for adj, and a set for edges.
    
    def remove_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            edges.discard((u, v))
            adj[u].discard(v)
            adj[v].discard(u)
    
    while edges:
        if time.time() > deadline:
            break
        
        # Try to find a cycle starting from each node with degree >= 2
        # Heuristic: start from a node with high degree
        start_nodes = [u for u in adj if len(adj[u]) >= 2]
        if not start_nodes:
            break
        
        # Sort by degree descending to find longer cycles faster?
        start_nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        found_cycle = False
        for start in start_nodes:
            if len(adj[start]) < 2:
                continue
            # DFS to find a cycle back to start
            # Stack: (current_node, path)
            stack = [(start, [start])]
            visited = {start: 0} # node -> index in path? No, just set of visited in this DFS branch?
            # Actually, for simple cycle finding, we can just do DFS and track path.
            # Since n is small, we can be a bit inefficient.
            
            # We want the LONGEST cycle.
            # Let's do a DFS that tries to extend the path as much as possible.
            # This is essentially finding a simple path from start to start.
            
            best_cycle = []
            
            def dfs(node, path):
                nonlocal best_cycle
                if time.time() > deadline:
                    return
                if len(path) > len(best_cycle):
                    if node == start and len(path) >= 4: # Cycle length >= 3 means path length >= 4 (start...start)
                        # Check if edge (last, start) exists
                        last = path[-2]
                        if (min(last, start), max(last, start)) in edges:
                            best_cycle = path[:]
                            return # Found a cycle, but maybe a longer one? 
                            # To find the longest, we should continue searching?
                            # But that's expensive. Let's just return the first one found via DFS which tends to be decent?
                            # Or continue to see if a longer one exists.
                            # Given the small size, let's try to continue if we have time.
                            # But DFS backtracking can be exponential.
                            # Let's just return the first cycle found of length > best_cycle.
                            # Actually, let's just return the first valid cycle found.
                            # To maximize length, we should prioritize neighbors that have high degree or are unvisited?
                            # Simple DFS order might be okay.
                            pass
                            
                for neighbor in adj[node]:
                    if neighbor == path[0] and len(path) >= 3:
                        # Found a cycle
                        cycle = path[:]
                        if len(cycle) > len(best_cycle):
                            best_cycle = cycle
                            # Don't return immediately, try to find longer?
                            # For now, let's just record it.
                            continue
                    if neighbor not in path:
                        # Check if edge exists (it should, since it's in adj)
                        dfs(neighbor, path + [neighbor])
                        if len(best_cycle) > 0:
                            # If we found a cycle, we might stop early to save time?
                            # But we want the longest.
                            # Let's continue searching.
                            pass
            
            # To avoid exponential blowup, limit the depth or just use a simpler heuristic.
            # Given n<=30, a simple DFS for the first cycle is fast.
            # Let's just find ANY cycle starting from `start`.
            # We can use a standard cycle finding algorithm.
            
            # Let's restart with a simpler approach:
            # Find a cycle using DFS.
            # To find a LONGER cycle, we can try to start from different nodes or use a heuristic.
            # For now, let's just find the first cycle found by DFS from `start`.
            
            # Standard DFS for cycle
            def find_cycle(start_node):
                # Returns a list of nodes in the cycle, or None
                visited = set()
                path = []
                parent = {}
                
                def dfs(u):
                    visited.add(u)
                    path.append(u)
                    for v in adj[u]:
                        if v == start_node and len(path) >= 2:
                            # Found cycle
                            return path[:]
                        if v not in visited:
                            res = dfs(v)
                            if res:
                                return res
                    path.pop()
                    return None
                
                return dfs(start_node)
            
            cycle = find_cycle(start)
            if cycle:
                # Verify the cycle edges exist
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if (min(u, v), max(u, v)) not in edges:
                        valid = False
                        break
                if valid and len(cycle) >= 3:
                    cycles.append(cycle)
                    # Remove edges
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        remove_edge(u, v)
                    found_cycle = True
                    break # Try next start node? Or restart the whole loop?
                    # Restarting the whole loop ensures we pick the best start node next time.
        
        if not found_cycle:
            break
            
    return cycles

def _decompose(edges_list):
    if not edges_list:
        return []
    
    adj = defaultdict(set)
    for u, v in edges_list:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 10 # 10 seconds per graph
    
    cycles = _find_cycles_in_graph(adj, deadline)
    
    # Reconstruct remaining edges
    used_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            used_edges.add((min(u, v), max(u, v)))
    
    remaining_edges = []
    for e in edges_list:
        if e not in used_edges:
            remaining_edges.append(list(e))
    
    return cycles + remaining_edges

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_deadline = start_time + 120 # 120s limit, keep margin
    
    for spec in specs:
        if time.time() > total_deadline - 10:
            # Not enough time, just return all edges as singletons for remaining
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = _instance(*spec)
        pieces = _decompose(edges)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
