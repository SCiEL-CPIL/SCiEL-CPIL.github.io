"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle_warnsdorff(adj, forbidden_edges, start=0):
    """
    Find a Hamiltonian cycle using DFS with Warnsdorff's heuristic.
    """
    n = P['n']
    visited = {start}
    path = [start]
    used_edges = set()
    
    def get_heuristic_score(node, visited, used_edges, forbidden_edges):
        """
        Count the number of valid neighbors for 'node' given current state.
        Lower score is preferred (Warnsdorff's rule).
        """
        count = 0
        for neighbor in adj[node]:
            if neighbor in visited:
                continue
            edge = frozenset((node, neighbor))
            if edge in used_edges or edge in forbidden_edges:
                continue
            count += 1
        return count

    def dfs(current):
        if len(path) == n:
            last = path[-1]
            edge = frozenset((last, start))
            if start in adj[last] and edge not in used_edges and edge not in forbidden_edges:
                return list(path)
            return None
        
        # Get candidates and sort by heuristic
        candidates = []
        for neighbor in adj[current]:
            if neighbor not in visited:
                edge = frozenset((current, neighbor))
                if edge not in used_edges and edge not in forbidden_edges:
                    # Calculate score: number of onward moves from neighbor
                    # Note: We haven't added 'neighbor' to visited yet, so we check neighbors of 'neighbor'
                    # excluding 'current' (which is now visited) and others.
                    # Actually, simpler: just count valid neighbors of 'neighbor' in the *next* state.
                    # To save time, we can approximate or just use the degree in the remaining graph.
                    
                    # Let's calculate the score for 'neighbor' assuming we move there.
                    # Neighbors of 'neighbor' that are not visited (excluding 'current' which becomes visited)
                    # and not forbidden/used.
                    score = 0
                    for n_neighbor in adj[neighbor]:
                        if n_neighbor == current:
                            continue
                        if n_neighbor in visited:
                            continue
                        edge_n = frozenset((neighbor, n_neighbor))
                        if edge_n in used_edges or edge_n in forbidden_edges:
                            continue
                        score += 1
                    candidates.append((score, neighbor))
        
        if not candidates:
            return None
            
        # Sort by score (ascending)
        candidates.sort(key=lambda x: x[0])
        
        for score, neighbor in candidates:
            edge = frozenset((current, neighbor))
            visited.add(neighbor)
            used_edges.add(edge)
            path.append(neighbor)
            
            result = dfs(neighbor)
            if result is not None:
                return result
                
            path.pop()
            used_edges.remove(edge)
            visited.remove(neighbor)
            
        return None

    return dfs(start)

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    cycles_found = []
    forbidden_edges = set()
    
    start_time = time.time()
    max_time = 55.0
    
    # We try to find 3 cycles.
    # If Warnsdorff fails on the first try for a specific cycle index, 
    # we might try different start vertices or restarts, but Warnsdorff is usually very strong.
    # To be safe, we allow a few restarts with different start nodes if the first fails.
    
    for i in range(3):
        if time.time() - start_time > max_time:
            break
            
        cycle = None
        # Try a few start nodes if the first one fails
        # In a symmetric graph, start node 0 is usually fine, but let's try 0, 1, 2...
        for start_node in range(n):
            if time.time() - start_time > max_time:
                break
            cycle = _find_cycle_warnsdorff(adj, forbidden_edges, start=start_node)
            if cycle is not None:
                break
        
        if cycle is None:
            # Failed to find the i-th cycle
            break
        
        # Add edges to forbidden
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            forbidden_edges.add(frozenset((u, v)))
            
        cycles_found.append(cycle)
        
        # If we found 3, we are done (max possible)
        if len(cycles_found) == 3:
            break

    # Ensure types are correct
    result = []
    for cycle in cycles_found:
        result.append([int(v) for v in cycle])
        
    return result

# EVOLVE-BLOCK-END
