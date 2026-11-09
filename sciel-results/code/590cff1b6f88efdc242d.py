"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(adj, nodes):
    """Check if the subgraph induced by 'nodes' is connected."""
    if not nodes:
        return True
    nodes_set = set(nodes)
    start = next(iter(nodes_set))
    visited = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in nodes_set and v not in visited:
                visited.add(v)
                stack.append(v)
    return len(visited) == len(nodes_set)

def _find_hamiltonian_cycle(adj, n, start, used_edges, deadline):
    """Find a Hamiltonian cycle starting from 'start' avoiding used_edges."""
    # Build available adjacency
    avail_adj = defaultdict(set)
    for u in range(n):
        for v in adj[u]:
            ek = _edge_key(u, v)
            if ek not in used_edges:
                avail_adj[u].add(v)
    
    # Check degree constraints: every vertex must have degree >= 2 in available graph
    for u in range(n):
        if len(avail_adj[u]) < 2:
            return None
    
    # Check connectivity of the available graph
    if not _is_connected(avail_adj, range(n)):
        return None

    path = [start]
    visited = {start}
    
    # Pre-compute initial degrees for heuristic
    # We will update degrees dynamically or just re-sort at each step (n=21 is small)
    
    def backtrack():
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            last = path[-1]
            if start in avail_adj[last]:
                return list(path)
            return None
        
        current = path[-1]
        
        # Heuristic: Sort neighbors by their current degree in the available graph
        # Fewer options first (Warnsdorff's rule)
        neighbors = list(avail_adj[current])
        
        # If we are at the second to last step, we must pick a neighbor that can connect back to start
        if len(path) == n - 1:
            # The next neighbor must be able to connect to start
            # Also, it must not be start (since start is visited and we are at n-1)
            valid_neighbors = []
            for nb in neighbors:
                if nb in visited:
                    continue
                # Check if nb can connect to start
                if start in avail_adj[nb]:
                    valid_neighbors.append(nb)
            neighbors = valid_neighbors
        
        # Sort by degree of the neighbor in the available graph
        # This helps prune branches that lead to isolated vertices
        neighbors.sort(key=lambda x: len(avail_adj[x]))
        
        for nb in neighbors:
            if nb in visited:
                continue
            
            # Pruning: Check if any unvisited vertex becomes isolated
            # This is expensive, so we only do it if the graph is small enough or we are deep
            # For n=21, we can do a quick check
            # Actually, checking connectivity of remaining nodes is O(n^2) which is fine for n=21
            # But we can skip it for speed and rely on degree checks in the next step
            # Let's add a lightweight degree check:
            # If we visit nb, the degree of nb's unvisited neighbors decreases by 1?
            # No, avail_adj is static. We need to check if any unvisited vertex 
            # (excluding those adjacent to current path end that might be saved) 
            # has degree 0 or 1 in the *remaining* graph?
            # It's tricky because the path end 'current' is removed from consideration for future steps 
            # except for the final connection.
            
            # Simple effective pruning:
            # If nb is not the last vertex, it must have at least one unvisited neighbor
            # (unless it is the last vertex, which connects to start)
            if len(path) < n - 1:
                unvisited_neighbors = [x for x in avail_adj[nb] if x not in visited and x != start]
                if not unvisited_neighbors:
                    continue
            
            path.append(nb)
            visited.add(nb)
            
            result = backtrack()
            if result is not None:
                return result
            
            path.pop()
            visited.remove(nb)
        
        return None
    
    return backtrack()

def _find_hamiltonian_cycle_v2(adj, n, used_edges, deadline):
    """Try multiple starting vertices to find a Hamiltonian cycle."""
    # Try starting from each vertex, ordered by degree
    vertices_by_degree = sorted(range(n), key=lambda v: len(adj[v]))
    
    for start in vertices_by_degree:
        if time.time() > deadline:
            return None
        result = _find_hamiltonian_cycle(adj, n, start, used_edges, deadline)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s safety margin
    
    n = P['n']
    all_edges = [tuple(sorted(e)) for e in P['edges']]
    adj = _build_adjacency(all_edges, n)
    
    used_edges = set()
    cycles = []
    
    # Theoretical max is 3. We aim for 3.
    max_cycles = 3
    
    for cycle_idx in range(max_cycles):
        if time.time() > deadline - 5:
            break
        
        found = None
        
        # Try standard backtracking
        result = _find_hamiltonian_cycle_v2(adj, n, used_edges, deadline - 5)
        if result is not None:
            found = result
        
        if found is None:
            break
        
        # Add edges of this cycle to used_edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            used_edges.add(_edge_key(u, v))
        
        cycles.append(found)
        
        # If we found the maximum possible, we can stop
        if len(cycles) == max_cycles:
            break
    
    # Ensure we have at least one cycle (the fallback)
    if not cycles:
        # Fallback: try the trivial cycle if all edges are available
        if len(used_edges) == 0:
            test_cycle = list(range(n))
            valid = True
            for i in range(n):
                u = test_cycle[i]
                v = test_cycle[(i + 1) % n]
                if _edge_key(u, v) not in set(all_edges):
                    valid = False
                    break
            if valid:
                cycles.append(test_cycle)
    
    # Convert to proper format: list of lists of Python ints
    result = []
    for cyc in cycles:
        result.append([int(v) for v in cyc])
    
    return result
# EVOLVE-BLOCK-END
