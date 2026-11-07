"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
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

def _check_connected(adj, n):
    """Check if the graph is connected using BFS."""
    if not adj or not adj.get(0):
        return False
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for nb in adj.get(node, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return len(visited) == n

def _find_hamiltonian(adj, n, rng=None, deadline=None):
    """Find a Hamiltonian cycle using Warnsdorff's rule with forced moves."""
    if rng is None:
        rng = random.Random()
    
    def dfs(current, path, visited):
        if deadline and time.time() > deadline:
            return None
        
        if len(path) == n:
            if start in adj.get(current, set()):
                return list(path)
            return None
        
        neighbors = [nb for nb in adj.get(current, set()) if nb not in visited]
        if not neighbors:
            return None
        
        # Forced move: if only one unvisited neighbor, must go there
        if len(neighbors) == 1:
            nb = neighbors[0]
            visited.add(nb)
            path.append(nb)
            result = dfs(nb, path, visited)
            path.pop()
            visited.remove(nb)
            return result
        
        # Warnsdorff: sort by degree in remaining graph (fewest unvisited neighbors first)
        neighbors.sort(key=lambda nb: len([x for x in adj.get(nb, set()) if x not in visited]))
        
        # Add some randomization to avoid getting stuck
        if rng.random() < 0.2:
            top_k = min(3, len(neighbors))
            neighbors = neighbors[:top_k]
            rng.shuffle(neighbors)
        
        for nb in neighbors:
            visited.add(nb)
            path.append(nb)
            result = dfs(nb, path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(nb)
        return None
    
    start = 0
    visited = {start}
    path = [start]
    return dfs(start, path, visited)

def _find_hamiltonian_multi(adj, n, rng, max_starts=10, deadline=None):
    """Try multiple starting points to find a Hamiltonian cycle."""
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts[:max_starts]:
        if deadline and time.time() > deadline:
            break
        # Temporarily modify start by creating a new function
        def dfs(current, path, visited):
            if deadline and time.time() > deadline:
                return None
            
            if len(path) == n:
                if start in adj.get(current, set()):
                    return list(path)
                return None
            
            neighbors = [nb for nb in adj.get(current, set()) if nb not in visited]
            if not neighbors:
                return None
            
            if len(neighbors) == 1:
                nb = neighbors[0]
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, path, visited)
                path.pop()
                visited.remove(nb)
                return result
            
            neighbors.sort(key=lambda nb: len([x for x in adj.get(nb, set()) if x not in visited]))
            
            if rng.random() < 0.2:
                top_k = min(3, len(neighbors))
                neighbors = neighbors[:top_k]
                rng.shuffle(neighbors)
            
            for nb in neighbors:
                visited.add(nb)
                path.append(nb)
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nb)
            return None
        
        visited = {start}
        path = [start]
        result = dfs(start, path, visited)
        if result is not None:
            return result
    return None

def _try_find_k_cycles(edges, n, rng, deadline, k=3):
    """Try to find k edge-disjoint Hamiltonian cycles."""
    original_edges = set()
    for u, v in edges:
        original_edges.add(_edge_key(u, v))
    
    cycles = []
    remaining_edges = set(original_edges)
    
    for cycle_idx in range(k):
        if time.time() > deadline:
            break
        
        adj = _build_adjacency(list(remaining_edges), n)
        
        # Check minimum degree and connectivity
        min_deg = min(len(adj[v]) for v in range(n)) if adj else 0
        if min_deg < 2:
            break
        if not _check_connected(adj, n):
            break
        
        # Try multiple random restarts with different orderings
        found_cycle = None
        attempts = 30
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Shuffle adjacency for different orderings
            shuffled_adj = {}
            for v in range(n):
                shuffled_adj[v] = list(adj[v])
                rng.shuffle(shuffled_adj[v])
            
            temp_adj = defaultdict(set)
            for v in range(n):
                temp_adj[v] = set(shuffled_adj[v])
            
            cycle = _find_hamiltonian(temp_adj, n, rng=rng, deadline=deadline)
            if cycle is not None:
                found_cycle = cycle
                break
        
        if found_cycle is None:
            # Try with different starting points
            found_cycle = _find_hamiltonian_multi(adj, n, rng, max_starts=10, deadline=deadline)
        
        if found_cycle is None:
            break
        
        cycles.append(found_cycle)
        
        # Remove used edges from remaining
        for i in range(len(found_cycle)):
            u = found_cycle[i]
            v = found_cycle[(i + 1) % len(found_cycle)]
            ek = _edge_key(u, v)
            remaining_edges.discard(ek)
    
    return cycles

def _validate_cycles(cycles, n, edges):
    """Validate that cycles are valid and edge-disjoint."""
    if not cycles:
        return False
    
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    all_vertices = None
    for cycle in cycles:
        if len(cycle) < 3:
            return False
        if len(set(cycle)) != len(cycle):
            return False
        if len(cycle) != n:
            return False
        vertices = set(cycle)
        if all_vertices is None:
            all_vertices = vertices
        else:
            if vertices != all_vertices:
                return False
    
    # Check all edges are valid and no edge is used twice
    used_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in used_edges:
                return False
            if ek not in edge_set:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    best_cycles = []
    
    # Strategy: Try many random seeds to find 3 edge-disjoint Hamiltonian cycles
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed * 31 + 7)
        cycles = _try_find_k_cycles(edges, n, rng, deadline, k=3)
        if len(cycles) > len(best_cycles):
            best_cycles = [c[:] for c in cycles]
            if len(best_cycles) == 3:
                if _validate_cycles(best_cycles, n, edges):
                    return [[int(v) for v in c] for c in best_cycles]
    
    # If we found 3 cycles, return them
    if len(best_cycles) == 3:
        if _validate_cycles(best_cycles, n, edges):
            return [[int(v) for v in c] for c in best_cycles]
    
    # Return whatever we found
    if best_cycles:
        return [[int(v) for v in c] for c in best_cycles]
    
    # Fallback: return a single Hamiltonian cycle if we can find one
    adj = _build_adjacency(edges, n)
    rng = random.Random(12345)
    cycle = _find_hamiltonian_multi(adj, n, rng, max_starts=20, deadline=deadline)
    if cycle:
        return [[int(v) for v in cycle]]
    
    # Last resort fallback
    return [list(range(n))]
# EVOLVE-BLOCK-END
