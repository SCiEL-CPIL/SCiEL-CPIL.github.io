"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_ham_cycle(adj, used_edges, start, deadline, rng, max_backtrack=50000):
    """Warnsdorff-guided DFS to find a Hamiltonian cycle in the remaining graph."""
    n = len(adj)
    path = [start]
    visited = [False] * n
    visited[start] = True
    backtrack_count = [0]
    
    def dfs(u):
        if time.time() > deadline:
            return False
        backtrack_count[0] += 1
        if backtrack_count[0] > max_backtrack:
            return False
        
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            if last in adj[start] and _edge_key(last, start) not in used_edges:
                return True
            return False
        
        # Get available neighbors (unvisited, edge not used)
        neighbors = []
        for v in adj[u]:
            if not visited[v] and _edge_key(u, v) not in used_edges:
                neighbors.append(v)
        
        if not neighbors:
            return False
        
        # Warnsdorff: sort by number of available moves (fewest first)
        def warnsdorff_score(v):
            avail = 0
            for w in adj[v]:
                if not visited[w] and _edge_key(v, w) not in used_edges:
                    avail += 1
            return avail
        
        neighbors.sort(key=warnsdorff_score)
        
        # Add small random perturbation for diversity
        if rng.random() < 0.2 and len(neighbors) > 1:
            idx = rng.randint(0, len(neighbors) - 1)
            neighbors[idx], neighbors[0] = neighbors[0], neighbors[idx]
        
        for v in neighbors:
            path.append(v)
            visited[v] = True
            if dfs(v):
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if dfs(start):
        return list(path)
    return None

def _cycle_edges(cycle):
    """Get the set of edges used by a cycle."""
    edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        edges.add(_edge_key(u, v))
    return edges

def _try_pack(adj, deadline, rng, max_cycles=3):
    """Try to pack edge-disjoint Hamiltonian cycles."""
    used_edges = set()
    cycles = []
    
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        # Find a start vertex with available edges
        candidates = []
        for v in range(P['n']):
            has_avail = False
            for w in adj[v]:
                if _edge_key(v, w) not in used_edges:
                    has_avail = True
                    break
            if has_avail:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Sort candidates by degree in remaining graph (highest first)
        candidates.sort(key=lambda v: sum(1 for w in adj[v] if _edge_key(v, w) not in used_edges), reverse=True)
        
        # Shuffle for diversity
        rng.shuffle(candidates)
        
        found = False
        for start in candidates:
            if time.time() > deadline:
                break
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng)
            if cycle:
                ce = _cycle_edges(cycle)
                if not ce & used_edges:
                    used_edges.update(ce)
                    cycles.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    return cycles

def _try_pack_with_backtracking(adj, deadline, rng, max_cycles=3):
    """Try to pack with more aggressive backtracking."""
    used_edges = set()
    cycles = []
    
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        # Try all possible starting vertices
        found = False
        for start in range(P['n']):
            if time.time() > deadline:
                break
            
            # Quick check: does start have enough available edges?
            avail_degree = sum(1 for w in adj[start] if _edge_key(start, w) not in used_edges)
            if avail_degree < 2:
                continue
            
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng, max_backtrack=100000)
            if cycle:
                ce = _cycle_edges(cycle)
                if not ce & used_edges:
                    used_edges.update(ce)
                    cycles.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    adj = _build_adj()
    
    best = []
    
    # Fallback: try to find at least one Hamiltonian cycle
    rng = random.Random(42)
    for attempt in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(attempt * 7919 + 13)
        
        # Try different packing strategies
        cycles = _try_pack(adj, deadline, rng, max_cycles=3)
        if len(cycles) > len(best):
            best = cycles
            if len(best) >= 3:
                break
        
        # If we didn't get 3, try with more backtracking
        if len(best) < 3 and time.time() < deadline - 5:
            cycles2 = _try_pack_with_backtracking(adj, deadline, rng, max_cycles=3)
            if len(cycles2) > len(best):
                best = cycles2
                if len(best) >= 3:
                    break
    
    # Final fallback: if we still don't have cycles, try one more time
    if not best:
        rng = random.Random(12345)
        for start in range(P['n']):
            if time.time() > deadline:
                break
            used_edges = set()
            cycle = _find_ham_cycle(adj, used_edges, start, deadline, rng)
            if cycle:
                best = [cycle]
                break
    
    # Convert to proper types
    result = []
    for c in best:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
