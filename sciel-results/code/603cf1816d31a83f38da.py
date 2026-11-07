# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, start=None, deadline=None):
    """Find a Hamiltonian cycle on the given vertex set using only unused edges."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    
    path = [start]
    visited = {start}
    node_count = 0
    
    def get_candidates(current):
        candidates = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                e = (min(current, nxt), max(current, nxt))
                if e not in used_edges:
                    deg = 0
                    for nnxt in adj[nxt]:
                        if nnxt not in visited and nnxt in vset:
                            e2 = (min(nxt, nnxt), max(nxt, nnxt))
                            if e2 not in used_edges:
                                deg += 1
                    if len(path) == k - 1:
                        if start in adj[nxt]:
                            e3 = (min(nxt, start), max(nxt, start))
                            if e3 not in used_edges:
                                candidates.append((0, nxt))
                                continue
                    candidates.append((deg, nxt))
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def dfs(current):
        nonlocal node_count
        node_count += 1
        if deadline is not None and node_count % 10000 == 0 and time.time() > deadline:
            return None
        
        if len(path) == k:
            if start in adj[current]:
                e = (min(current, start), max(current, start))
                if e not in used_edges:
                    return list(path)
            return None
        
        for nxt in get_candidates(current):
            e = (min(current, nxt), max(current, nxt))
            if e not in used_edges:
                visited.add(nxt)
                path.append(nxt)
                result = dfs(nxt)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    return dfs(start)

def _find_cycles_for_vertex_set(vertices, adj, time_limit, max_cycles=10, seed=42):
    """Find max number of edge-disjoint cycles on the given vertex set."""
    vertices = list(vertices)
    used_edges = set()
    cycles = []
    start_time = time.time()
    deadline = start_time + time_limit
    
    for _ in range(max_cycles):
        if time.time() > deadline - 0.5:
            break
        
        cycle = _find_ham_cycle(vertices, adj, used_edges, deadline=deadline)
        if cycle is None:
            break
        
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                valid = False
                break
            used_edges.add(e)
        
        if valid:
            cycles.append(cycle)
    
    return cycles

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add((min(u, v), max(u, v)))
    
    adj = _build_adj(n, edges)
    full_vertices = list(range(n))
    
    best_cycles = []
    best_count = 0
    
    # Try multiple random seeds and start strategies
    for seed in [42, 123, 7, 999, 555]:
        if time.time() > deadline - 2:
            break
        
        random.seed(seed)
        # Shuffle vertex order for diversity in tie-breaking
        shuffled_vertices = full_vertices[:]
        random.shuffle(shuffled_vertices)
        
        cycles = _find_cycles_for_vertex_set(shuffled_vertices, adj, deadline - time.time(), max_cycles=10, seed=seed)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
        
        if best_count >= 10:
            break
    
    # If we didn't find many, try deterministic approach with different starts
    if best_count < 5:
        for start_node in range(n):
            if time.time() > deadline - 1:
                break
            
            used_edges = set()
            cycles = []
            
            for _ in range(10):
                if time.time() > deadline - 0.5:
                    break
                cycle = _find_ham_cycle(full_vertices, adj, used_edges, start=start_node if _ == 0 else None, deadline=deadline)
                if cycle is None:
                    break
                
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                
                if valid:
                    cycles.append(cycle)
            
            if len(cycles) > best_count:
                best_count = len(cycles)
                best_cycles = cycles
            
            if best_count >= 10:
                break
    
    # Ensure best_cycles is non-empty
    if not best_cycles:
        cycles = _find_cycles_for_vertex_set(full_vertices, adj, deadline - time.time(), max_cycles=10)
        if cycles:
            best_cycles = cycles
    
    # Convert to proper types (list of tuples)
    result = []
    for cycle in best_cycles[:10]:
        result.append(tuple(cycle))
    
    return result

# EVOLVE-BLOCK-END
