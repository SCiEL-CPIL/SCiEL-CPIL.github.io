# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_hamiltonian_cycle_dfs(vertices, adj, used_edges, start, vset, k):
    """
    Find a Hamiltonian cycle on 'vertices' (size k) using only unused edges.
    Uses Warnsdorff's heuristic.
    """
    path = [start]
    visited = {start}
    
    def get_candidates(current):
        candidates = []
        for nxt in adj[current]:
            if nxt not in visited and nxt in vset:
                e = _edge_key(current, nxt)
                if e not in used_edges:
                    # Calculate "degree" of nxt in the unvisited, unused-edge subgraph
                    deg = 0
                    for nnxt in adj[nxt]:
                        if nnxt not in visited and nnxt in vset:
                            e2 = _edge_key(nxt, nnxt)
                            if e2 not in used_edges:
                                deg += 1
                    
                    # Special check for the last step
                    if len(path) == k - 1:
                        if start in adj[nxt]:
                            e3 = _edge_key(nxt, start)
                            if e3 not in used_edges:
                                candidates.append((0, nxt))
                                continue
                        else:
                            continue
                    
                    candidates.append((deg, nxt))
        
        candidates.sort(key=lambda x: x[0])
        return [c[1] for c in candidates]

    def dfs(current):
        if len(path) == k:
            e = _edge_key(current, start)
            if start in adj[current] and e not in used_edges:
                return list(path)
            return None
        
        for nxt in get_candidates(current):
            e = _edge_key(current, nxt)
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

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin

    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    vertices = list(range(n))
    vset = set(vertices)
    k = n
    
    best_cycles = []
    
    # Strategy 1: Deterministic search with start=0
    # This is fast and often finds the solution.
    used_edges = set()
    cycles = []
    
    for i in range(3):  # Max 3 cycles possible
        if time.time() - start_time > deadline:
            break
        cycle = _find_hamiltonian_cycle_dfs(vertices, adj, used_edges, 0, vset, k)
        if cycle is None:
            break
        
        # Validate and add edges
        valid = True
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            e = _edge_key(u, v)
            if e in used_edges:
                valid = False
                break
            used_edges.add(e)
        
        if not valid:
            break
            
        cycles.append(cycle)
        
        if len(cycles) == 3:
            best_cycles = cycles
            # Early exit: found max possible
            return [[int(v) for v in c] for c in best_cycles]
            
    if len(cycles) > len(best_cycles):
        best_cycles = cycles
        
    # Strategy 2: Random restarts if deterministic failed or found < 3
    # Try different start nodes
    random.seed(42)
    for attempt in range(20):
        if time.time() - start_time > deadline:
            break
            
        used_edges = set()
        cycles = []
        
        # Pick a random start node for the first cycle
        start_node = random.randint(0, n - 1)
        
        for i in range(3):
            if time.time() - start_time > deadline:
                break
                
            # For subsequent cycles, stick to start=0 or random?
            # Start=0 is consistent.
            cur_start = start_node if i == 0 else 0
            
            cycle = _find_hamiltonian_cycle_dfs(vertices, adj, used_edges, cur_start, vset, k)
            if cycle is None:
                break
            
            valid = True
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = _edge_key(u, v)
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            
            if not valid:
                break
                
            cycles.append(cycle)
            
            if len(cycles) == 3:
                return [[int(v) for v in c] for c in cycles]
                
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            
    return [[int(v) for v in c] for c in best_cycles]

# EVOLVE-BLOCK-END
