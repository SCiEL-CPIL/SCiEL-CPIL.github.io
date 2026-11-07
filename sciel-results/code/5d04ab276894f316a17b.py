"""Model-and-optimize: find edge-disjoint Hamiltonian cycles via residual decomposition."""
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(_edge_key(u, v))
    return adj, edge_set

def _cycle_edges(cycle):
    m = len(cycle)
    edges = []
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        edges.append(_edge_key(a, b))
    return edges

def _is_valid_cycle(cycle, adj, edge_set):
    if len(cycle) < 3:
        return False
    if len(cycle) != len(set(cycle)):
        return False
    m = len(cycle)
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        if b not in adj[a]:
            return False
        if _edge_key(a, b) not in edge_set:
            return False
    return True

def _find_hamiltonian_cycle(adj, edge_set, used_edges, start=0, deadline=None, rng=None, heuristic='greedy'):
    """Find a Hamiltonian cycle avoiding used_edges."""
    n = P['n']
    
    def dfs(path, visited):
        if deadline and time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used_edges and e in edge_set:
                return path[:]
            return None
        
        cur = path[-1]
        neighbors = list(adj[cur])
        
        # Filter valid neighbors
        valid = []
        for nb in neighbors:
            if nb not in visited:
                e = _edge_key(cur, nb)
                if e not in used_edges:
                    valid.append(nb)
        
        if not valid:
            return None
        
        if rng and rng.random() < 0.3:
            rng.shuffle(valid)
        elif heuristic == 'greedy':
            # Prefer neighbors with fewer unvisited neighbors (Warnsdorff's rule)
            def count_unvisited(nb):
                cnt = 0
                for x in adj[nb]:
                    if x not in visited and x != cur:
                        cnt += 1
                return cnt
            valid.sort(key=count_unvisited)
        elif heuristic == 'random':
            rng.shuffle(valid)
        else:
            valid.sort()
        
        for nb in valid:
            result = dfs(path + [nb], visited | {nb})
            if result is not None:
                return result
        return None
    
    return dfs([start], {start})

def _extract_cycle_from_residual(residual_adj, start=0):
    """Extract a Hamiltonian cycle from a 2-regular residual graph."""
    n = P['n']
    # Check if 2-regular
    for u in range(n):
        if len(residual_adj[u]) != 2:
            return None
    
    # Walk the cycle
    cyc = [start]
    prev = -1
    cur = start
    for _ in range(n):
        nxts = residual_adj[cur]
        if len(nxts) == 0:
            return None
        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            nxt = nxts[0] if nxts[0] != prev else nxts[1]
        if nxt == start and len(cyc) < n:
            return None  # Closed early
        cyc.append(nxt)
        prev, cur = cur, nxt
    
    if len(cyc) != n + 1 or cyc[-1] != start:
        return None
    return cyc[1:]

def _build_residual_adj(edge_set, used_edges):
    """Build adjacency from remaining edges."""
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        e = _edge_key(u, v)
        if e not in used_edges:
            adj[u].add(v)
            adj[v].add(u)
    return adj

def _find_cycles_decomposition(adj, edge_set, deadline, rng):
    """Find up to 3 edge-disjoint Hamiltonian cycles via residual decomposition."""
    
    # Try multiple random starting points and heuristics
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        start = rng.randint(0, P['n'] - 1)
        heuristic = rng.choice(['greedy', 'random', 'sorted'])
        
        used_edges = set()
        cycles = []
        
        # Find first Hamiltonian cycle
        c1 = _find_hamiltonian_cycle(adj, edge_set, used_edges, start, deadline, rng, heuristic)
        if c1 is None:
            continue
        if not _is_valid_cycle(c1, adj, edge_set):
            continue
        
        cycles.append(c1)
        for e in _cycle_edges(c1):
            used_edges.add(e)
        
        # Extract second cycle from residual
        res_adj = _build_residual_adj(edge_set, used_edges)
        c2 = _extract_cycle_from_residual(res_adj)
        if c2 is None:
            continue
        if not _is_valid_cycle(c2, adj, edge_set):
            continue
        
        cycles.append(c2)
        for e in _cycle_edges(c2):
            used_edges.add(e)
        
        # Extract third cycle from residual
        res_adj = _build_residual_adj(edge_set, used_edges)
        c3 = _extract_cycle_from_residual(res_adj)
        if c3 is None:
            continue
        if not _is_valid_cycle(c3, adj, edge_set):
            continue
        
        cycles.append(c3)
        for e in _cycle_edges(c3):
            used_edges.add(e)
        
        # Verify all cycles are edge-disjoint and valid
        all_edges = set()
        valid = True
        for c in cycles:
            for e in _cycle_edges(c):
                if e in all_edges:
                    valid = False
                    break
                all_edges.add(e)
            if not valid:
                break
        
        if valid and len(cycles) == 3:
            return [list(c) for c in cycles]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    rng = random.Random(42)
    
    adj, edge_set = _build_adjacency()
    
    # Try residual decomposition approach
    result = _find_cycles_decomposition(adj, edge_set, deadline, rng)
    
    if result is not None and len(result) >= 3:
        # Convert to plain Python ints
        final = []
        for c in result:
            final.append([int(v) for v in c])
        return final
    
    # Fallback: find as many cycles as possible via sequential search
    used_edges = set()
    cycles = []
    
    for i in range(3):
        if time.time() > deadline - 5:
            break
        
        best_cycle = None
        for attempt in range(50):
            start = rng.randint(0, P['n'] - 1)
            heuristic = rng.choice(['greedy', 'random'])
            c = _find_hamiltonian_cycle(adj, edge_set, used_edges, start, deadline - 5, rng, heuristic)
            if c is not None and _is_valid_cycle(c, adj, edge_set):
                best_cycle = c
                break
        
        if best_cycle is None:
            break
        
        cycles.append(best_cycle)
        for e in _cycle_edges(best_cycle):
            used_edges.add(e)
    
    if cycles:
        final = []
        for c in cycles:
            final.append([int(v) for v in c])
        return final
    
    # Absolute fallback: single known cycle
    return [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
