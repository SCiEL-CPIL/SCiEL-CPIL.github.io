"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _cycle_edges(cycle):
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges

def _find_ham_cycle(adj, n, forbidden_edges, start, deadline, rng):
    """
    Find a Hamiltonian cycle starting at `start` avoiding `forbidden_edges`.
    Uses DFS with pruning:
    1. Degree check: Unvisited vertices must have at least 2 available edges 
       (one to enter, one to leave, unless it's the start/end connection).
    2. Connectivity check (optional, expensive).
    3. Warnsdorff heuristic for ordering.
    """
    path = [start]
    visited = [False] * n
    visited[start] = True
    used_edges = set()
    
    # Pre-compute adjacency lists for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # To speed up degree checks, we maintain a count of available edges for each vertex
    # available_deg[v] = number of neighbors u such that edge (v,u) is not forbidden and not used
    # Note: For the start node, it needs 2 edges in the final cycle.
    # For intermediate nodes, 2.
    # For the end node (which connects to start), 1 to start + 1 to prev.
    
    # Initialize available degrees
    avail_deg = [0] * n
    for i in range(n):
        for v in adj_lists[i]:
            e = (min(i, v), max(i, v))
            if e not in forbidden_edges:
                avail_deg[i] += 1
    
    # If start has < 2 available edges, impossible
    if avail_deg[start] < 2:
        return None

    def dfs(path, visited, used_edges, avail_deg):
        if time.time() > deadline:
            return None
            
        if len(path) == n:
            last = path[-1]
            first = path[0]
            e = (min(last, first), max(last, first))
            if first in adj[last] and e not in forbidden_edges and e not in used_edges:
                return list(path)
            return None
        
        curr = path[-1]
        
        # Heuristic: Choose next vertex with minimum available degree (Warnsdorff)
        # But we must ensure we don't isolate other vertices.
        
        candidates = []
        for v in adj_lists[curr]:
            if not visited[v]:
                e = (min(curr, v), max(curr, v))
                if e not in forbidden_edges and e not in used_edges:
                    candidates.append(v)
        
        if not candidates:
            return None
            
        # Sort candidates by available degree of the candidate vertex
        # We want to visit vertices with fewer options first
        candidates.sort(key=lambda v: avail_deg[v])
        
        for v in candidates:
            e = (min(curr, v), max(curr, v))
            
            # Tentative move
            path.append(v)
            visited[v] = True
            used_edges.add(e)
            
            # Update avail_deg
            # v loses 1 degree (edge to curr is used)
            avail_deg[v] -= 1
            # curr loses 1 degree (edge to v is used)
            avail_deg[curr] -= 1
            
            # Check for dead ends:
            # Any unvisited vertex w must have avail_deg[w] >= 2.
            # Exception: If w is the start node, it needs 1 more edge to connect to the end.
            # Actually, start is visited. So we only check unvisited nodes.
            # Also, the end node (last in path) will need to connect to start.
            # If len(path) == n-1, the last node v needs to connect to start.
            
            dead_end = False
            for w in range(n):
                if not visited[w]:
                    if avail_deg[w] < 2:
                        dead_end = True
                        break
            
            # Special check: If we are at n-1, the current last node (v) must connect to start.
            if not dead_end and len(path) == n - 1:
                last_v = v
                e_close = (min(last_v, start), max(last_v, start))
                if e_close not in adj[last_v] or e_close in forbidden_edges or e_close in used_edges:
                    dead_end = True
            
            if dead_end:
                # Undo
                path.pop()
                visited[v] = False
                used_edges.discard(e)
                avail_deg[v] += 1
                avail_deg[curr] += 1
                continue
            
            result = dfs(path, visited, used_edges, avail_deg)
            if result is not None:
                return result
            
            # Undo
            path.pop()
            visited[v] = False
            used_edges.discard(e)
            avail_deg[v] += 1
            avail_deg[curr] += 1
            
        return None

    return dfs(path, visited, used_edges, avail_deg)

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin for 90s limit
    n = P['n']
    adj = _build_adj()
    
    best_cycles = []
    max_cycles = 3
    
    # Strategy: Try to find 3 cycles.
    # 1. Find first cycle.
    # 2. Find second cycle disjoint from first.
    # 3. Find third cycle disjoint from first and second.
    
    # Helper to get forbidden edges from current cycles
    def get_forbidden(cycles):
        forb = set()
        for c in cycles:
            forb.update(_cycle_edges(c))
        return forb
    
    # 1. Find first cycle
    # Try different starts to find a "good" first cycle?
    # Usually any Ham cycle is fine, but some might make it harder to find others.
    # Let's just find one quickly.
    
    found = False
    for start in range(n):
        if time.time() > deadline - 5: break
        cycle = _find_ham_cycle(adj, n, set(), start, deadline, random.Random(0))
        if cycle:
            # Validate
            if len(set(cycle)) == n:
                edges_c = _cycle_edges(cycle)
                # Check all edges exist
                valid = True
                for u, v in edges_c:
                    if v not in adj[u]:
                        valid = False
                        break
                if valid:
                    best_cycles.append(cycle)
                    found = True
                    break
    
    if not found:
        return [list(range(n))] # Fallback, likely invalid but prevents crash
    
    # 2. Find second cycle
    forb = get_forbidden(best_cycles)
    found2 = False
    # Try multiple starts and random restarts if stuck
    for start in range(n):
        if time.time() > deadline - 10: break
        if found2: break
        cycle = _find_ham_cycle(adj, n, forb, start, deadline, random.Random(1))
        if cycle:
            if len(set(cycle)) == n:
                edges_c = _cycle_edges(cycle)
                if not (edges_c & forb):
                    best_cycles.append(cycle)
                    found2 = True
    
    if not found2:
        # Random restarts for 2nd cycle
        rng = random.Random(42)
        for _ in range(20):
            if time.time() > deadline - 10: break
            if found2: break
            start = rng.randint(0, n-1)
            cycle = _find_ham_cycle(adj, n, forb, start, deadline, rng)
            if cycle:
                if len(set(cycle)) == n:
                    edges_c = _cycle_edges(cycle)
                    if not (edges_c & forb):
                        best_cycles.append(cycle)
                        found2 = True
                        forb = get_forbidden(best_cycles)
    
    if len(best_cycles) < 2:
        return best_cycles
    
    # 3. Find third cycle
    forb = get_forbidden(best_cycles)
    found3 = False
    for start in range(n):
        if time.time() > deadline - 2: break
        if found3: break
        cycle = _find_ham_cycle(adj, n, forb, start, deadline, random.Random(2))
        if cycle:
            if len(set(cycle)) == n:
                edges_c = _cycle_edges(cycle)
                if not (edges_c & forb):
                    best_cycles.append(cycle)
                    found3 = True
    
    if not found3:
        # Random restarts for 3rd cycle
        rng = random.Random(99)
        for _ in range(50):
            if time.time() > deadline - 2: break
            if found3: break
            start = rng.randint(0, n-1)
            cycle = _find_ham_cycle(adj, n, forb, start, deadline, rng)
            if cycle:
                if len(set(cycle)) == n:
                    edges_c = _cycle_edges(cycle)
                    if not (edges_c & forb):
                        best_cycles.append(cycle)
                        found3 = True
    
    # Final validation
    if not best_cycles:
        return [list(range(n))]
        
    # Ensure types are correct
    final_cycles = []
    for c in best_cycles:
        final_cycles.append([int(v) for v in c])
        
    return final_cycles

# EVOLVE-BLOCK-END
