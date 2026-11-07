"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _ham_cycle_dp(vertices, adj, used_edges):
    """Bitmask DP for Hamiltonian cycle on small vertex sets (up to ~15)."""
    k = len(vertices)
    if k < 3 or k > 15:
        return None
    vset = set(vertices)
    # Map vertices to 0..k-1
    vlist = sorted(vset)
    vidx = {v: i for i, v in enumerate(vlist)}
    n = k
    # Check if all needed edges exist and are unused
    # Build adjacency bitmask
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u = vlist[i]
            v = vlist[j]
            e = _edge_key(u, v)
            if v in adj[u] and e not in used_edges:
                adj_mask[i] |= (1 << j)
    
    # Check min degree
    for i in range(n):
        if bin(adj_mask[i]).count('1') < 2:
            return None
    
    # DP: dp[mask][last] = reachable
    # mask is n-bit, last is 0..n-1
    # Start at vertex 0
    full_mask = (1 << n) - 1
    dp = [None] * (1 << n)
    dp[1] = [0]  # Only vertex 0 visited, at vertex 0
    
    for mask in range(1 << n):
        if dp[mask] is None:
            continue
        for last in dp[mask]:
            # Try to extend
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                if not (adj_mask[last] & (1 << nxt)):
                    continue
                new_mask = mask | (1 << nxt)
                if dp[new_mask] is None:
                    dp[new_mask] = []
                if nxt not in dp[new_mask]:
                    dp[new_mask].append(nxt)
    
    if dp[full_mask] is None:
        return None
    
    # Check if we can close the cycle
    for last in dp[full_mask]:
        if adj_mask[last] & 1:  # Edge from last to vertex 0
            # Reconstruct path
            path = [last]
            mask = full_mask
            while mask != 1:
                mask ^= (1 << path[-1])
                for prev in range(n):
                    if mask & (1 << prev):
                        continue
                    if dp[mask | (1 << prev)] and last in dp[mask | (1 << prev)]:
                        # prev -> last is valid
                        if adj_mask[prev] & (1 << last):
                            path.append(prev)
                            last = prev
                            break
                else:
                    return None
            path.reverse()
            # Convert back to original vertices
            return [vlist[i] for i in path]
    return None

def _find_ham_cycle_heuristic(vertices, adj, used_edges, time_limit=5.0):
    """Find a Hamiltonian cycle using heuristic + backtracking with pruning."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    start = min(vset)
    start_time = time.time()
    
    # Precompute available neighbors for each vertex
    avail = {}
    for v in vset:
        avail[v] = [u for u in adj[v] if u in vset and _edge_key(v, u) not in used_edges]
        if len(avail[v]) < 2:
            return None
    
    # Sort neighbors by degree (Warnsdorff heuristic)
    for v in vset:
        avail[v].sort(key=lambda u: len(avail.get(u, [])))
    
    path = [start]
    visited = {start}
    
    def dfs(current, depth):
        if time.time() - start_time > time_limit:
            return None
        if depth == k:
            if start in adj[current] and _edge_key(current, start) not in used_edges:
                return list(path)
            return None
        for nxt in avail[current]:
            if nxt not in visited:
                # Pruning: check if remaining unvisited vertices can still form a path
                remaining = vset - visited - {nxt}
                if remaining:
                    # Check connectivity of remaining + current path endpoints
                    # Simple check: each remaining vertex must have at least 1 available neighbor
                    # in remaining or in {current, start} (for closing)
                    for rv in remaining:
                        has_conn = False
                        for rn in avail[rv]:
                            if rn in remaining or rn == start:
                                has_conn = True
                                break
                        if not has_conn:
                            break
                    if not has_conn:
                        continue
                visited.add(nxt)
                path.append(nxt)
                result = dfs(nxt, depth + 1)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nxt)
        return None
    
    return dfs(start, 1)

def _find_cycles_for_vertex_set(vertices, all_edges_set, adj, time_limit, max_cycles=10):
    """Find max number of edge-disjoint cycles on the given vertex set."""
    vertices = list(vertices)
    used_edges = set()
    cycles = []
    start_time = time.time()
    
    for _ in range(max_cycles):
        if time.time() - start_time > time_limit:
            break
        # Choose method based on vertex set size
        if len(vertices) <= 15:
            cycle = _ham_cycle_dp(vertices, adj, used_edges)
        else:
            cycle = _find_ham_cycle_heuristic(vertices, adj, used_edges, min(time_limit, 10.0))
        if cycle is None:
            break
        # Mark edges as used
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = _edge_key(u, v)
            if e in used_edges:
                valid = False
                break
            used_edges.add(e)
        if valid:
            cycles.append(cycle)
        else:
            # Roll back
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = _edge_key(u, v)
                used_edges.discard(e)
            break
    
    return cycles

def construct():
    start_time = time.time()
    deadline = 80  # seconds, leave margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    all_edges_set = set()
    for u, v in edges:
        all_edges_set.add(_edge_key(u, v))
    
    adj = _build_adj(n, edges)
    
    best_cycles = []
    best_count = 0
    
    # Strategy 1: Full vertex set - try to find 3 edge-disjoint Hamiltonian cycles
    full_vertices = list(range(n))
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1:
        cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, remaining_time)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
    
    # Strategy 2: Try subsets of various sizes
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1:
        import random
        random.seed(42)
        for subset_size in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
            if time.time() - start_time > deadline - 1:
                break
            for trial in range(30):
                if time.time() - start_time > deadline - 1:
                    break
                subset = random.sample(range(n), subset_size)
                sset = set(subset)
                induced_edges = sum(1 for u, v in edges if u in sset and v in sset)
                # Need at least 2*subset_size edges for 2 cycles
                if induced_edges < 2 * subset_size:
                    continue
                cycles = _find_cycles_for_vertex_set(subset, all_edges_set, adj, min(remaining_time, 3))
                if len(cycles) > best_count:
                    best_count = len(cycles)
                    best_cycles = cycles
    
    # Strategy 3: Structured subsets
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1:
        for start_v in range(0, n - 6):
            if time.time() - start_time > deadline - 1:
                break
            for size in [7, 8, 9, 10, 11]:
                if start_v + size > n:
                    continue
                subset = list(range(start_v, start_v + size))
                cycles = _find_cycles_for_vertex_set(subset, all_edges_set, adj, min(remaining_time, 2))
                if len(cycles) > best_count:
                    best_count = len(cycles)
                    best_cycles = cycles
    
    # Ensure best_cycles is non-empty
    if not best_cycles:
        cycles = _find_cycles_for_vertex_set(full_vertices, all_edges_set, adj, deadline - (time.time() - start_time))
        if cycles:
            best_cycles = cycles
        else:
            # Last resort: find any cycle
            for u, v in edges:
                for w in adj.get(u, set()):
                    if w != v and _edge_key(v, w) in all_edges_set:
                        best_cycles = [[u, v, w]]
                        break
                if best_cycles:
                    break
            if not best_cycles:
                best_cycles = [[0, 1, 2]]
    
    # Convert to proper types
    result = []
    for cycle in best_cycles[:10]:
        result.append(tuple(cycle))
    
    return result

# EVOLVE-BLOCK-END
