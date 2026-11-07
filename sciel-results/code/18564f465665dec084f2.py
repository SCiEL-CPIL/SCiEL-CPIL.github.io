"""Model-and-optimize: search vertex sets, fit edge-disjoint Hamiltonian cycles via randomized greedy."""
import itertools
import random
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE_MARGIN = 10  # seconds before kill to stop

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_one_ham_cycle(S_list, adj_S, used_edges, rng):
    """Find one Hamiltonian cycle on S_list avoiding used_edges. Returns list of vertices or None."""
    k = len(S_list)
    start = S_list[0]
    
    # Try multiple random orderings of neighbors
    # Reduced attempts to 100 as greedy usually succeeds for this graph density
    for attempt in range(100):
        visited = {start}
        path = [start]
        local_edges = set()
        success = True
        for _ in range(k - 1):
            cur = path[-1]
            neighbors = list(adj_S[cur])
            # Filter valid neighbors
            valid = []
            for nb in neighbors:
                if nb not in visited:
                    e = frozenset([cur, nb])
                    if e not in used_edges and e not in local_edges:
                        valid.append(nb)
            if not valid:
                success = False
                break
            if attempt < 50:
                # Pick the one with fewest remaining options (greedy heuristic)
                best_nb = None
                best_count = float('inf')
                for nb in valid:
                    unvisited_neighbors = 0
                    for x in adj_S[nb]:
                        if x not in visited and x != cur:
                            unvisited_neighbors += 1
                    if unvisited_neighbors < best_count:
                        best_count = unvisited_neighbors
                        best_nb = nb
                nb = best_nb
            elif attempt < 100:
                # Pick randomly
                nb = rng.choice(valid)
            else:
                # Pick randomly but avoid last option if possible
                if len(valid) > 1:
                    nb = rng.choice(valid[:-1])
                else:
                    nb = valid[0]
            visited.add(nb)
            local_edges.add(frozenset([path[-1], nb]))
            path.append(nb)
        if not success:
            continue
        # Check closing edge
        last = path[-1]
        close_edge = frozenset([last, start])
        if close_edge not in used_edges and close_edge not in local_edges:
            if start in adj_S[last]:
                return path
    return None

def _find_cycles_for_set(S_list, adj, target_count, deadline, rng):
    """Find up to target_count edge-disjoint Hamiltonian cycles on S_list."""
    S_set = frozenset(S_list)
    adj_S = {}
    for v in S_list:
        adj_S[v] = adj[v] & S_set
    
    # Check degree: each vertex must have degree >= 2*target_count in adj_S
    for v in S_list:
        if len(adj_S[v]) < 2 * target_count:
            return None  # Not enough edges
    
    used_edges = set()
    cycles = []
    for i in range(target_count):
        if time.time() > deadline:
            break
        cycle = _find_one_ham_cycle(S_list, adj_S, used_edges, rng)
        if cycle is None:
            break
        # Add edges to used
        n = len(cycle)
        for j in range(n):
            u = cycle[j]
            v = cycle[(j + 1) % n]
            used_edges.add(frozenset([u, v]))
        cycles.append(cycle)
    
    if len(cycles) == target_count:
        return cycles
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    rng = random.Random(42)
    
    adj = _build_adjacency()
    n = P['n']
    all_vertices = list(range(n))
    
    best_cycles = None
    best_count = 0
    
    # Try the full 21-vertex set first (max 3 cycles)
    # Then try smaller sets if needed
    
    # First, try full set with multiple seeds
    for seed in [42, 1, 7, 13, 99, 123, 456, 789, 1000, 2024]:
        if time.time() > deadline - 5:
            break
        rng_local = random.Random(seed)
        S_list = list(all_vertices)
        cycles = _find_cycles_for_set(S_list, adj, 3, deadline, rng_local)
        if cycles is not None:
            best_cycles = [list(c) for c in cycles]
            best_count = len(cycles)
            break
    
    # If full set didn't work, try smaller sets
    if best_count < 3:
        sizes_to_try = [14, 15, 16, 17, 18, 19, 20, 12, 13, 10, 11, 8, 9, 7, 6, 5, 4, 3]
        
        for s in sizes_to_try:
            if time.time() > deadline - 3:
                break
            
            max_possible = min(3, 6)  # max 3 due to degree constraint
            if max_possible <= best_count:
                continue
            
            # Generate candidate vertex sets
            if s >= 18:
                # Try removing a few vertices
                candidates = []
                removed = list(range(n - s, n))
                for r in range(min(5, n - s + 1)):
                    if r == 0:
                        continue
                    rem = list(removed[:r])
                    cand = [v for v in all_vertices if v not in rem]
                    if len(cand) == s:
                        candidates.append(tuple(cand))
            elif s >= 14:
                candidates = list(itertools.combinations(all_vertices, s))[:300]
            elif s >= 10:
                candidates = list(itertools.combinations(all_vertices, s))[:500]
            else:
                candidates = list(itertools.combinations(all_vertices, s))[:1000]
            
            rng.shuffle(candidates)
            
            for candidate in candidates:
                if time.time() > deadline - 2:
                    break
                
                S_list = list(candidate)
                cycles = _find_cycles_for_set(S_list, adj, max_possible, deadline, rng)
                
                if cycles is not None:
                    best_cycles = [list(c) for c in cycles]
                    best_count = len(cycles)
                    
                    if best_count >= 3:
                        break
            
            if best_count >= 3:
                break
    
    # Fallback: find any single cycle
    if best_cycles is None or len(best_cycles) == 0:
        # Find a triangle
        edge_set = set()
        for u, v in P['edges']:
            edge_set.add(frozenset([u, v]))
        
        for u, v in P['edges']:
            if time.time() > deadline:
                break
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    if frozenset([w, u]) in edge_set and frozenset([w, v]) in edge_set:
                        best_cycles = [[u, v, w]]
                        best_count = 1
                        break
            if best_cycles:
                break
    
    if best_cycles is None or len(best_cycles) == 0:
        # Absolute fallback
        return [list(range(P['n']))]
    
    # Validate and convert types
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result
