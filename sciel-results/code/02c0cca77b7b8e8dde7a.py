import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, remaining_edges, start, rng, max_backtrack=50000):
    """Try to find a Hamiltonian cycle starting from 'start' using Warnsdorff + randomized backtracking."""
    n = P['n']
    vertices = set(range(n))
    
    # Build local adjacency from remaining edges
    local_adj = defaultdict(set)
    for u, v in remaining_edges:
        local_adj[u].add(v)
        local_adj[v].add(u)
    
    # Check if start has enough degree
    if len(local_adj[start]) < 2:
        return None
    
    # We need a cycle: start -> ... -> start, visiting all n vertices exactly once
    # This is a Hamiltonian cycle in the subgraph defined by remaining_edges
    
    # Use iterative deepening backtracking with Warnsdorff heuristic
    def backtrack(path, visited):
        if len(path) == n:
            # Check if last vertex connects back to start
            if start in local_adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        # Get unvisited neighbors, sorted by Warnsdorff (fewest onward options)
        candidates = []
        for nb in local_adj[current]:
            if nb not in visited:
                # Count onward options from nb (excluding current)
                onward = len([x for x in local_adj[nb] if x not in visited and x != current])
                candidates.append((onward, nb))
        
        if not candidates:
            return None
        
        # Sort by Warnsdorff: fewest onward options first (to avoid dead ends)
        candidates.sort(key=lambda x: x[0])
        
        # Add some randomness: shuffle among top candidates
        # Take top k candidates and shuffle
        k = min(len(candidates), 3)
        top = candidates[:k]
        rng.shuffle(top)
        # Also include the rest in case
        rest = candidates[k:]
        shuffled = top + rest
        
        for _, nb in shuffled:
            result = backtrack(path + [nb], visited | {nb})
            if result is not None:
                return result
        
        return None
    
    # Try with the start vertex
    result = backtrack([start], {start})
    return result

def _extract_cycle(remaining_edges, rng, max_attempts=200):
    """Try to extract one Hamiltonian cycle from remaining edges."""
    n = P['n']
    
    for attempt in range(max_attempts):
        # Pick a random start vertex that has degree >= 2 in remaining graph
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        valid_starts = [v for v in range(n) if len(adj[v]) >= 2]
        if not valid_starts:
            return None
        
        start = rng.choice(valid_starts)
        
        # Try to find Hamiltonian cycle
        cycle = _find_hamiltonian_cycle(adj, remaining_edges, start, rng)
        if cycle is not None:
            # Verify it's valid
            if len(set(cycle)) != n:
                continue
            # Check all edges exist in remaining
            valid = True
            for i in range(n):
                u, v = cycle[i], cycle[(i+1) % n]
                ek = _edge_key(u, v)
                if ek not in remaining_edges:
                    valid = False
                    break
            if valid:
                return cycle
    return None

def _verify_packing(cycles, all_edges_set):
    """Verify that all cycles are valid, edge-disjoint, and use the same vertex set."""
    if not cycles:
        return False
    
    n = P['n']
    used_edges = set()
    
    for cycle in cycles:
        # Check it's a list of ints
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        
        # Check all vertices are valid ints
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
        
        # Check distinct vertices
        if len(set(cycle)) != len(cycle):
            return False
        
        # Check vertex set is exactly all n vertices
        if set(cycle) != set(range(n)):
            return False
        
        # Check edges
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    all_edges = P['edges']
    all_edges_set = set()
    for u, v in all_edges:
        all_edges_set.add(_edge_key(u, v))
    
    best_packing = []
    
    # Try multiple random seeds with varied strategies
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        remaining = set(all_edges_set)
        cycles = []
        
        for _ in range(10):  # Try up to 10 cycles (max allowed)
            if time.time() > deadline:
                break
            
            # Determine vertex set size: try smaller sets for more cycles
            # First try full Hamiltonian, then smaller subsets
            if len(cycles) == 0:
                # Try Hamiltonian cycle on all 21 vertices
                cycle = _extract_cycle(remaining, rng, max_attempts=30)
                if cycle is None:
                    # Try smaller vertex sets
                    break
                cycles.append(cycle)
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i+1) % len(cycle)]
                    ek = _edge_key(u, v)
                    remaining.discard(ek)
            else:
                # For subsequent cycles, we must use the same vertex set
                # Check if we can find another cycle on the same vertex set
                if len(cycles[0]) == 21:
                    # Hamiltonian: try to find another on all 21
                    cycle = _extract_cycle(remaining, rng, max_attempts=30)
                    if cycle is None or len(cycle) != 21:
                        break
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i+1) % len(cycle)]
                        ek = _edge_key(u, v)
                        remaining.discard(ek)
                else:
                    # Smaller vertex set: find cycle on same vertex set
                    target_vertices = set(cycles[0])
                    # Build subgraph on target_vertices
                    sub_edges = set()
                    for ek in remaining:
                        u, v = ek
                        if u in target_vertices and v in target_vertices:
                            sub_edges.add(ek)
                    if not sub_edges:
                        break
                    # Find cycle using only edges in sub_edges
                    cycle = _extract_cycle(sub_edges, rng, max_attempts=30)
                    if cycle is None or set(cycle) != target_vertices:
                        break
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i+1) % len(cycle)]
                        ek = _edge_key(u, v)
                        remaining.discard(ek)
        
        if len(cycles) > len(best_packing):
            best_packing = cycles[:]
        
        # Early exit if we found 3 or more
        if len(best_packing) >= 3:
            break
    
    # Verify best packing
    if best_packing:
        if _verify_packing(best_packing, all_edges_set):
            return [list(c) for c in best_packing]
    
    # Fallback: try a simpler deterministic approach
    if not best_packing:
        rng = random.Random(42)
        remaining = set(all_edges_set)
        cycle = _extract_cycle(remaining, rng, max_attempts=100)
        if cycle is not None:
            if _verify_packing([cycle], all_edges_set):
                return [list(cycle)]
    
    return []
# EVOLVE-BLOCK-END
