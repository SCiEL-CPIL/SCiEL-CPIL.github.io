import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin for 90s limit
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    best_cycles = []
    best_count = 0
    
    def find_hamiltonian_cycle_fast(vertices, used_edges, adj, deadline):
        """Find a Hamiltonian cycle using DFS with strong pruning."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Precompute available neighbors
        avail = {}
        for v in vertices:
            avail[v] = [u for u in adj[v] if u in vertices and (min(u,v), max(u,v)) not in used_edges]
            if len(avail[v]) < 2:
                return None
        
        # Choose start vertex with minimum available degree
        start = min(vertices, key=lambda v: len(avail[v]))
        remaining = set(vertices) - {start}
        path = [start]
        
        # Use iterative deepening with time limit
        def backtrack(path, remaining, depth):
            nonlocal deadline
            if time.time() > deadline:
                return None
            if len(path) == k:
                last = path[-1]
                edge = (min(last, start), max(last, start))
                if edge not in used_edges and last in avail[start]:
                    return path[:]
                return None
            
            last = path[-1]
            # Get candidates sorted by future options (most constrained first)
            candidates = []
            for v in remaining:
                if v in avail[last]:
                    # Count remaining options for v after using edge (last,v)
                    opts = 0
                    for w in avail[v]:
                        if w != last and w in remaining:
                            opts += 1
                    candidates.append((opts, v))
            
            if not candidates:
                return None
            
            candidates.sort()
            
            for _, v in candidates:
                if time.time() > deadline:
                    return None
                edge = (min(last, v), max(last, v))
                if edge not in used_edges:
                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining, depth + 1)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
            return None
        
        return backtrack(path, remaining, 0)
    
    def find_cycle_with_restarts(vertices, used_edges, adj, deadline, num_restarts=20):
        """Try multiple random starts and orderings to find a Hamiltonian cycle."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        rng = random.Random(hash(frozenset(vertices)) ^ hash(frozenset(used_edges)) & 0xFFFF)
        
        for attempt in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Precompute available neighbors
            avail = {}
            valid = True
            for v in vertices:
                avail[v] = [u for u in adj[v] if u in vertices and (min(u,v), max(u,v)) not in used_edges]
                if len(avail[v]) < 2:
                    valid = False
                    break
            if not valid:
                return None
            
            # Random start
            start = rng.choice(vertices)
            remaining = set(vertices) - {start}
            path = [start]
            
            def backtrack(path, remaining):
                nonlocal deadline
                if time.time() > deadline:
                    return None
                if len(path) == k:
                    last = path[-1]
                    edge = (min(last, start), max(last, start))
                    if edge not in used_edges and last in avail[start]:
                        return path[:]
                    return None
                
                last = path[-1]
                candidates = []
                for v in remaining:
                    if v in avail[last]:
                        opts = 0
                        for w in avail[v]:
                            if w != last and w in remaining:
                                opts += 1
                        candidates.append((opts, v))
                
                if not candidates:
                    return None
                
                # Sort by constraints, then shuffle for diversity
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 3:
                    # Take top few most constrained, shuffle them
                    top_k = min(3, len(candidates))
                    top = candidates[:top_k]
                    rng.shuffle(top)
                    candidates = top + candidates[top_k:]
                
                for _, v in candidates:
                    if time.time() > deadline:
                        return None
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        path.append(v)
                        remaining.remove(v)
                        result = backtrack(path, remaining)
                        if result is not None:
                            return result
                        path.pop()
                        remaining.add(v)
                return None
            
            result = backtrack(path, remaining)
            if result is not None:
                return result
        
        return None
    
    def try_vertex_subset(vertices, deadline):
        """Try to find max edge-disjoint Hamiltonian cycles on this vertex set."""
        nonlocal best_cycles, best_count
        if time.time() > deadline:
            return
        
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return
        
        # Count edges in induced subgraph
        induced_edges = set()
        for u in vertices:
            for v in adj[u]:
                if v in vertices:
                    edge = (min(u, v), max(u, v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily
        max_attempts = min(max_possible, 5)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Try fast deterministic first
            cycle = find_hamiltonian_cycle_fast(vertices, used_edges, adj, deadline)
            if cycle is None:
                # Try with random restarts
                cycle = find_cycle_with_restarts(vertices, used_edges, adj, deadline, num_restarts=30)
            
            if cycle is None:
                break
            
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
    
    all_vertices = set(range(n))
    
    # Priority order for trying subsets: full set, then size 20, 19, etc.
    subsets_to_try = [all_vertices]
    
    # All single-vertex removals (size 20)
    for removed in range(n):
        subsets_to_try.append(all_vertices - {removed})
    
    # Some random subsets of size 19
    random.seed(42)
    for _ in range(20):
        subset = set(random.sample(range(n), 19))
        subsets_to_try.append(subset)
    
    # Some random subsets of size 18
    for _ in range(20):
        subset = set(random.sample(range(n), 18))
        subsets_to_try.append(subset)
    
    # Try each subset
    for subset in subsets_to_try:
        if time.time() > deadline:
            break
        try_vertex_subset(subset, deadline)
    
    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: try to find any cycle on full vertex set
        cycle = find_hamiltonian_cycle_fast(all_vertices, set(), adj, deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        # Last resort: return a simple cycle if we can find one
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Try to find a cycle through u-v
                    pass
        # Absolute fallback
        return [[0, 1, 2]]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
