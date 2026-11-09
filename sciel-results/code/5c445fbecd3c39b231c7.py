import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_initial_decomposition(edges, seed):
        """
        Greedy construction: repeatedly find the longest available path.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        while remaining:
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
                
            # Start from a high-degree node
            start = max(deg.keys(), key=lambda x: (deg[x], rng.random()))
            
            path = [start]
            visited = {start}
            used_edges = set()
            
            while True:
                current = path[-1]
                candidates = []
                for v in adj[current]:
                    e = (min(current, v), max(current, v))
                    if e in remaining and v not in visited:
                        candidates.append(v)
                
                if not candidates:
                    break
                
                # Heuristic: pick neighbor with most unvisited neighbors
                best_next = None
                max_future = -1
                for v in candidates:
                    future = 0
                    for w in adj[v]:
                        e2 = (min(v, w), max(v, w))
                        if e2 in remaining and w not in visited:
                            future += 1
                    if future > max_future:
                        max_future = future
                        best_next = v
                
                if best_next is None:
                    break
                    
                path.append(best_next)
                visited.add(best_next)
                e = (min(current, best_next), max(current, best_next))
                used_edges.add(e)
                current = best_next
                
            if len(path) > 1:
                paths.append(path)
                for e in used_edges:
                    remaining.discard(e)
            else:
                if remaining:
                    e = next(iter(remaining))
                    paths.append(list(e))
                    remaining.discard(e)
                    
        return paths

    def decompose_via_sa(edges, seed, time_budget):
        """
        Simulated Annealing on the set of paths.
        State: List of paths (each path is a list of vertices).
        Objective: Minimize len(paths).
        Moves: 
        1. Merge two paths if possible (reduces count by 1).
        2. Split a path into two (increases count by 1) - allows escaping local minima.
        3. Swap segments between paths (maintains count, changes structure).
        """
        
        # Precompute adjacency for fast edge checking
        adj = defaultdict(set)
        edge_set = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def is_edge(u, v):
            return (min(u, v), max(u, v)) in edge_set

        def paths_to_edges(paths):
            """Verify that the paths cover exactly the edge_set and are simple."""
            covered = set()
            for path in paths:
                if len(path) < 2:
                    continue
                # Check simplicity
                if len(set(path)) != len(path):
                    return None
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if not is_edge(u, v):
                        return None
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return None
                    covered.add(e)
            if covered != edge_set:
                return None
            return covered

        def merge_paths(paths):
            """Try to merge any two paths. Returns new list or None."""
            n = len(paths)
            for i in range(n):
                p1 = paths[i]
                if len(p1) < 2: continue
                s1, e1 = p1[0], p1[-1]
                set1 = set(p1)
                
                for j in range(n):
                    if i == j: continue
                    p2 = paths[j]
                    if len(p2) < 2: continue
                    s2, e2 = p2[0], p2[-1]
                    
                    # 4 combinations
                    # 1. e1 - s2
                    if is_edge(e1, s2) and not (set1 & set(p2[1:])):
                        new_p = p1 + p2[1:]
                        return [p for k, p in enumerate(paths) if k != i and k != j] + [new_p]
                    # 2. e1 - e2
                    if is_edge(e1, e2) and not (set1 & set(p2)):
                        new_p = p1 + p2[::-1]
                        return [p for k, p in enumerate(paths) if k != i and k != j] + [new_p]
                    # 3. s1 - s2
                    if is_edge(s1, s2) and not (set1 & set(p2)):
                        new_p = p1[::-1] + p2
                        return [p for k, p in enumerate(paths) if k != i and k != j] + [new_p]
                    # 4. s1 - e2
                    if is_edge(s1, e2) and not (set1 & set(p2)):
                        new_p = p2 + p1[::-1]
                        return [p for k, p in enumerate(paths) if k != i and k != j] + [new_p]
            return None

        def split_path(paths):
            """Split a random path at a random internal point. Increases count by 1."""
            if len(paths) == 0:
                return None
            # Pick a path with length > 2
            candidates = [i for i, p in enumerate(paths) if len(p) > 2]
            if not candidates:
                return None
            idx = random.choice(candidates)
            p = paths[idx]
            # Pick a split point (vertex index 1 to len-2)
            split_idx = random.randint(1, len(p) - 2)
            p1 = p[:split_idx + 1]
            p2 = p[split_idx:]
            new_paths = [p for k, p in enumerate(paths) if k != idx]
            new_paths.append(p1)
            new_paths.append(p2)
            return new_paths

        def swap_segments(paths):
            """
            Try to swap a segment from path i with a segment from path j.
            This is a complex move. For simplicity and speed, we'll stick to 
            merge/split and a simpler 're-route' move if we can find one.
            Actually, let's implement a 'cut and paste' move:
            Take a segment from path A and try to insert it into path B?
            Or simpler: Just use merge and split. The SA dynamics with 
            merge (downhill) and split (uphill) can effectively explore the space.
            """
            # To keep it robust and fast, we'll rely on Merge and Split.
            # However, pure merge/split can be slow to find optimal structures.
            # Let's add a simple 'reverse' move? No, that doesn't change count.
            # Let's add a 'swap endpoints' move? 
            # If we have paths P1=[a,b,c] and P2=[d,e,f], and edges (c,d) and (a,f) exist,
            # we could potentially merge them into one path [a,b,c,d,e,f] if valid.
            # This is covered by merge.
            
            # What if we have P1=[a,b,c] and P2=[d,e] and edge (b,d) exists?
            # We can't merge directly. But if we split P1 at b? 
            # P1a=[a,b], P1b=[b,c]. Then merge P1a with P2? [a,b,d,e]. 
            # So split+merge is the key mechanism.
            
            return None # Placeholder, we use merge and split

        # Initialize
        best_paths = None
        best_count = float('inf')
        
        # Run SA for multiple restarts or one long run?
        # Given time budget, one long run with restarts on stagnation is good.
        
        current_paths = get_initial_decomposition(edges, seed)
        current_count = len(current_paths)
        
        # Verify initial
        if paths_to_edges(current_paths) is None:
             # Fallback to simple greedy
             current_paths = get_initial_decomposition(edges, 0)
             current_count = len(current_paths)
             
        if current_count < best_count:
            best_count = current_count
            best_paths = [p[:] for p in current_paths]
            
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 0.01
        rng = random.Random(seed + 100)
        
        start_time = time.time()
        
        while time.time() - start_time < time_budget and temperature > min_temperature:
            # Choose a move
            # Probability of split increases with temperature
            p_split = min(0.5, temperature)
            
            new_paths = None
            if rng.random() < p_split:
                new_paths = split_path(current_paths)
            else:
                new_paths = merge_paths(current_paths)
                
            if new_paths is None:
                # No valid move of this type, try the other or restart
                if p_split < 0.5:
                    new_paths = merge_paths(current_paths)
                else:
                    new_paths = split_path(current_paths)
                    
                if new_paths is None:
                    # Stuck? Restart with a slightly perturbed greedy
                    # Or just cool down
                    temperature *= cooling_rate
                    continue
            
            new_count = len(new_paths)
            delta = new_count - current_count
            
            # Acceptance criterion
            if delta < 0:
                # Better
                current_paths = new_paths
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_paths = [p[:] for p in current_paths]
            else:
                # Worse or same
                # Boltzmann factor
                # We want to accept with probability exp(-delta / T)
                # Since delta is integer >= 0, and T is small, this will be rare for large T?
                # Wait, standard SA: accept if exp(-DeltaE/T) > random()
                # Here DeltaE = delta (positive for worse).
                # If T is high, exp(0) = 1. If T is low, exp(-large) = 0.
                prob = np.exp(-delta / temperature)
                if rng.random() < prob:
                    current_paths = new_paths
                    current_count = new_count
            
            temperature *= cooling_rate
            
        return best_paths

    result = []
    # Distribute time budget among the 3 graphs
    # Graphs are small, so 40s each is plenty.
    time_per_graph = 40.0
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        
        # Run SA
        paths = decompose_via_sa(edges, spec[2], time_per_graph)
        
        # Fallback
        if paths is None:
            paths = get_initial_decomposition(edges, 0)
            
        result.append(paths)
        
    return result

# EVOLVE-BLOCK-END
