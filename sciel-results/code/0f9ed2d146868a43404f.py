"""Combined program: multi-strategy greedy + local search for path cover."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        if not paths:
            return len(edges) == 0
        edge_set = set(tuple(sorted(e)) for e in edges)
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                return False
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        return used_edges == edge_set

    def greedy_path_cover(edges, strategy, time_limit):
        """
        Greedy path cover with different strategies.
        strategy: 0=degree_desc, 1=degree_asc, 2=random, 3=weighted
        """
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        rem_deg = {i: 0 for i in range(n)}
        for u, v in remaining:
            rem_deg[u] += 1
            rem_deg[v] += 1
        
        while remaining and time.time() < time_limit:
            # Find start vertex
            candidates = [v for v in range(n) if rem_deg[v] > 0]
            if not candidates:
                break
            
            if strategy == 0:  # degree descending
                start = max(candidates, key=lambda v: rem_deg[v])
            elif strategy == 1:  # degree ascending
                start = min(candidates, key=lambda v: rem_deg[v])
            elif strategy == 2:  # random
                start = random.choice(candidates)
            else:  # weighted: prefer high degree but with some randomness
                weights = [rem_deg[v] ** 2 for v in candidates]
                total = sum(weights)
                r = random.random() * total
                cum = 0
                start = candidates[-1]
                for v in candidates:
                    cum += weights[candidates.index(v)]
                    if r <= cum:
                        start = v
                        break
            
            path = [start]
            used_vertices = {start}
            current = start
            
            while True:
                candidates_next = []
                for neighbor in adj[current]:
                    e = tuple(sorted((current, neighbor)))
                    if e in remaining and neighbor not in used_vertices:
                        candidates_next.append(neighbor)
                
                if not candidates_next:
                    break
                
                if strategy == 0:  # degree descending
                    next_v = max(candidates_next, key=lambda v: rem_deg[v])
                elif strategy == 1:  # degree ascending
                    next_v = min(candidates_next, key=lambda v: rem_deg[v])
                elif strategy == 2:  # random
                    next_v = random.choice(candidates_next)
                else:  # weighted
                    weights = [rem_deg[v] ** 2 for v in candidates_next]
                    total = sum(weights)
                    r = random.random() * total
                    cum = 0
                    next_v = candidates_next[-1]
                    for v in candidates_next:
                        cum += weights[candidates_next.index(v)]
                        if r <= cum:
                            next_v = v
                            break
                
                path.append(next_v)
                used_vertices.add(next_v)
                
                e = tuple(sorted((current, next_v)))
                remaining.discard(e)
                rem_deg[current] -= 1
                rem_deg[next_v] -= 1
                
                current = next_v
            
            if len(path) >= 2:
                paths.append(path)
        
        # Handle any remaining edges as individual paths
        for u, v in remaining:
            paths.append([u, v])
        
        return paths

    def local_search_improve(paths, edges, time_limit, max_iters=200):
        """
        Try to merge paths or rearrange to reduce path count.
        Strategy: try to extend paths by finding alternative routings.
        """
        if not paths:
            return paths
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        
        # Build adjacency from edges
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = paths[:]
        best_count = len(paths)
        
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            improved = False
            
            # Try to merge two paths by finding a valid reconnection
            for i in range(len(best_paths)):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, len(best_paths)):
                    if time.time() > time_limit:
                        break
                    
                    path_i = best_paths[i]
                    path_j = best_paths[j]
                    
                    if len(path_i) < 2 or len(path_j) < 2:
                        continue
                    
                    # Try all possible merges:
                    # end of path_i -> start of path_j, or
                    # end of path_j -> start of path_i
                    
                    merges = []
                    # Merge i's end to j's start
                    if len(path_i) >= 2 and len(path_j) >= 2:
                        end_i = path_i[-1]
                        start_j = path_j[0]
                        if (min(end_i, start_j), max(end_i, start_j)) in edge_set:
                            # Check if this edge is not already used in either path
                            used_in_i = set()
                            for k in range(len(path_i) - 1):
                                used_in_i.add(tuple(sorted((path_i[k], path_i[k+1]))))
                            used_in_j = set()
                            for k in range(len(path_j) - 1):
                                used_in_j.add(tuple(sorted((path_j[k], path_j[k+1]))))
                            merge_edge = tuple(sorted((end_i, start_j)))
                            if merge_edge not in used_in_i and merge_edge not in used_in_j:
                                merged = path_i + path_j[1:]
                                if len(set(merged)) == len(merged):  # simple path
                                    merges.append(merged)
                    
                    # Merge j's end to i's start
                    if len(path_j) >= 2 and len(path_i) >= 2:
                        end_j = path_j[-1]
                        start_i = path_i[0]
                        if (min(end_j, start_i), max(end_j, start_i)) in edge_set:
                            used_in_i = set()
                            for k in range(len(path_i) - 1):
                                used_in_i.add(tuple(sorted((path_i[k], path_i[k+1]))))
                            used_in_j = set()
                            for k in range(len(path_j) - 1):
                                used_in_j.add(tuple(sorted((path_j[k], path_j[k+1]))))
                            merge_edge = tuple(sorted((end_j, start_i)))
                            if merge_edge not in used_in_i and merge_edge not in used_in_j:
                                merged = path_j + path_i[1:]
                                if len(set(merged)) == len(merged):  # simple path
                                    merges.append(merged)
                    
                    # Try reversing paths
                    if len(path_i) >= 2 and len(path_j) >= 2:
                        # end of reversed i -> start of j
                        start_i_rev = path_i[-1]
                        end_i_rev = path_i[0]
                        start_j = path_j[0]
                        if (min(start_i_rev, start_j), max(start_i_rev, start_j)) in edge_set:
                            used_in_i = set()
                            for k in range(len(path_i) - 1):
                                used_in_i.add(tuple(sorted((path_i[k], path_i[k+1]))))
                            used_in_j = set()
                            for k in range(len(path_j) - 1):
                                used_in_j.add(tuple(sorted((path_j[k], path_j[k+1]))))
                            merge_edge = tuple(sorted((start_i_rev, start_j)))
                            if merge_edge not in used_in_i and merge_edge not in used_in_j:
                                merged = list(reversed(path_i)) + path_j[1:]
                                if len(set(merged)) == len(merged):
                                    merges.append(merged)
                    
                    # end of j -> start of reversed i
                    if len(path_j) >= 2 and len(path_i) >= 2:
                        end_j = path_j[-1]
                        start_i_rev = path_i[-1]
                        if (min(end_j, start_i_rev), max(end_j, start_i_rev)) in edge_set:
                            used_in_i = set()
                            for k in range(len(path_i) - 1):
                                used_in_i.add(tuple(sorted((path_i[k], path_i[k+1]))))
                            used_in_j = set()
                            for k in range(len(path_j) - 1):
                                used_in_j.add(tuple(sorted((path_j[k], path_j[k+1]))))
                            merge_edge = tuple(sorted((end_j, start_i_rev)))
                            if merge_edge not in used_in_i and merge_edge not in used_in_j:
                                merged = path_j + list(reversed(path_i))[1:]
                                if len(set(merged)) == len(merged):
                                    merges.append(merged)
                    
                    if merges:
                        # Try the first valid merge
                        new_paths = best_paths[:]
                        new_paths.pop(j)
                        new_paths.pop(i)
                        new_paths.append(merges[0])
                        
                        if validate_paths(new_paths, edges):
                            best_paths = new_paths
                            best_count = len(best_paths)
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_paths

    def solve_one_instance(n, p, seed, time_budget):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        best_paths = None
        best_count = len(edges) + 1  # worst case: each edge is a path
        
        # Try multiple strategies
        strategies = [0, 1, 2, 3]
        rng = random.Random(seed * 1000 + n)
        
        for strategy in strategies:
            if time.time() > t0 + time_budget:
                break
            
            # Try multiple times with randomness
            for trial in range(3):
                if time.time() > t0 + time_budget:
                    break
                
                random.seed(seed * 10000 + strategy * 100 + trial)
                paths = greedy_path_cover(edges, strategy, time.time())
                
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths[:]
        
        # Apply local search to improve
        if best_paths is not None:
            remaining_time = t0 + time_budget - time.time()
            if remaining_time > 5:
                improved = local_search_improve(
                    best_paths, edges, 
                    time.time() + remaining_time * 0.5,
                    max_iters=100
                )
                if validate_paths(improved, edges) and len(improved) < best_count:
                    best_paths = improved
                    best_count = len(improved)
        
        if best_paths is None:
            best_paths = [[u, v] for u, v in edges]
        
        return best_paths

    # Solve each instance with allocated time
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for idx, spec in enumerate(specs):
        elapsed = time.time() - t0
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        # Allocate time proportionally, with more time for larger graphs
        if idx == 0:
            time_budget = min(remaining_total * 0.3, 40)
        elif idx == 1:
            time_budget = min(remaining_total * 0.35, 50)
        else:
            time_budget = min(remaining_total * 0.35, 60)
        
        paths = solve_one_instance(spec[0], spec[1], spec[2], time_budget)
        results.append(paths)
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        spec = specs[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results[:3]
# EVOLVE-BLOCK-END
