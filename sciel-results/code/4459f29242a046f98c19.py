import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edges, n):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_simple(path):
        return len(set(path)) == len(path)

    def try_merge(paths, i, j):
        """Try to merge paths[i] and paths[j] by connecting endpoints."""
        P1 = list(paths[i])
        P2 = list(paths[j])
        
        # P1 end matches P2 start
        if P1[-1] == P2[0]:
            merged = P1 + P2[1:]
            if is_simple(merged):
                return merged
        
        # P1 end matches P2 end (reverse P2)
        if P1[-1] == P2[-1]:
            P2r = P2[::-1]
            merged = P1 + P2r[1:]
            if is_simple(merged):
                return merged
        
        # P1 start matches P2 start (reverse P1)
        if P1[0] == P2[0]:
            P1r = P1[::-1]
            merged = P1r + P2[1:]
            if is_simple(merged):
                return merged
        
        # P1 start matches P2 end (reverse P1)
        if P1[0] == P2[-1]:
            P1r = P1[::-1]
            merged = P1r + P2[1:]
            if is_simple(merged):
                return merged

        return None

    def improve_by_merge(paths, max_iters=1000):
        """Repeatedly try to merge any two paths until no more merges possible."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            n = len(paths)
            for i in range(n):
                for j in range(i + 1, n):
                    merged = try_merge(paths, i, j)
                    if merged is not None:
                        paths[i] = merged
                        paths.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return paths

    def greedy_path_cover(edges, adj, n, rng):
        """
        Greedy path cover with degree-based heuristics.
        """
        remaining = set(edges)
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        paths = []
        while remaining:
            candidates = [v for v in range(n) if deg.get(v, 0) > 0]
            if not candidates:
                break
            
            # Prefer odd-degree vertices, then leaves, then low degree
            odd = [v for v in candidates if deg[v] % 2 == 1]
            leaves = [v for v in candidates if deg[v] == 1]
            
            if odd and rng.random() < 0.7:
                start = rng.choice(odd)
            elif leaves and rng.random() < 0.5:
                start = rng.choice(leaves)
            else:
                min_d = min(deg[v] for v in candidates)
                low = [v for v in candidates if deg[v] == min_d]
                start = rng.choice(low)
            
            path = [start]
            in_path = {start}
            
            while True:
                v = path[-1]
                nbrs = []
                for u in adj[v]:
                    e = tuple(sorted((v, u)))
                    if e in remaining:
                        nbrs.append(u)
                if not nbrs:
                    break
                
                # Prefer unvisited neighbors with minimum degree (Warnsdorff)
                unvis = [u for u in nbrs if u not in in_path]
                if unvis:
                    min_d = min(deg[u] for u in unvis)
                    best = [u for u in unvis if deg[u] == min_d]
                    nxt = rng.choice(best)
                else:
                    break
                
                e = tuple(sorted((v, nxt)))
                remaining.discard(e)
                deg[v] -= 1
                deg[nxt] -= 1
                if deg[v] <= 0:
                    del deg[v]
                if deg[nxt] <= 0:
                    del deg[nxt]
                
                path.append(nxt)
                in_path.add(nxt)
            
            if len(path) >= 2:
                paths.append(path)
        
        # Add any remaining edges as single-edge paths
        for e in remaining:
            paths.append([e[0], e[1]])
        
        return paths

    def eulerian_decompose(edges, adj, n, rng):
        """
        Eulerian decomposition: pair odd vertices, find Eulerian circuit,
        split at dummy edges to get trails, then fix simplicity.
        """
        remaining = set(edges)
        if not remaining:
            return []
        
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        odd = [v for v in range(n) if deg.get(v, 0) % 2 == 1]
        n_odd = len(odd)
        
        dummy_edges = []
        if n_odd > 0:
            rng.shuffle(odd)
            for i in range(0, n_odd, 2):
                if i + 1 < n_odd:
                    dummy_edges.append((odd[i], odd[i+1]))
        
        aug_edges = list(remaining) + [(a, b) for a, b in dummy_edges]
        
        # Build edge index for Hierholzer's algorithm
        edge_index = defaultdict(list)
        for idx, (u, v) in enumerate(aug_edges):
            edge_index[u].append((v, idx))
            edge_index[v].append((u, idx))
        
        circuit = []
        stack = []
        start_v = aug_edges[0][0]
        stack.append(start_v)
        used_edges = set()
        
        while stack:
            v = stack[-1]
            found = False
            while edge_index.get(v):
                nb, idx = edge_index[v].pop()
                if idx not in used_edges:
                    used_edges.add(idx)
                    stack.append(nb)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        
        dummy_set = {tuple(sorted((a, b))) for a, b in dummy_edges}
        
        paths = []
        current_path = [circuit[0]]
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            e = tuple(sorted((u, v)))
            if e in dummy_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = [v]
            else:
                current_path.append(v)
        
        if len(current_path) >= 2:
            paths.append(current_path)
        
        # Fix simplicity by splitting at repeated vertices
        fixed_paths = []
        for trail in paths:
            if is_simple(trail):
                fixed_paths.append(trail)
            else:
                # Split at first repeated vertex
                seen = {}
                split_idx = None
                for i, v in enumerate(trail):
                    if v in seen:
                        split_idx = i
                        break
                    seen[v] = i
                
                if split_idx is not None:
                    part1 = trail[:split_idx+1]
                    part2 = trail[split_idx:]
                    for p in [part1, part2]:
                        if len(p) >= 2:
                            fixed_paths.append(p)
                else:
                    fixed_paths.append(trail)
        
        return fixed_paths

    def solve_one(edges, n, time_limit=35.0):
        """Solve one graph instance, trying multiple strategies."""
        adj = build_adj(edges, n)
        best = None
        best_count = float('inf')
        t0 = time.time()
        
        # Strategy 1: Greedy with multiple seeds
        for trial in range(2000):
            if time.time() - t0 > time_limit - 2.0:
                break
            seed = trial * 7919 + 13
            rng = random.Random(seed)
            paths = greedy_path_cover(edges, adj, n, rng)
            
            # Improve by merging
            paths = improve_by_merge(paths, max_iters=200)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best = [list(p) for p in paths]
        
        # Strategy 2: Eulerian decomposition with multiple pairings
        for trial in range(500):
            if time.time() - t0 > time_limit - 1.0:
                break
            seed = trial * 104729 + 7
            rng = random.Random(seed)
            paths = eulerian_decompose(edges, adj, n, rng)
            
            # Improve by merging
            paths = improve_by_merge(paths, max_iters=200)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best = [list(p) for p in paths]
        
        # Final improvement pass on best solution
        if best is not None:
            best = improve_by_merge(best, max_iters=500)
            best_count = len(best)
        
        # Fallback: single edge paths
        if best is None:
            best = [[u, v] for u, v in edges]
        
        return best

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time per instance
    total_time = deadline - start_time
    time_per_instance = total_time / 3 - 2.0  # 2s buffer
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        if not edges:
            results.append([])
            continue
        
        remaining_time = deadline - time.time() - 1.0
        if remaining_time < 1.0:
            # Emergency fallback
            results.append([[u, v] for u, v in edges])
            continue
        
        t_start = time.time()
        paths = solve_one(edges, n, time_limit=min(time_per_instance, remaining_time - 1.0))
        results.append(paths)
    
    return results
