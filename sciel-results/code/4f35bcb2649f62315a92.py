import time
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(edges):
        degree = {}
        for u, v in edges:
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        return [v for v in degree if degree[v] % 2 == 1]

    def find_eulerian_trail(adj, start, end=None):
        """
        Find Eulerian trail from start to end (if end is None, start==end for circuit).
        Returns list of vertices.
        """
        if start is None:
            return []
        
        circuit = []
        stack = [start]
        visited_edges = set()
        
        while stack:
            v = stack[-1]
            # Find an unvisited edge
            next_v = None
            # To ensure we find the trail correctly, we should iterate through neighbors
            # If we want to end at 'end', we should prefer edges leading to 'end' only if we are at the end of the trail?
            # Hierholzer's algorithm doesn't naturally support fixed endpoints unless we add a dummy edge.
            # We assume adj includes the dummy edge if we added one.
            
            for u in adj[v]:
                e = tuple(sorted((v, u)))
                if e not in visited_edges:
                    next_v = u
                    visited_edges.add(e)
                    break
            
            if next_v is not None:
                stack.append(next_v)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def is_simple_path(path):
        return len(set(path)) == len(path)

    def path_cover_eulerian(edges, n, rng):
        if not edges:
            return []
        
        odd_verts = get_odd_vertices(edges)
        
        if not odd_verts:
            # Add dummy edge
            # Pick two distinct vertices
            u, v = 0, 1
            while u == v and n > 1:
                u = rng.randint(0, n-1)
                v = rng.randint(0, n-1)
            if n == 1:
                return []
            dummy_edge = tuple(sorted((u, v)))
            augmented_edges = set(edges) | {dummy_edge}
            pairs = [(u, v)]
        else:
            # Pair up odd vertices
            # Try a random permutation of odd vertices and pair consecutive ones
            # We can try a few different pairings
            augmented_edges = set(edges)
            pairs = []
            # Shuffle odd_verts
            shuffled_odd = odd_verts[:]
            rng.shuffle(shuffled_odd)
            for i in range(0, len(shuffled_odd), 2):
                u, v = shuffled_odd[i], shuffled_odd[i+1]
                pairs.append((u, v))
                augmented_edges.add(tuple(sorted((u, v))))

        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in augmented_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find Eulerian circuit in augmented graph
        # Start at one of the paired vertices
        start = pairs[0][0]
        circuit = find_eulerian_trail(adj, start)
        
        if not circuit:
            return None

        # The circuit is a closed loop.
        # We need to split it at the dummy edges.
        dummy_edges_set = {tuple(sorted((u, v))) for u, v in pairs}
        
        # Extract paths
        paths = []
        current_path = [circuit[0]]
        
        # Iterate through the circuit edges
        # The circuit is v0, v1, ..., vk, v0
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i+1]
            e = tuple(sorted((u, v)))
            if e in dummy_edges_set:
                # End current path
                if len(current_path) > 1:
                    paths.append(current_path)
                # Start new path
                current_path = [v]
            else:
                current_path.append(v)
        
        # Handle the wrap-around edge (last to first)
        u, v = circuit[-1], circuit[0]
        e = tuple(sorted((u, v)))
        if e in dummy_edges_set:
            if len(current_path) > 1:
                paths.append(current_path)
        else:
            # The path continues from the start?
            # This implies the segment containing circuit[0] and circuit[-1] is connected.
            # Since we started with current_path = [circuit[0]], and appended vertices,
            # if the last edge is not a dummy, current_path ends at circuit[-1].
            # But wait, if the first edge (circuit[0], circuit[1]) was a dummy, 
            # we started a new path at circuit[1].
            # So the segment [circuit[0]] was discarded (len 1).
            # The segment ending at circuit[-1] is current_path.
            # If the last edge is NOT a dummy, then circuit[-1] is connected to circuit[0].
            # But circuit[0] is the start of the *previous* segment which was cut?
            # Let's trace:
            # Circuit: A - B - C - A
            # Dummy: A-B
            # i=0: A-B is dummy. current_path=[A] -> len 1, discard. current_path=[B].
            # i=1: B-C not dummy. current_path=[B, C].
            # Last: C-A not dummy. current_path=[B, C, A].
            # This is a valid path B-C-A.
            # Correct.
            
            if len(current_path) > 1:
                paths.append(current_path)
        
        # Verify paths are simple and cover all edges
        if not paths:
            return None
            
        covered = set()
        for p in paths:
            if not is_simple_path(p):
                return None
            for i in range(len(p) - 1):
                covered.add(tuple(sorted((p[i], p[i+1]))))
                
        if covered != set(edges):
            return None
            
        return paths

    def path_cover_greedy_merge(edges, n, rng):
        remaining_edges = set(edges)
        paths = []
        
        # Greedy construction
        while remaining_edges:
            # Find a vertex with max degree in remaining graph
            deg = {}
            for u, v in remaining_edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
                
            # Pick vertex with max degree
            start = max(deg, key=lambda x: deg[x])
            
            # Build adjacency for remaining edges
            sub_adj = {i: set() for i in range(n)}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                candidates = [nb for nb in sub_adj[current] if nb not in visited]
                if not candidates:
                    break
                # Warnsdorff's rule: choose neighbor with min degree in remaining graph
                # We need degrees in the *remaining* graph, not just sub_adj size?
                # sub_adj size is the degree in the remaining graph.
                best_next = min(candidates, key=lambda nb: len(sub_adj[nb]))
                
                path.append(best_next)
                visited.add(best_next)
                
                # Remove edge
                e = tuple(sorted((current, best_next)))
                remaining_edges.discard(e)
                sub_adj[current].discard(best_next)
                sub_adj[best_next].discard(current)
                
                current = best_next
            
            if len(path) > 1:
                paths.append(path)
        
        # Merging phase
        # Try to merge paths that share endpoints
        merged = True
        while merged:
            merged = False
            new_paths = []
            used = [False] * len(paths)
            
            for i in range(len(paths)):
                if used[i]:
                    continue
                p1 = paths[i]
                
                for j in range(i + 1, len(paths)):
                    if used[j]:
                        continue
                    p2 = paths[j]
                    
                    # Check if p1 and p2 can be merged
                    # Endpoints of p1: p1[0], p1[-1]
                    # Endpoints of p2: p2[0], p2[-1]
                    
                    # Possible merges:
                    # 1. p1[-1] == p2[0] -> p1 + p2[1:]
                    # 2. p1[-1] == p2[-1] -> p1 + p2[-2::-1]
                    # 3. p1[0] == p2[0] -> p2[1:] + p1
                    # 4. p1[0] == p2[-1] -> p2[::-1] + p1[1:]
                    
                    new_p = None
                    if p1[-1] == p2[0]:
                        candidate = p1 + p2[1:]
                        if is_simple_path(candidate):
                            new_p = candidate
                    elif p1[-1] == p2[-1]:
                        candidate = p1 + p2[-2::-1]
                        if is_simple_path(candidate):
                            new_p = candidate
                    elif p1[0] == p2[0]:
                        candidate = p2[1:] + p1
                        if is_simple_path(candidate):
                            new_p = candidate
                    elif p1[0] == p2[-1]:
                        candidate = p2[::-1] + p1[1:]
                        if is_simple_path(candidate):
                            new_p = candidate
                    
                    if new_p:
                        new_paths.append(new_p)
                        used[i] = True
                        used[j] = True
                        merged = True
                        break # Restart merging with new list? Or just mark and continue?
                        # Breaking inner loop, but we need to clear new_paths and restart outer?
                        # Simpler: just add to a list of merges and apply later?
                        # Or, replace paths[i] with new_p, mark j as used, and continue.
                        # But changing list size while iterating is tricky.
                        # Let's just collect all merges and apply them in a batch.
                        # Actually, greedy merging one by one is fine if we restart.
                        pass 
                
                if not used[i]:
                    new_paths.append(p1)
            
            if merged:
                paths = new_paths
            else:
                break
                
        return paths

    results = []
    rng = Random(42)
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Fallback: just return edges as paths (valid but high score)
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        best_paths = None
        
        # Try Eulerian with a few random pairings
        for _ in range(10):
            if time.time() > DEADLINE:
                break
            paths = path_cover_eulerian(edges, n, rng)
            if paths:
                if best_paths is None or len(paths) < len(best_paths):
                    best_paths = paths
                    if len(best_paths) == len(get_odd_vertices(edges)) // 2:
                        break # Optimal for trails, likely optimal for paths
        
        # If Eulerian failed or didn't find optimal, use Greedy+Merge
        if best_paths is None or (best_paths and len(best_paths) > len(get_odd_vertices(edges)) // 2):
            greedy_paths = path_cover_greedy_merge(edges, n, rng)
            if best_paths is None or len(greedy_paths) < len(best_paths):
                best_paths = greedy_paths
        
        if best_paths is None:
            # Final fallback
            best_paths = [list(e) for e in edges]
            
        results.append(best_paths)
    
    return results
