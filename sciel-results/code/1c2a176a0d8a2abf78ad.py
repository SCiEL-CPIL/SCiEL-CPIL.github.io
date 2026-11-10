import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 145.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _bfs_path(adj, s, t, forbidden):
        """Shortest simple path from s to t, internal vertices not in forbidden."""
        if s == t:
            return [s]
        if s in forbidden or t in forbidden:
            return None
        prev = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        # Check if internal vertices are forbidden
        if len(path) >= 3 and set(path[1:-1]) & forbidden:
            return None
        return path

    def _pair_odd_vertices_improved(adj, odd_vertices, max_restarts=200):
        """
        Find vertex-disjoint paths pairing up all odd vertices.
        Returns a list of paths (vertex lists) or None if failed.
        Uses improved strategy with backtracking.
        """
        if not odd_vertices:
            return []
        
        best_result = None
        best_len = float('inf')
        
        for attempt in range(max_restarts):
            remaining = odd_vertices[:]
            random.shuffle(remaining)
            used_internal = set()
            pair_paths = []
            success = True
            
            while len(remaining) >= 2:
                # Try all pairs, prefer shorter paths
                candidates = []
                for i in range(len(remaining)):
                    for j in range(i + 1, len(remaining)):
                        s, t = remaining[i], remaining[j]
                        p = _bfs_path(adj, s, t, used_internal)
                        if p is not None:
                            candidates.append((len(p), i, j, p))
                
                if not candidates:
                    success = False
                    break
                
                # Sort by path length (prefer shorter)
                candidates.sort(key=lambda x: x[0])
                
                # Try the best candidate first, with some randomness
                if len(candidates) > 1 and random.random() < 0.3:
                    # Occasionally try a random candidate
                    idx = random.randint(0, min(3, len(candidates) - 1))
                    chosen = candidates[idx]
                else:
                    chosen = candidates[0]
                
                _, best_i, best_j, best_path = chosen
                
                # Commit this path
                used_internal |= set(best_path[1:-1])
                pair_paths.append(best_path)
                
                # Remove the endpoints from remaining
                new_remaining = []
                for idx2, v in enumerate(remaining):
                    if idx2 == best_i or idx2 == best_j:
                        continue
                    new_remaining.append(v)
                remaining = new_remaining
            
            if success:
                total_len = sum(len(p) for p in pair_paths)
                if total_len < best_len:
                    best_len = total_len
                    best_result = pair_paths
                if len(pair_paths) == len(odd_vertices) // 2:
                    # Optimal number of paths, return immediately
                    return pair_paths
        
        return best_result

    def _solve_one_gallai(edges, n):
        if not edges:
            return []
            
        # Build adjacency for original graph
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Find odd vertices
        odd_vertices = [v for v in range(n) if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Eulerian graph: need to find a Hamiltonian path or decompose
            # Try to find a Hamiltonian path
            for start in range(n):
                # Try DFS to find Hamiltonian path
                visited = set()
                path = [start]
                
                def dfs(v):
                    if len(path) == n:
                        return True
                    for nb in adj[v]:
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            if dfs(nb):
                                return True
                            path.pop()
                            visited.discard(nb)
                    return False
                
                if dfs(start):
                    return [path]
            
            # If no Hamiltonian path found, use Hierholzer and split
            # For now, return a single trail (may not be simple)
            hadj = {i: list(adj[i]) for i in range(n)}
            circuit = []
            stack = [0]
            while stack:
                v = stack[-1]
                if hadj[v]:
                    u = hadj[v].pop()
                    hadj[u].remove(v)
                    stack.append(u)
                else:
                    circuit.append(stack.pop())
            circuit.reverse()
            
            # Try to split into simple paths
            paths = []
            current = [circuit[0]]
            seen = {circuit[0]}
            for i in range(1, len(circuit)):
                v = circuit[i]
                if v in seen:
                    # Split here
                    if len(current) >= 2:
                        paths.append(current)
                    current = [v]
                    seen = {v}
                else:
                    current.append(v)
                    seen.add(v)
            if len(current) >= 2:
                paths.append(current)
            
            return paths if paths else [circuit]
            
        # Pair odd vertices using improved algorithm
        pair_paths = _pair_odd_vertices_improved(adj, odd_vertices)
        
        if pair_paths is None:
            # Fallback to greedy if pairing fails
            return _greedy_decompose(edges, n)
            
        # Build multigraph H = G + pair_paths
        edge_list = []
        for u, v in edges:
            edge_list.append((u, v, False))
            
        for pp in pair_paths:
            for i in range(len(pp) - 1):
                u, v = pp[i], pp[i+1]
                edge_list.append((u, v, True))
                
        # Build adjacency for Hierholzer
        adj_h = {i: [] for i in range(n)}
        for idx, (u, v, is_added) in enumerate(edge_list):
            adj_h[u].append((v, idx))
            adj_h[v].append((u, idx))
            
        # Hierholzer
        used_edge = [False] * len(edge_list)
        stack = [0]
        circuit_vertices = []
        edge_seq = []
        
        while stack:
            v = stack[-1]
            advanced = False
            while adj_h[v]:
                u, idx = adj_h[v].pop()
                if not used_edge[idx]:
                    used_edge[idx] = True
                    stack.append(u)
                    edge_seq.append(idx)
                    advanced = True
                    break
            if not advanced:
                circuit_vertices.append(stack.pop())
                
        circuit_vertices.reverse()
        edge_seq.reverse()
        
        # Extract paths by cutting at added edges
        paths = []
        current_path = []
        
        for i in range(len(edge_seq)):
            idx = edge_seq[i]
            is_added = edge_list[idx][2]
            u = circuit_vertices[i]
            v = circuit_vertices[i+1]
            
            if is_added:
                if current_path:
                    paths.append(current_path)
                    current_path = []
                current_path = [v]
            else:
                if not current_path:
                    current_path = [u]
                current_path.append(v)
                
        if current_path:
            paths.append(current_path)
        
        paths = [p for p in paths if len(p) >= 2]
        
        if not paths:
            return _greedy_decompose(edges, n)
            
        return paths

    def _greedy_decompose(edges, n, max_time=8.0):
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        edge_set = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def has_edge(u, v, eset):
            return (u, v) in eset or (v, u) in eset

        def find_longest_path(start, eset, adj_local, visited_set):
            path = [start]
            visited = set(visited_set) | {start}
            current = start
            
            while True:
                available = []
                for nb in adj_local[current]:
                    if nb not in visited and has_edge(current, nb, eset):
                        available.append(nb)
                
                if not available:
                    break
                
                # Choose neighbor with minimum degree (Warnsdorff's rule)
                min_deg = float('inf')
                best_nb = None
                for nb in available:
                    deg = 0
                    for x in adj_local[nb]:
                        if x not in visited and has_edge(nb, x, eset):
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_nb = nb
                
                # Add some randomness
                if random.random() < 0.15 and len(available) > 1:
                    best_nb = random.choice(available)
                
                u, v = current, best_nb
                if (u, v) in eset:
                    eset.discard((u, v))
                else:
                    eset.discard((v, u))
                
                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb
                
            return path

        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 3.0:
            remaining = set(edge_set)
            paths = []
            
            deg = {i: 0 for i in range(n)}
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
            
            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                
                if rem_odd and random.random() < 0.85:
                    start_v = random.choice(rem_odd)
                else:
                    active = [v for v in rem_deg if rem_deg[v] > 0]
                    if not active:
                        break
                    start_v = random.choice(active)
                
                path = find_longest_path(start_v, remaining, adj, set())
                
                if not path or len(path) < 2:
                    found_edge = False
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                    if not found_edge:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if (u, v) in remaining:
                            remaining.discard((u, v))
                        else:
                            remaining.discard((v, u))
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    def _local_improve(paths, edges, n, max_iters=50):
        """
        Try to merge paths or reduce count via local search.
        If two paths P1, P2 can be connected by removing an edge from one and adding two edges,
        or if we can swap endpoints, etc.
        For simplicity, we try to re-decompose small clusters or just validate.
        A better heuristic: if we have a path A-B and a path C-D, and edge A-C and B-D exist,
        we can form A-C ... D-B? No, that's complex.
        
        Let's try a simpler improvement: 
        Check if any path can be extended by an edge from another path (breaking it).
        This is risky.
        
        Instead, let's just ensure the greedy decomposition is as good as possible by
        running it with slightly different parameters if time permits.
        But the prompt asks for a coherent program combining ideas.
        The main improvement from the "2-matching" idea is that we want to maximize
        the number of edges in the paths, which is equivalent to minimizing the number
        of paths (since total edges is fixed).
        
        Actually, the current greedy already tries to find long paths.
        Let's add a post-processing step that tries to merge paths if possible.
        """
        # This is a placeholder for potential local search improvements
        # Given time constraints, we'll rely on the robust greedy + gallai hybrid.
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = min(12.0, _remaining() - 2.0)
        if max_t < 1.0:
            max_t = 1.0
            
        # Try Gallai first
        paths = _solve_one_gallai(edges, n)
        
        # Verify paths
        valid = True
        used_edges = set()
        for path in paths:
            if len(path) < 2:
                valid = False
                break
            if len(set(path)) != len(path):
                valid = False
                break
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break
                
        if not valid or len(used_edges) != len(edges):
            # Fallback to Greedy
            paths = _greedy_decompose(edges, n, max_time=max_t)
        else:
            # If Gallai worked, try to see if greedy can do better (unlikely but possible)
            # Or just keep Gallai as it is structurally sound.
            # Let's stick with Gallai if valid.
            pass

        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
