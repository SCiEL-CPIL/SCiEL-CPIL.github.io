import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n):
        """Find a long cycle using BFS-based approach with multiple starts."""
        if not adj:
            return None
        candidates = [u for u in adj if len(adj[u]) >= 2]
        if not candidates:
            return None
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        starts = candidates[:min(10, len(candidates))]
        
        best_cycle = None
        best_len = 0
        
        for start in starts:
            # BFS to find a cycle
            visited = {start: -1}
            dist = {start: 0}
            parent = {}
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if v == parent.get(u, -1):
                        continue
                    if v not in visited:
                        visited[v] = u
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent.get(u, -1) and dist.get(v, 0) < dist.get(u, 0):
                        # Found a back edge, extract cycle
                        cyc = [u]
                        cur = u
                        while cur != v:
                            cur = parent[cur]
                            cyc.append(cur)
                        cyc.reverse()
                        if len(cyc) > best_len:
                            best_len = len(cyc)
                            best_cycle = cyc
                        break
                if best_cycle and best_len >= 15:
                    break
        
        return best_cycle

    def decompose(edges):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n = max(adj.keys()) + 1 if adj else 0
        edge_set = set(edges)
        
        # Phase 1: Find maximum 2-matching using bipartite matching
        row_ind = []
        col_ind = []
        for u, v in edges:
            row_ind.append(u)
            col_ind.append(n + v)
            row_ind.append(v)
            col_ind.append(n + u)
        
        if not row_ind:
            return [[u, v] for u, v in edges]
        
        data = np.ones(len(row_ind), dtype=np.int8)
        bip_graph = csr_matrix((data, (row_ind, col_ind)), shape=(n, 2 * n))
        
        try:
            matching = maximum_bipartite_matching(bip_graph)
        except Exception:
            matching = None
        
        dir_out = defaultdict(list)
        dir_in = defaultdict(set)
        selected_edges = set()
        
        if matching is not None:
            for u in range(n):
                v = matching[u]
                if v != -1 and v >= n:
                    orig_v = v - n
                    dir_out[u].append(orig_v)
                    dir_in[orig_v].add(u)
                    selected_edges.add((min(u, orig_v), max(u, orig_v)))
        
        cycles = []
        paths = []
        visited = set()
        
        for u in range(n):
            if u in visited or not dir_out[u]:
                continue
            
            path = [u]
            visited.add(u)
            curr = u
            while dir_out[curr] and dir_out[curr][0] not in visited:
                curr = dir_out[curr][0]
                path.append(curr)
                visited.add(curr)
            
            if curr in path:
                idx = path.index(curr)
                cycle = path[idx:]
                if len(cycle) >= 3:
                    cycles.append(cycle)
            else:
                if len(path) >= 2:
                    paths.append(path)
        
        new_cycles = []
        for path in paths:
            start, end = path[0], path[-1]
            e = (min(start, end), max(start, end))
            if e in edge_set and e not in selected_edges:
                new_cycles.append(path)
                selected_edges.add(e)
        
        cycles.extend(new_cycles)
        
        remaining_edges = [e for e in edges if e not in selected_edges]
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        greedy_cycles = []
        rng = random.Random(42)
        
        while time.time() - start_time < DEADLINE - 5:
            if not any(len(neigh) >= 2 for neigh in rem_adj.values()):
                break
            
            cycle = find_longest_cycle(rem_adj, n)
            if cycle is None:
                break
            
            for j in range(len(cycle)):
                a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                rem_adj[a].discard(b)
                rem_adj[b].discard(a)
                selected_edges.add((min(a, b), max(a, b)))
            
            greedy_cycles.append(cycle)
        
        all_cycles = cycles + greedy_cycles
        
        pieces = [list(c) for c in all_cycles]
        used_edges = set()
        for c in all_cycles:
            for j in range(len(c)):
                a, b = c[j], c[(j + 1) % len(c)]
                used_edges.add((min(a, b), max(a, b)))
        
        for e in edges:
            if e not in used_edges:
                pieces.append([e[0], e[1]])
        
        # Local search: try to merge cycles that share a vertex
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() - start_time < DEADLINE - 3:
            improved = False
            iteration += 1
            
            vertex_to_pieces = defaultdict(list)
            for i, piece in enumerate(pieces):
                if len(piece) >= 3:
                    for v in piece:
                        vertex_to_pieces[v].append(i)
                else:
                    vertex_to_pieces[piece[0]].append(i)
                    vertex_to_pieces[piece[1]].append(i)
            
            for v in list(vertex_to_pieces.keys()):
                if len(vertex_to_pieces[v]) < 2:
                    continue
                
                piece_indices = vertex_to_pieces[v][:5]
                
                for i in range(len(piece_indices)):
                    for j in range(i + 1, len(piece_indices)):
                        pi, pj = piece_indices[i], piece_indices[j]
                        if pi >= len(pieces) or pj >= len(pieces):
                            continue
                        
                        piece_i = pieces[pi]
                        piece_j = pieces[pj]
                        
                        if len(piece_i) < 3 or len(piece_j) < 3:
                            continue
                        
                        edges_i = set()
                        for k in range(len(piece_i)):
                            a, b = piece_i[k], piece_i[(k + 1) % len(piece_i)]
                            edges_i.add((min(a, b), max(a, b)))
                        
                        edges_j = set()
                        for k in range(len(piece_j)):
                            a, b = piece_j[k], piece_j[(k + 1) % len(piece_j)]
                            edges_j.add((min(a, b), max(a, b)))
                        
                        if edges_i & edges_j:
                            continue
                        
                        try:
                            pos_i = piece_i.index(v)
                            pos_j = piece_j.index(v)
                            
                            next_i = piece_i[(pos_i + 1) % len(piece_i)]
                            prev_i = piece_i[(pos_i - 1) % len(piece_i)]
                            next_j = piece_j[(pos_j + 1) % len(piece_j)]
                            prev_j = piece_j[(pos_j - 1) % len(piece_j)]
                            
                            path_i = []
                            cur = next_i
                            while cur != prev_i and len(path_i) <= len(piece_i):
                                path_i.append(cur)
                                idx = piece_i.index(cur)
                                cur = piece_i[(idx + 1) % len(piece_i)]
                                if cur == v:
                                    break
                            if cur == prev_i:
                                path_i.append(prev_i)
                            
                            path_j = []
                            cur = next_j
                            while cur != prev_j and len(path_j) <= len(piece_j):
                                path_j.append(cur)
                                idx = piece_j.index(cur)
                                cur = piece_j[(idx + 1) % len(piece_j)]
                                if cur == v:
                                    break
                            if cur == prev_j:
                                path_j.append(prev_j)
                            
                            merged = [v] + path_i + path_j
                            
                            if len(merged) < 3:
                                continue
                            if len(set(merged[:-1])) != len(merged) - 1:
                                continue
                            
                            valid = True
                            for k in range(len(merged)):
                                a, b = merged[k], merged[(k + 1) % len(merged)]
                                e = (min(a, b), max(a, b))
                                if e not in edge_set:
                                    valid = False
                                    break
                                if e in used_edges and e not in edges_i and e not in edges_j:
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            pieces[pi] = merged
                            pieces[pj] = []
                            improved = True
                            
                        except Exception:
                            continue
                        
                        if improved:
                            break
                    if improved:
                        break
            
            pieces = [p for p in pieces if p]
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
