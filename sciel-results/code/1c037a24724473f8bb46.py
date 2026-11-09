import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before the 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_best_cycle(adj, rng, piece_id_map=None):
        """
        Finds a long cycle in the graph represented by adj.
        If piece_id_map is provided, it tries to find a cycle that touches many distinct pieces.
        """
        candidates = [u for u, neighbors in adj.items() if len(neighbors) > 1]
        if not candidates:
            return None

        best_cycle = None
        best_score = 0

        candidates.sort(key=lambda x: len(adj[x]), reverse=True)
        starts = candidates[:8]
        if len(candidates) > 8:
            starts += rng.sample(candidates, min(8, len(candidates)))
        starts = list(set(starts))
        rng.shuffle(starts)

        for start in starts:
            color = {start: 1}
            parent = {start: -1}
            path = [start]
            path_index = {start: 0}
            local_best_cycle = None
            local_best_score = 0
            
            def get_shuffled_iter(neighs):
                lst = list(neighs)
                rng.shuffle(lst)
                return iter(lst)
            
            stack = [(start, get_shuffled_iter(adj[start]))]
            
            while stack:
                u, it = stack[-1]
                found_next = False
                for w in it:
                    if w == parent[u]:
                        continue
                    if color.get(w, 0) == 0:
                        color[w] = 1
                        parent[w] = u
                        path.append(w)
                        path_index[w] = len(path) - 1
                        stack.append((w, get_shuffled_iter(adj[w])))
                        found_next = True
                        break
                    elif color.get(w, 0) == 1:
                        if w in path_index:
                            idx = path_index[w]
                            cyc = path[idx:]
                            if len(cyc) >= 3:
                                # Calculate score
                                if piece_id_map:
                                    # Count distinct pieces
                                    pieces_in_cycle = set()
                                    for j in range(len(cyc)):
                                        a, b = cyc[j], cyc[(j + 1) % len(cyc)]
                                        e = (min(a, b), max(a, b))
                                        if e in piece_id_map:
                                            pieces_in_cycle.add(piece_id_map[e])
                                    score = len(pieces_in_cycle) * 100 + len(cyc)
                                else:
                                    score = len(cyc)
                                
                                if score > local_best_score:
                                    local_best_score = score
                                    local_best_cycle = list(cyc)
                
                if not found_next:
                    stack.pop()
                    color[u] = 2
                    if path and path[-1] == u:
                        path.pop()
                        if u in path_index:
                            del path_index[u]
            
            if local_best_cycle and local_best_score > best_score:
                best_score = local_best_score
                best_cycle = local_best_cycle

        return best_cycle

    def decompose_greedy(edges, rng):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        current_adj = {u: set(v) for u, v in adj.items()}
        current_edges = set(edges)
        
        while True:
            if time.time() - start_time > DEADLINE:
                break
            if not any(len(neigh) > 0 for neigh in current_adj.values()):
                break
            
            cycle = find_best_cycle(current_adj, rng)
            if cycle is None:
                break
            
            # Remove cycle edges
            for j in range(len(cycle)):
                a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                current_edges.discard(e)
                if b in current_adj[a]:
                    current_adj[a].discard(b)
                if a in current_adj[b]:
                    current_adj[b].discard(a)
            
            pieces.append(list(cycle))
        
        for e in current_edges:
            pieces.append([e[0], e[1]])
            
        return pieces

    def local_search(pieces, edges_set, rng):
        # Map edge to piece index
        edge_to_piece = {}
        for i, p in enumerate(pieces):
            if len(p) == 2:
                e = (min(p[0], p[1]), max(p[0], p[1]))
                edge_to_piece[e] = i
            else:
                for j in range(len(p)):
                    a, b = p[j], p[(j + 1) % len(p)]
                    e = (min(a, b), max(a, b))
                    edge_to_piece[e] = i
        
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            # Build adj from all edges
            adj = defaultdict(set)
            for u, v in edges_set:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find a cycle that touches many pieces
            cycle = find_best_cycle(adj, rng, edge_to_piece)
            if not cycle:
                break
            
            # Identify pieces involved
            involved_pieces = set()
            cycle_edges = set()
            for j in range(len(cycle)):
                a, b = cycle[j], cycle[(j + 1) % len(cycle)]
                e = (min(a, b), max(a, b))
                cycle_edges.add(e)
                if e in edge_to_piece:
                    involved_pieces.add(edge_to_piece[e])
            
            if len(involved_pieces) < 2:
                # No improvement possible if only one piece is involved
                continue
            
            # Get remaining edges from these pieces
            rem_edges = set()
            for p_idx in involved_pieces:
                p = pieces[p_idx]
                if len(p) == 2:
                    e = (min(p[0], p[1]), max(p[0], p[1]))
                    if e not in cycle_edges:
                        rem_edges.add(e)
                else:
                    for j in range(len(p)):
                        a, b = p[j], p[(j + 1) % len(p)]
                        e = (min(a, b), max(a, b))
                        if e not in cycle_edges:
                            rem_edges.add(e)
            
            # Decompose rem_edges
            new_pieces = decompose_greedy(list(rem_edges), rng)
            
            if len(new_pieces) < len(involved_pieces):
                # Perform swap
                # Remove old pieces
                for p_idx in sorted(involved_pieces, reverse=True):
                    del pieces[p_idx]
                # Add new pieces
                pieces.extend(new_pieces)
                improved = True
                # Rebuild edge_to_piece
                edge_to_piece = {}
                for i, p in enumerate(pieces):
                    if len(p) == 2:
                        e = (min(p[0], p[1]), max(p[0], p[1]))
                        edge_to_piece[e] = i
                    else:
                        for j in range(len(p)):
                            a, b = p[j], p[(j + 1) % len(p)]
                            e = (min(a, b), max(a, b))
                            edge_to_piece[e] = i

        return pieces

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_pieces = None
        best_count = float('inf')
        
        max_restarts = 200 # Reduced to allow time for local search
        
        for i in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                break
                
            rng = random.Random(i)
            pieces = decompose_greedy(edges, rng)
            
            # Local search polish
            pieces = local_search(pieces, set(edges), rng)
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [[u, v] for u, v in edges]
            
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
