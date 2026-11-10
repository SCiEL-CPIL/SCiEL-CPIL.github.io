"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def get_2_core(edge_set):
        """Extract the 2-core of the graph."""
        if not edge_set:
            return set(), {}
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for u in list(adj.keys()):
                if len(adj[u]) < 2:
                    to_remove.add(u)
            if to_remove:
                changed = True
                for u in to_remove:
                    for v in adj[u]:
                        adj[v].discard(u)
                    del adj[u]
        
        core_edges = set()
        for u in adj:
            for v in adj[u]:
                if u < v:
                    core_edges.add((u, v))
        return core_edges, dict(adj)
    
    def find_cycle_bfs(adj, remaining_edges, start=None):
        """Find any cycle in the subgraph using BFS. Returns cycle as list of vertices or None."""
        if not remaining_edges:
            return None
        
        # Build adjacency from remaining edges
        rem_adj = defaultdict(set)
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        if not rem_adj:
            return None
        
        # Try to find a cycle by doing BFS from each vertex with degree >= 2
        candidates = [u for u in rem_adj if len(rem_adj[u]) >= 2]
        if not candidates:
            return None
        
        random.shuffle(candidates)
        
        for s in candidates[:10]:
            # BFS from s
            parent = {s: None}
            queue = deque([s])
            found = False
            while queue and not found:
                cur = queue.popleft()
                for nb in sorted(rem_adj[cur]):
                    if nb == s:
                        # Found a cycle: trace back from cur to s
                        if cur != s:
                            path = [cur]
                            node = cur
                            while parent[node] is not None and parent[node] != s:
                                node = parent[node]
                                path.append(node)
                            path.append(s)
                            path.reverse()
                            if len(path) >= 3:
                                return path
                        found = True
                        break
                    elif nb not in parent:
                        parent[nb] = cur
                        queue.append(nb)
        
        return None
    
    def find_best_decomposition(edges, n):
        """Find decomposition with minimum pieces."""
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not edge_set:
            return []
        
        # Pre-compute 2-core
        core_edges, core_adj = get_2_core(edge_set)
        
        best_pieces = None
        best_count = len(edge_set) + 1
        
        def greedy_decompose(random_seed=None):
            """One greedy pass. Returns list of pieces."""
            if random_seed is not None:
                random.seed(random_seed)
            
            remaining = set(edge_set)
            pieces = []
            
            # Phase 1: Extract cycles from 2-core region using BFS
            remaining_core = set(core_edges)
            while remaining_core:
                if time.time() > deadline - 20:
                    break
                
                cycle = find_cycle_bfs(core_adj, remaining_core)
                if cycle and len(cycle) >= 3:
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        e = tuple(sorted((u, v)))
                        remaining_core.discard(e)
                        remaining.discard(e)
                    pieces.append(cycle)
                else:
                    break
            
            # Phase 2: Handle remaining edges
            while remaining:
                if time.time() > deadline - 20:
                    break
                
                # Try to find a cycle in remaining
                cycle = find_cycle_bfs(core_adj, remaining)
                if cycle and len(cycle) >= 3:
                    for i in range(len(cycle)):
                        cu, cv = cycle[i], cycle[(i + 1) % len(cycle)]
                        ce = tuple(sorted((cu, cv)))
                        remaining.discard(ce)
                    pieces.append(cycle)
                else:
                    # Take one edge
                    e = next(iter(remaining))
                    remaining.discard(e)
                    pieces.append(list(e))
            
            return pieces
        
        def local_improve(pieces):
            """Try to improve by merging adjacent single edges into cycles or re-extracting."""
            if not pieces:
                return pieces, len(pieces)
            
            # Build edge-to-piece-index map
            edge_to_piece = {}
            for i, piece in enumerate(pieces):
                if len(piece) == 2:
                    e = tuple(sorted(piece))
                    edge_to_piece[e] = i
                else:
                    for j in range(len(piece)):
                        u, v = piece[j], piece[(j + 1) % len(piece)]
                        e = tuple(sorted((u, v)))
                        edge_to_piece[e] = i
            
            improved = True
            count = len(pieces)
            
            while improved and time.time() < deadline - 25:
                improved = False
                count = len(pieces)
                
                # Try to find a cycle in the current decomposition's edges
                all_edges = set()
                for piece in pieces:
                    if len(piece) == 2:
                        all_edges.add(tuple(sorted(piece)))
                    else:
                        for j in range(len(piece)):
                            u, v = piece[j], piece[(j + 1) % len(piece)]
                            all_edges.add(tuple(sorted((u, v))))
                
                # Try to extract a longer cycle from all edges
                cycle = find_cycle_bfs(core_adj, all_edges)
                if cycle and len(cycle) >= 4:  # Only improve if we get a longer cycle
                    # Check if this cycle uses edges from multiple pieces
                    cycle_edges = set()
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        cycle_edges.add(tuple(sorted((u, v))))
                    
                    # Find which pieces are affected
                    affected = set()
                    for e in cycle_edges:
                        if e in edge_to_piece:
                            affected.add(edge_to_piece[e])
                    
                    if len(affected) >= 2:
                        # This could be an improvement
                        # Remove affected pieces and add the new cycle
                        new_pieces = [p for i, p in enumerate(pieces) if i not in affected]
                        # Add back edges not in the cycle
                        for i in affected:
                            piece = pieces[i]
                            if len(piece) == 2:
                                e = tuple(sorted(piece))
                                if e not in cycle_edges:
                                    new_pieces.append(piece)
                            else:
                                for j in range(len(piece)):
                                    u, v = piece[j], piece[(j + 1) % len(piece)]
                                    e = tuple(sorted((u, v)))
                                    if e not in cycle_edges:
                                        new_pieces.append([u, v])
                        new_pieces.append(cycle)
                        
                        if len(new_pieces) < count:
                            pieces = new_pieces
                            count = len(pieces)
                            improved = True
                            # Rebuild edge_to_piece
                            edge_to_piece = {}
                            for i, piece in enumerate(pieces):
                                if len(piece) == 2:
                                    e = tuple(sorted(piece))
                                    edge_to_piece[e] = i
                                else:
                                    for j in range(len(piece)):
                                        u, v = piece[j], piece[(j + 1) % len(piece)]
                                        e = tuple(sorted((u, v)))
                                        edge_to_piece[e] = i
            
            return pieces, len(pieces)
        
        # Multiple random restarts
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            pieces = greedy_decompose(random_seed=iteration)
            count = len(pieces)
            
            # Local improvement
            pieces, count = local_improve(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = [p[:] for p in pieces]
        
        return best_pieces if best_pieces else [list(e) for e in edge_set]
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = find_best_decomposition(edges, n)
        results.append(pieces)
        if time.time() > deadline - 5:
            # Fallback: just use individual edges for remaining graphs
            for spec2 in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results):]:
                n2, p2, seed2 = spec2
                edges2 = instance(n2, p2, seed2)
                results.append([list(e) for e in edges2])
            break
    
    return results
# EVOLVE-BLOCK-END
