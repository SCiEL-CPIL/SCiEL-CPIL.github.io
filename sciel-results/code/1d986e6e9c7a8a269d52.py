"""Edge path cover via maximum matching formulation."""
import time
from collections import defaultdict
from random import Random
import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    _start = time.time()
    _DEADLINE = 148.0

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))
        covered = set()
        for p in paths:
            if len(p) < 2:
                return False
            if len(set(p)) != len(p):
                return False
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                key = (u, v) if u <= v else (v, u)
                if key not in edge_set:
                    return False
                if key in covered:
                    return False
                covered.add(key)
        return covered == edge_set

    def path_cover_via_matching(edges):
        """
        Find minimum edge path cover using maximum matching.
        
        For each edge e=(u,v), we can orient it as u->v or v->u.
        We want to maximize the number of "transitions" where edge i ends
        at vertex x and edge j starts at vertex x.
        
        Formulation: Create a bipartite graph where:
        - Left side: each edge has two possible "start" nodes (u_start, v_start)
        - Right side: each edge has two possible "end" nodes (u_end, v_end)
        
        Actually, simpler: For each undirected edge (u,v), we decide its orientation.
        If oriented u->v, it can follow any edge ending at u and can be followed by any edge starting at v.
        
        Better formulation: 
        - Each edge e has endpoints (u_e, v_e)
        - We create a bipartite graph with:
          * Left nodes: (e, 0) meaning edge e starts at u_e, ends at v_e
          * Left nodes: (e, 1) meaning edge e starts at v_e, ends at u_e
          * Right nodes: (f, 0) meaning edge f starts at u_f, ends at v_f  
          * Right nodes: (f, 1) meaning edge f starts at v_f, ends at u_f
          
        Actually, let me think more carefully.
        
        A path cover of edges is a set of directed paths that are edge-disjoint and cover all edges.
        Each path is a sequence of edges where consecutive edges share a vertex.
        
        We can model this as: each edge is a node in a directed graph. There's an arc from edge e to edge f
        if e ends at a vertex that f starts at (for some orientation). We want to find a collection of
        vertex-disjoint directed paths (in this meta-graph) that covers all edge-nodes.
        
        This is equivalent to: maximize the number of arcs in a vertex-disjoint path cover.
        The minimum number of paths = |E| - (maximum number of arcs in such a cover).
        
        To find the maximum number of arcs: this is a maximum matching problem!
        
        Create bipartite graph:
        - Left: each edge e has two copies: e_in (represents e as a predecessor) and... 
        
        Wait, let me reconsider. 
        
        Standard reduction: Given a directed graph D=(V,E), the minimum number of directed paths
        that cover all edges (edge-disjoint) equals |E| - max_flow where we create:
        - Source -> each edge-node (capacity 1)
        - Each edge-node -> sink (capacity 1)
        - For each pair of edges (e,f) where e ends at vertex v and f starts at vertex v,
          add arc e_node -> f_node with capacity 1
        
        Actually, the standard result is:
        Minimum path cover (vertex-disjoint paths covering all vertices) in DAG = |V| - max_matching
        in the bipartite graph formed by splitting each vertex into in/out copies.
        
        For edge cover by paths, we need to think differently.
        
        Let me use a different approach: 
        1. Find an Eulerian augmentation: add edges to make graph Eulerian, find Eulerian circuit,
           then the added edges tell us how many paths we need.
        
        Actually, here's a cleaner formulation:
        
        For each vertex v, let deg(v) be its degree. In any path cover:
        - Each path has exactly 2 endpoints (start and end)
        - At each internal vertex of a path, 2 edges are used (one in, one out)
        - At endpoints, 1 edge is used
        
        So if we have k paths, the total "endpoint incidences" = 2k.
        At vertex v, the number of path endpoints at v is deg(v) - 2*(number of times v is internal).
        Sum over all v: sum_v deg(v) - 2*(total internal visits) = 2k
        2|E| - 2*(total internal visits) = 2k
        k = |E| - (total internal visits)
        
        To minimize k, maximize total internal visits. A vertex v can be internal at most floor(deg(v)/2) times.
        So max possible total internal visits = sum_v floor(deg(v)/2).
        Thus min k >= |E| - sum_v floor(deg(v)/2).
        
        Gallai's conjecture says k <= floor((n-1)/2) for connected graphs, which is much stronger.
        
        For the algorithm, let's use the flow/matching approach:
        
        Build a bipartite graph B:
        - Left partition: one node per edge, representing "this edge as a predecessor"
        - Right partition: one node per edge, representing "this edge as a successor"
        - For edges e=(u,v) and f=(u',v'):
          * There's an arc from left-e to right-f if v == u' or v == v' or u == u' or u == v'
            i.e., if e and f share a common endpoint, and we can orient them so e's end matches f's start.
        
        But we need to ensure consistency of orientations. This gets complex.
        
        Simpler practical approach: 
        1. Use the degree-based lower bound as a guide.
        2. Use greedy + local improvement.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        if n_edges == 1:
            return [list(edges[0])]
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Compute degrees
        deg = {v: len(adj[v]) for v in adj}
        
        # Lower bound: |E| - sum_v floor(deg(v)/2)
        lb = n_edges - sum(len(adj[v]) // 2 for v in adj)
        
        # Greedy: build paths by extending from endpoints
        # Start with all edges as separate paths
        # Then try to merge paths
        
        # Represent each edge with its two endpoints
        # We'll build paths greedily
        
        remaining_edges = set()
        for u, v in edges:
            remaining_edges.add((min(u,v), max(u,v)))
        
        paths = []
        used_edges = set()
        
        # Greedy: repeatedly find longest possible path
        while remaining_edges:
            # Find a good starting edge
            # Prefer edges with high-degree endpoints
            best_start = None
            best_score = -1
            for e in remaining_edges:
                u, v = e
                score = deg[u] + deg[v]
                if score > best_score:
                    best_score = score
                    best_start = e
            
            if best_start is None:
                break
                
            u, v = best_start
            # Try both orientations
            best_path = None
            for start, end in [(u, v), (v, u)]:
                path = [start, end]
                used_in_path = {(min(start,end), max(start,end))}
                remaining_edges.discard((min(start,end), max(start,end)))
                
                current = end
                prev = start
                while True:
                    # Find next edge extending from current
                    next_edge = None
                    best_next_score = -1
                    for neighbor in adj[current]:
                        if neighbor == prev:
                            continue
                        e = (min(current, neighbor), max(current, neighbor))
                        if e in used_in_path or e not in remaining_edges:
                            continue
                        # Score: prefer going to high-degree nodes that have more unused edges
                        unused_degree = sum(1 for nb in adj[neighbor] if nb != current 
                                           and (min(neighbor, nb), max(neighbor, nb)) in remaining_edges
                                           and (min(neighbor, nb), max(neighbor, nb)) not in used_in_path)
                        score = unused_degree * 10 + deg[neighbor]
                        if score > best_next_score:
                            best_next_score = score
                            next_edge = (neighbor, e)
                    
                    if next_edge is None:
                        break
                    
                    next_vertex, e = next_edge
                    path.append(next_vertex)
                    used_in_path.add(e)
                    remaining_edges.discard(e)
                    prev = current
                    current = next_vertex
            
            if best_path is None or (best_path is not None and len(path) > len(best_path)):
                best_path = path
            
            if best_path and len(best_path) >= 2:
                paths.append(best_path)
                for i in range(len(best_path) - 1):
                    e = (min(best_path[i], best_path[i+1]), max(best_path[i], best_path[i+1]))
                    used_edges.add(e)
            else:
                # Fallback: just add one edge
                e = next(iter(remaining_edges))
                remaining_edges.discard(e)
                paths.append(list(e))
                used_edges.add(e)
        
        # Verify and fix
        if not validate_paths(paths, edges):
            # Fallback: each edge as separate path
            return [[u, v] for u, v in edges]
        
        return paths

    def greedy_path_cover(edges):
        """Simple greedy fallback."""
        if not edges:
            return []
        edge_set = set()
        for e in edges:
            u, v = e
            edge_set.add((u, v) if u <= v else (v, u))
        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        used_edges = set()
        paths = []
        while edge_set - used_edges:
            remaining = edge_set - used_edges
            if not remaining:
                break
            rem_adj = defaultdict(list)
            for (u, v) in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            start = max(rem_adj.keys(), key=lambda x: len(rem_adj[x])) if rem_adj else None
            if start is None:
                break
            path = [start]
            visited_in_path = {start}
            current = start
            while True:
                neighbors = [n for n in rem_adj[current] if n not in visited_in_path]
                if not neighbors:
                    break
                best_neighbor = max(neighbors, key=lambda nb: len([x for x in rem_adj[nb] if x not in visited_in_path]))
                path.append(best_neighbor)
                visited_in_path.add(best_neighbor)
                current = best_neighbor
            path_edges = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if u <= v else (v, u)
                path_edges.add(key)
            valid = all(e in remaining for e in path_edges)
            if valid and path_edges:
                used_edges |= path_edges
                paths.append(path)
            else:
                e = next(iter(remaining))
                used_edges.add(e)
                paths.append([e[0], e[1]])
        return paths

    def eulerian_augmentation_path_cover(edges):
        """
        Use Eulerian augmentation: 
        1. Find odd-degree vertices
        2. Pair them up (minimum matching on odd vertices)
        3. Add those pairing edges to make graph Eulerian
        4. Find Eulerian circuit
        5. Remove the added edges; the circuit breaks into paths
        
        Number of paths = number of added edges = (number of odd-degree vertices) / 2
        """
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find odd-degree vertices
        odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
        
        if not odd_vertices:
            # Graph is Eulerian, one path (circuit)
            # Find Eulerian circuit
            return _find_eulerian_circuit(edges, adj)
        
        # Pair up odd vertices to minimize added edges
        # Simple: just pair them arbitrarily
        # Better: find minimum weight perfect matching on odd vertices
        # For now, greedy pairing
        
        n_odd = len(odd_vertices)
        if n_odd < 2:
            # Shouldn't happen for connected graph
            return greedy_path_cover(edges)
        
        # Greedy pairing: pair each odd vertex with its nearest odd neighbor
        # (where "nearest" means directly connected if possible)
        paired = set()
        added_edges = []
        used = set()
        
        for i, v in enumerate(odd_vertices):
            if v in used:
                continue
            # Find best partner
            best_partner = None
            best_dist = float('inf')
            for j in range(i + 1, n_odd):
                w = odd_vertices[j]
                if w in used:
                    continue
                if w in adj[v]:
                    # Directly connected, prefer this
                    if best_partner is None or len(adj[v] & adj[w]) < len(adj[v] & adj[best_partner]):
                        best_partner = w
                    break
                else:
                    # BFS distance
                    dist = _bfs_distance(v, w, adj)
                    if dist < best_dist:
                        best_dist = dist
                        best_partner = w
            
            if best_partner is not None:
                # Add edge between v and best_partner (this is a virtual edge)
                added_edges.append((v, best_partner))
                used.add(v)
                used.add(best_partner)
            else:
                # Shouldn't happen
                break
        
        # If we couldn't pair all, fall back to greedy
        if len(used) < n_odd:
            return greedy_path_cover(edges)
        
        # Now we have added_edges. Build augmented graph
        aug_adj = defaultdict(set)
        for u, v in edges:
            aug_adj[u].add(v)
            aug_adj[v].add(u)
        for u, v in added_edges:
            aug_adj[u].add(v)
            aug_adj[v].add(u)
        
        # Find Eulerian circuit in augmented graph
        # We need to track which edges are original and which are added
        all_edges = set()
        for u, v in edges:
            all_edges.add((min(u,v), max(u,v)))
        added_set = set()
        for u, v in added_edges:
            added_set.add((min(u,v), max(u,v)))
        
        # Find Eulerian circuit
        circuit = _find_eulerian_circuit_augmented(all_edges, added_set, aug_adj)
        
        if circuit is None:
            return greedy_path_cover(edges)
        
        # Extract paths: the circuit is a sequence of vertices.
        # When we hit an added edge, we cut the path there.
        # Actually, we need to track which edges in the circuit are added.
        
        # Rebuild: walk the circuit, track edges
        paths = []
        current_path = []
        for i in range(len(circuit) - 1):
            u, v = circuit[i], circuit[i + 1]
            e = (min(u,v), max(u,v))
            if not current_path:
                current_path = [u, v]
            else:
                current_path.append(v)
            
            if e in added_set:
                if len(current_path) >= 2:
                    paths.append(current_path)
                current_path = []
        
        if len(current_path) >= 2:
            paths.append(current_path)
        
        # Validate
        if validate_paths(paths, edges):
            return paths
        
        return greedy_path_cover(edges)

    def _bfs_distance(src, dst, adj):
        if src == dst:
            return 0
        visited = {src}
        queue = [src]
        dist = 0
        while queue:
            next_queue = []
            for node in queue:
                for neighbor in adj[node]:
                    if neighbor == dst:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_queue.append(neighbor)
            queue = next_queue
            dist += 1
        return float('inf')

    def _find_eulerian_circuit(edges, adj):
        """Find Eulerian circuit in an Eulerian graph."""
        if not edges:
            return []
        
        # Use Hierholzer's algorithm
        # Build adjacency list with edge indices
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u,v), max(u,v)))
        
        adj_list = defaultdict(list)
        for u, v in edge_set:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Hierholzer
        stack = [next(iter(adj_list))]
        circuit = []
        
        while stack:
            v = stack[-1]
            if adj_list[v]:
                # Find an unused edge
                w = adj_list[v].pop()
                # Remove the reverse edge
                if w in adj_list[v]:
                    adj_list[w].remove(v)
                stack.append(w)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def _find_eulerian_circuit_augmented(original_edges, added_edges, adj):
        """
        Find Eulerian circuit, tracking which edges are added.
        adj is the augmented adjacency.
        """
        # We need to build a proper multigraph representation
        # Since we might have parallel edges, use a dict of edge counts
        edge_multiset = {}
        for u, v in original_edges:
            key = (min(u,v), max(u,v))
            edge_multiset[key] = edge_multiset.get(key, 0) + 1
        for u, v in added_edges:
            key = (min(u,v), max(u,v))
            edge_multiset[key] = edge_multiset.get(key, 0) + 1
        
        # Build adjacency with edge identifiers
        adj_with_ids = defaultdict(list)
        edge_id = 0
        edge_map = {}  # (u,v) -> list of edge_ids
        for key, count in edge_multiset.items():
            u, v = key
            for _ in range(count):
                eid = edge_id
                edge_id += 1
                edge_map.setdefault(key, []).append(eid)
                adj_with_ids[u].append((v, eid))
                adj_with_ids[v].append((u, eid))
        
        # Hierholzer's algorithm
        if not adj_with_ids:
            return []
        
        start_vertex = next(iter(adj_with_ids))
        stack = [(start_vertex, None)]  # (vertex, incoming_edge_id)
        circuit_vertices = []
        circuit_edge_ids = []
        
        while stack:
            v, in_eid = stack[-1]
            if adj_with_ids[v]:
                # Take an edge
                w, eid = adj_with_ids[v].pop()
                # Remove the reverse reference
                # Find and remove (v, eid) from adj_with_ids[w]
                for i, (nb, e) in enumerate(adj_with_ids[w]):
                    if e == eid:
                        adj_with_ids[w].pop(i)
                        break
                stack.append((w, eid))
            else:
                circuit_vertices.append(v)
                if in_eid is not None:
                    circuit_edge_ids.append(in_eid)
                stack.pop()
        
        circuit_vertices.reverse()
        circuit_edge_ids.reverse()
        
        return circuit_vertices

    def local_improve(paths, edges, time_limit=20.0):
        """
        Local improvement: try to merge pairs of paths if they can be connected.
        Two paths can be merged if an endpoint of one connects to an endpoint of the other
        via an edge that... wait, no. The paths are already edge-disjoint and cover all edges.
        We can't add edges. We can only rearrange.
        
        Actually, merging isn't straightforward since we can't add edges.
        Instead, try to swap segments between paths to reduce total length.
        
        For now, just return paths as-is.
        """
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - _start
        if elapsed > _DEADLINE - 30:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        
        # Try multiple strategies, keep best
        candidates = []
        
        # Strategy 1: Eulerian augmentation
        try:
            paths_euler = eulerian_augmentation_path_cover(edges)
            if paths_euler and validate_paths(paths_euler, edges):
                candidates.append(paths_euler)
        except Exception:
            pass
        
        # Strategy 2: Greedy
        try:
            paths_greedy = greedy_path_cover(edges)
            if paths_greedy and validate_paths(paths_greedy, edges):
                candidates.append(paths_greedy)
        except Exception:
            pass
        
        # Strategy 3: Matching-based
        try:
            paths_match = path_cover_via_matching(edges)
            if paths_match and validate_paths(paths_match, edges):
                candidates.append(paths_match)
        except Exception:
            pass
        
        if candidates:
            best = min(candidates, key=len)
            results.append(best)
        else:
            # Last resort
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))

    return results
