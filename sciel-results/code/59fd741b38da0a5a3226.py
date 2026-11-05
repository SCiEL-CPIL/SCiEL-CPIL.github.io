import time
import random
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit (evaluator kills at 150s)

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0

        edge_set = set(edges)
        used_edges = set()

        for path in paths:
            if len(path) < 2:
                continue
            if len(set(path)) != len(path):
                return False
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)

        return used_edges == edge_set

    def pairing_to_paths(pair_at, edges, adj, vertices):
        """
        Given a pairing structure (pair_at[v] maps edge->edge at vertex v),
        extract simple paths. Returns list of vertex lists.
        """
        # Build transition: from (edge, at_vertex) to next state
        # State: (edge, v) where v is an endpoint of edge, and we're at v
        # Next: if pair_at[v] has edge, go to (next_edge, other_endpoint_of_next_edge)
        #       else: dead end (endpoint of a path)
        
        # Find all endpoints: (e, v) where v is endpoint of e and e not paired at v
        endpoints = []
        for v in vertices:
            for e in adj[v]:
                if e not in pair_at[v]:
                    endpoints.append((e, v))
        
        # Build transition function
        def transition(e, v):
            """From state (e, v), get next state or None if endpoint."""
            next_e = pair_at[v].get(e)
            if next_e is None:
                return None
            # next_e is incident to v, find other endpoint
            if next_e[0] == v:
                next_v = next_e[1]
            else:
                next_v = next_e[0]
            return (next_e, next_v)
        
        # Extract paths from endpoints
        visited_states = set()
        paths = []
        
        for e, v in endpoints:
            state = (e, v)
            if state in visited_states:
                continue
            
            # Walk forward from this endpoint
            path_states = [state]
            visited_states.add(state)
            cur_e, cur_v = e, v
            
            while True:
                nxt = transition(cur_e, cur_v)
                if nxt is None or nxt in visited_states:
                    break
                cur_e, cur_v = nxt
                path_states.append((cur_e, cur_v))
                visited_states.add((cur_e, cur_v))
            
            # Convert to vertex path
            # State (e, v) means: we're at vertex v, having just traversed edge e into v
            # The path starts at the first state's vertex, then follows edges
            # Actually: state[0] = (e0, v0) means we start at v0, take e0 to v1
            # state[1] = (e1, v1) means we're at v1, take e1 to v2
            # So vertices are: v0, v1, v2, ...
            # v_{i+1} = other endpoint of e_i from v_i
            
            if not path_states:
                continue
            
            v_path = [path_states[0][1]]
            for i, (e, v) in enumerate(path_states):
                if i < len(path_states) - 1:
                    # Next vertex is other endpoint of e from v
                    if e[0] == v:
                        nxt_v = e[1]
                    else:
                        nxt_v = e[0]
                    v_path.append(nxt_v)
            
            if len(v_path) >= 2:
                paths.append(v_path)
        
        # Check for cycles (edges paired at both ends)
        # Find any unvisited states
        for e in edges:
            u, v = e
            for vert in [u, v]:
                state = (e, vert)
                if state not in visited_states:
                    # This is part of a cycle
                    # Walk around the cycle
                    cycle_states = [state]
                    visited_states.add(state)
                    cur_e, cur_v = e, vert
                    while True:
                        nxt = transition(cur_e, cur_v)
                        if nxt is None or nxt == state or nxt in visited_states:
                            break
                        cur_e, cur_v = nxt
                        cycle_states.append((cur_e, cur_v))
                        visited_states.add((cur_e, cur_v))
                    
                    # Convert cycle to path by removing one edge
                    if len(cycle_states) >= 2:
                        v_path = [cycle_states[0][1]]
                        for i, (ce, cv) in enumerate(cycle_states):
                            if i < len(cycle_states) - 1:
                                if ce[0] == cv:
                                    nxt_v = ce[1]
                                else:
                                    nxt_v = ce[0]
                                v_path.append(nxt_v)
                        if len(v_path) >= 2:
                            paths.append(v_path)
        
        return paths

    def greedy_pairing(edges, adj, vertices, rng, strategy='random'):
        """
        Greedily pair edges at each vertex.
        Returns pair_at dict.
        """
        pair_at = {v: {} for v in vertices}
        
        # Process vertices in order of degree
        deg_order = sorted(vertices, key=lambda v: -len(adj[v]))
        
        for v in deg_order:
            # Get edges incident to v that haven't been paired at v yet
            used_at_v = set(pair_at[v].keys()) | set(pair_at[v].values())
            avail = [e for e in adj[v] if e not in used_at_v]
            
            if strategy == 'random':
                rng.shuffle(avail)
            elif strategy == 'degree':
                # Sort by degree of the other endpoint
                def other_deg(e):
                    if e[0] == v:
                        return len(adj[e[1]])
                    else:
                        return len(adj[e[0]])
                avail.sort(key=other_deg)
            
            i = 0
            while i < len(avail) - 1:
                e1 = avail[i]
                e2 = avail[i + 1]
                pair_at[v][e1] = e2
                pair_at[v][e2] = e1
                i += 2
            # If odd, leave one unpaired (it becomes an endpoint)
        
        return pair_at

    def local_search(pair_at, edges, adj, vertices, rng, max_iters=1000):
        """
        Improve pairing by swapping endpoints.
        Returns improved pair_at.
        """
        best_pair = {v: dict(pair_at[v]) for v in vertices}
        best_paths = pairing_to_paths(best_pair, edges, adj, vertices)
        best_count = len(best_paths)
        
        for it in range(max_iters):
            # Pick a random vertex and try to change its pairing
            v = rng.choice(vertices)
            used_at_v = set(best_pair[v].keys()) | set(best_pair[v].values())
            avail = [e for e in adj[v] if e not in used_at_v]
            
            if len(avail) < 2:
                continue
            
            # Try swapping two pairs or changing one pair
            # Simple approach: pick two random available edges and pair them differently
            i = rng.randint(0, len(avail) - 1)
            j = rng.randint(0, len(avail) - 1)
            if i == j:
                continue
            
            e1, e2 = avail[i], avail[j]
            
            # Unpair e1 and e2 at v
            old_partner1 = best_pair[v].pop(e1, None)
            old_partner2 = best_pair[v].pop(e2, None)
            
            # Pair them together
            best_pair[v][e1] = e2
            best_pair[v][e2] = e1
            
            # Re-pair old partners if they exist
            if old_partner1 is not None and old_partner2 is not None:
                best_pair[v][old_partner1] = old_partner2
                best_pair[v][old_partner2] = old_partner1
            elif old_partner1 is not None:
                pass  # old_partner1 becomes unpaired
            elif old_partner2 is not None:
                pass  # old_partner2 becomes unpaired
            
            # Evaluate
            new_paths = pairing_to_paths(best_pair, edges, adj, vertices)
            new_count = len(new_paths)
            
            if new_count < best_count:
                best_count = new_count
                best_paths = new_paths
            else:
                # Revert
                best_pair[v].pop(e1, None)
                best_pair[v].pop(e2, None)
                if old_partner1 is not None:
                    best_pair[v][e1] = old_partner1
                    best_pair[v][old_partner1] = e1
                if old_partner2 is not None:
                    best_pair[v][e2] = old_partner2
                    best_pair[v][old_partner2] = e2
        
        return best_pair, best_paths, best_count

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add((u, v) if u < v else (v, u))
            adj[v].add((u, v) if u < v else (v, u))
        vertices = sorted(adj.keys())
        
        # Compute lower bound
        deg = {v: len(neigh) for v, neigh in adj.items()}
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lower_bound = max(1, (odd_count + 1) // 2)
        
        # Allocate time per instance
        time_per_instance = 35.0
        inst_start = time.time()
        inst_deadline = inst_start + time_per_instance
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Greedy pairing with multiple strategies and restarts
        for restart in range(500):
            if time.time() > inst_deadline - 3:
                break
            
            rng = random.Random(seed * 100000 + restart)
            strategy = rng.choice(['random', 'random', 'degree'])
            pair_at = greedy_pairing(edges, adj, vertices, rng, strategy)
            paths = pairing_to_paths(pair_at, edges, adj, vertices)
            
            if paths and len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                # Try local search
                if time.time() < inst_deadline - 5:
                    improved_pair, improved_paths, improved_count = local_search(
                        pair_at, edges, adj, vertices, rng, max_iters=200)
                    if improved_count < best_count:
                        best_count = improved_count
                        best_paths = improved_paths
        
        # Strategy 2: Targeted greedy (from original)
        if time.time() < inst_deadline - 5:
            for target_k in range(lower_bound, min(len(edges), lower_bound + 3) + 1):
                if time.time() > inst_deadline - 3:
                    break
                
                for restart in range(2000):
                    if time.time() > inst_deadline - 3:
                        break
                    
                    seed_val = seed * 100000 + 1000000 + restart
                    paths = greedy_target(edges, target_k, seed_val)
                    if paths is not None:
                        if validate_paths(paths, edges):
                            if target_k < best_count:
                                best_count = target_k
                                best_paths = paths
                            break
        
        if best_paths is None:
            # Fallback: greedy without target
            rng = random.Random(seed + 9999)
            remaining = set(edges)
            
            paths = []
            while remaining:
                degs = {}
                for u, v in remaining:
                    degs[u] = degs.get(u, 0) + 1
                    degs[v] = degs.get(v, 0) + 1
                if not degs:
                    break
                max_d = max(degs.values())
                cands = [v for v, d in degs.items() if d == max_d]
                start_u = rng.choice(cands)

                neighbors = [nb for nb in adj[start_u]
                             if (start_u, nb) in remaining or (nb, start_u) in remaining]
                if not neighbors:
                    e = next(iter(remaining))
                    start_u, start_v = e
                else:
                    start_v = rng.choice(neighbors)

                path = [start_u, start_v]
                if (start_u, start_v) in remaining:
                    remaining.remove((start_u, start_v))
                else:
                    remaining.remove((start_v, start_u))

                changed = True
                while changed:
                    changed = False
                    back = path[-1]
                    cands = [nb for nb in adj[back]
                             if nb not in path and
                             ((back, nb) in remaining or (nb, back) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (back, nb) in remaining:
                            remaining.remove((back, nb))
                        else:
                            remaining.remove((nb, back))
                        path.append(nb)
                        changed = True
                        continue
                    front = path[0]
                    cands = [nb for nb in adj[front]
                             if nb not in path and
                             ((front, nb) in remaining or (nb, front) in remaining)]
                    if cands:
                        nb = rng.choice(cands)
                        if (front, nb) in remaining:
                            remaining.remove((front, nb))
                        else:
                            remaining.remove((nb, front))
                        path.insert(0, nb)
                        changed = True
                        continue

                paths.append(path)

            return paths

        return best_paths

    def greedy_target(edges, k, seed_val):
        """
        Attempts to cover all edges with exactly k simple paths.
        Returns the list of paths if successful, else None.
        """
        if k <= 0 and edges:
            return None
        if len(edges) < k:
            return None

        rng = random.Random(seed_val)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        paths = [[] for _ in range(k)]

        # Pick k distinct edges to start k paths
        edge_list = list(remaining)
        rng.shuffle(edge_list)

        start_edges = edge_list[:k]

        for i in range(k):
            u, v = start_edges[i]
            if (u, v) in remaining:
                remaining.remove((u, v))
            else:
                remaining.remove((v, u))
            paths[i] = [u, v]

        # Extend paths greedily
        changed = True
        while changed and remaining:
            changed = False
            idxs = list(range(k))
            rng.shuffle(idxs)

            for i in idxs:
                if not paths[i]:
                    continue

                # Try extending back
                back = paths[i][-1]
                candidates = []
                for nb in adj[back]:
                    if nb in paths[i]:
                        continue
                    if (back, nb) in remaining or (nb, back) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (back, nb) in remaining:
                        remaining.remove((back, nb))
                    else:
                        remaining.remove((nb, back))
                    paths[i].append(nb)
                    changed = True
                    continue

                # Try extending front
                front = paths[i][0]
                candidates = []
                for nb in adj[front]:
                    if nb in paths[i]:
                        continue
                    if (front, nb) in remaining or (nb, front) in remaining:
                        candidates.append(nb)

                if candidates:
                    nb = rng.choice(candidates)
                    if (front, nb) in remaining:
                        remaining.remove((front, nb))
                    else:
                        remaining.remove((nb, front))
                    paths[i].insert(0, nb)
                    changed = True
                    continue

        if remaining:
            return None
        return paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []

    for idx, spec in enumerate(specs):
        n, p, seed = spec
        if time.time() > deadline - 10:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        paths = solve_instance(n, p, seed)
        results.append(paths)

    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])

    return results


if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
