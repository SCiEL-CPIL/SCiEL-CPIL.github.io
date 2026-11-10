"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from random import Random
import numpy as np

def construct():
    _start = time.time()
    _DEADLINE = 145.0  # safety margin before 150s kill

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

    def simulate_annealing_path_cover(edges, time_limit=30.0):
        """
        Simulated annealing over edge orientations + successor assignments.
        
        State: for each edge i, we store:
          - orient[i]: 0 means edges[i][0]->edges[i][1], 1 means reversed
          - succ[i]: index of edge that follows i (or -1 if none)
        
        The state defines a set of directed chains. Cost = number of chains 
        (paths) + penalty for any vertex appearing more than once in a chain.
        
        Moves:
          1. Flip orientation of a random edge.
          2. Change successor of a random edge (swap with another or set to -1).
          3. Swap successors of two edges.
        """
        n_edges = len(edges)
        if n_edges == 0:
            return []
        if n_edges == 1:
            return [list(edges[0])]

        edge_set = set()
        for (u, v) in edges:
            edge_set.add((u, v) if u <= v else (v, u))

        # Build vertex-adjacency for quick edge lookup
        vertex_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_edges[u].append(i)
            vertex_edges[v].append(i)

        # Precompute: for each edge i and orientation, the start and end vertex
        def get_endpoints(i, orient):
            u, v = edges[i]
            if orient == 0:
                return u, v
            else:
                return v, u

        # Build compatibility: can edge j follow edge i?
        # Edge j can follow edge i if end(i) == start(j) for some orientation of j
        # We'll compute this on the fly during cost evaluation

        # Initial state: random orientations, no successors
        orient = [random.randint(0, 1) for _ in range(n_edges)]
        succ = [-1] * n_edges
        pred = [-1] * n_edges

        def build_pred_from_succ():
            nonlocal pred
            pred = [-1] * n_edges
            for i in range(n_edges):
                if succ[i] != -1:
                    pred[succ[i]] = i

        build_pred_from_succ()

        def evaluate():
            """
            Returns (num_paths, penalty, paths_list)
            num_paths = number of chains
            penalty = sum over chains of (num_vertex_repeats)
            paths_list = list of vertex paths for each chain (may have repeats)
            """
            # Find all chains
            visited = [False] * n_edges
            chains = []
            for i in range(n_edges):
                if pred[i] != -1:
                    continue
                # Start of a chain
                chain_edges = []
                cur = i
                while cur != -1 and not visited[cur]:
                    visited[cur] = True
                    chain_edges.append(cur)
                    cur = succ[cur]
                if chain_edges:
                    chains.append(chain_edges)

            # Check for any unvisited edges (shouldn't happen if succ is a permutation subset)
            for i in range(n_edges):
                if not visited[i]:
                    chains.append([i])

            num_paths = len(chains)
            penalty = 0
            paths_list = []

            for chain in chains:
                # Build vertex path
                vpath = []
                for idx, ei in enumerate(chain):
                    u, v = get_endpoints(ei, orient[ei])
                    if idx == 0:
                        vpath.append(u)
                        vpath.append(v)
                    else:
                        # Check continuity
                        last = vpath[-1]
                        if last == u:
                            vpath.append(v)
                        elif last == v:
                            vpath.append(u)
                        else:
                            # Discontinuity: add penalty
                            penalty += 1
                            vpath.append(u)
                            vpath.append(v)

                # Check for vertex repeats
                if len(vpath) != len(set(vpath)):
                    penalty += (len(vpath) - len(set(vpath))) * 2

                paths_list.append(vpath)

            return num_paths, penalty, paths_list

        def extract_valid_paths():
            """
            Extract valid simple paths from current state.
            Break chains at discontinuities or vertex repeats.
            """
            visited = [False] * n_edges
            valid_paths = []

            for i in range(n_edges):
                if pred[i] != -1:
                    continue
                # Follow chain
                chain_edges = []
                cur = i
                while cur != -1 and not visited[cur]:
                    visited[cur] = True
                    chain_edges.append(cur)
                    cur = succ[cur]

                if not chain_edges:
                    continue

                # Build vertex path, breaking at invalid points
                current_path = []
                for idx, ei in enumerate(chain_edges):
                    u, v = get_endpoints(ei, orient[ei])
                    if idx == 0:
                        current_path = [u, v]
                    else:
                        last = current_path[-1]
                        if last == u:
                            current_path.append(v)
                        elif last == v:
                            current_path.append(u)
                        else:
                            # Break here
                            if len(current_path) >= 2:
                                valid_paths.append(current_path)
                            current_path = [u, v]

                    # Check if path is still simple
                    if len(current_path) != len(set(current_path)):
                        # Find the last valid prefix
                        # We need to backtrack
                        # Simple approach: just take the longest simple prefix
                        k = len(current_path)
                        while k > 1 and len(set(current_path[:k])) != k:
                            k -= 1
                        if k >= 2:
                            valid_paths.append(current_path[:k])
                        # The remaining edges from the break point will be handled
                        # by being unvisited... but they are marked visited.
                        # This is a problem. Let's use a different approach.
                        # For now, just add what we have and mark rest as separate.
                        current_path = []

                if len(current_path) >= 2:
                    valid_paths.append(current_path)

            # Add any unvisited edges as single-edge paths
            for i in range(n_edges):
                if not visited[i]:
                    u, v = edges[i]
                    valid_paths.append([u, v])
                    visited[i] = True

            # Validate and deduplicate
            edge_set_local = set()
            for (u, v) in edges:
                edge_set_local.add((u, v) if u <= v else (v, u))

            final_paths = []
            covered = set()
            for p in valid_paths:
                if len(p) < 2:
                    continue
                if len(set(p)) != len(p):
                    # Break into single edges
                    for i in range(len(p) - 1):
                        u, v = p[i], p[i + 1]
                        key = (u, v) if u <= v else (v, u)
                        if key in edge_set_local and key not in covered:
                            final_paths.append([u, v])
                            covered.add(key)
                    continue
                ok = True
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    key = (u, v) if u <= v else (v, u)
                    if key not in edge_set_local or key in covered:
                        ok = False
                        break
                    covered.add(key)
                if ok:
                    final_paths.append(p)

            # Add uncovered edges
            for (u, v) in edges:
                key = (u, v) if u <= v else (v, u)
                if key not in covered:
                    final_paths.append([u, v])
                    covered.add(key)

            return final_paths

        # Simulated annealing
        best_cost = float('inf')
        best_orient = orient[:]
        best_succ = succ[:]
        best_paths = None

        current_cost, _, _ = evaluate()
        T = 1.0
        T_min = 0.01
        cooling = 0.9995

        start_time = time.time()
        max_iter = 500000

        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break

            # Choose a move
            move_type = random.random()
            old_orient = orient[:]
            old_succ = succ[:]

            if move_type < 0.3:
                # Flip orientation
                i = random.randint(0, n_edges - 1)
                orient[i] = 1 - orient[i]
            elif move_type < 0.6:
                # Change successor of one edge
                i = random.randint(0, n_edges - 1)
                old_succ_val = succ[i]
                # Choose new successor: -1 or a random edge that doesn't already have a predecessor
                candidates = [-1]
                for j in range(n_edges):
                    if j != i and pred[j] == -1:
                        candidates.append(j)
                if candidates:
                    new_val = random.choice(candidates)
                    # Remove i from old successor's pred
                    if old_succ_val != -1:
                        pred[old_succ_val] = -1
                    succ[i] = new_val
                    if new_val != -1:
                        pred[new_val] = i
            elif move_type < 0.8:
                # Swap successors of two edges
                i, j = random.sample(range(n_edges), 2)
                si, sj = succ[i], succ[j]
                succ[i], succ[j] = sj, si
                # Update pred
                if si != -1:
                    pred[si] = j
                if sj != -1:
                    pred[sj] = i
            else:
                # Random restart of a portion
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, n_edges - 1)
                    if succ[i] != -1:
                        pred[succ[i]] = -1
                        succ[i] = -1

            new_cost, _, _ = evaluate()
            delta = new_cost - current_cost

            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_orient = orient[:]
                    best_succ = succ[:]
                    best_paths = extract_valid_paths()
            else:
                # Revert
                orient = old_orient[:]
                succ = old_succ[:]
                build_pred_from_succ()

            T *= cooling
            if T < T_min:
                T = T_min

        if best_paths is None:
            best_paths = extract_valid_paths()

        return best_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > _DEADLINE - 20:
            edges = instance(*spec)
            results.append(greedy_path_cover(edges))
            continue

        edges = instance(*spec)
        
        # Get a valid initial solution from greedy
        paths_greedy = greedy_path_cover(edges)
        
        # Run simulated annealing with remaining time
        remaining_time = _DEADLINE - time.time() - 5.0
        if remaining_time > 5:
            try:
                paths_sa = simulate_annealing_path_cover(edges, time_limit=min(remaining_time, 40.0))
                if paths_sa and validate_paths(paths_sa, edges):
                    if len(paths_sa) < len(paths_greedy):
                        paths_greedy = paths_sa
            except Exception:
                pass

        # Final validation
        if not validate_paths(paths_greedy, edges):
            paths_greedy = greedy_path_cover(edges)
            if not validate_paths(paths_greedy, edges):
                paths_greedy = [[list(e)] for e in edges]

        results.append(paths_greedy)

    return results
# EVOLVE-BLOCK-END
