def construct():
    import time
    import random
    from collections import defaultdict

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, num_trials=50, time_limit=60):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Precompute adjacency for original graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        best_partition = None
        best_count = float('inf')
        start_time = time.time()

        # Use a fixed RNG for reproducibility within trials
        trial_rng = random.Random(seed + 9999)

        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break

            current_edges = set(edges)
            partition = []

            while current_edges:
                if time.time() - start_time > time_limit:
                    # Fallback: cover remaining edges individually
                    for e in list(current_edges):
                        partition.append(list(e))
                    break

                # Strategy: Pick a random edge, then expand greedily
                # But bias towards edges incident to high-degree vertices
                # Compute degrees in current_edges
                deg = [0] * n
                for u, v in current_edges:
                    deg[u] += 1
                    deg[v] += 1

                # Pick a random edge with probability weighted by degree sum
                # For simplicity, pick random edge from list
                edge_list = list(current_edges)
                e = trial_rng.choice(edge_list)
                u_start, v_start = e

                # Start clique with {u_start, v_start}
                clique = {u_start, v_start}

                # Find candidates: vertices connected to ALL clique members via uncovered edges
                # Initial candidates: common neighbors of u_start and v_start in current_edges
                cand = set()
                for w in adj[u_start]:
                    if (min(u_start, w), max(u_start, w)) in current_edges and \
                       (min(v_start, w), max(v_start, w)) in current_edges:
                        cand.add(w)

                # Greedy expansion: repeatedly add a candidate that maintains clique property
                while cand:
                    # Pick candidate with max degree in current_edges (among candidates)
                    # Or random pick for diversity
                    # Let's try: pick the one with max degree in current_edges
                    best_w = None
                    max_deg = -1
                    for w in cand:
                        if deg[w] > max_deg:
                            max_deg = deg[w]
                            best_w = w

                    if best_w is None:
                        break

                    # Check if best_w can be added (it should by construction, but verify)
                    can_add = True
                    for w in clique:
                        edge = (min(best_w, w), max(best_w, w))
                        if edge not in current_edges:
                            can_add = False
                            break

                    if not can_add:
                        # Remove best_w from cand and try next
                        cand.discard(best_w)
                        continue

                    clique.add(best_w)

                    # Update candidates: must be connected to best_w in current_edges
                    new_cand = set()
                    for x in cand:
                        if x == best_w:
                            continue
                        edge = (min(best_w, x), max(best_w, x))
                        if edge in current_edges:
                            new_cand.add(x)
                    cand = new_cand

                # Add clique to partition
                partition.append(list(clique))

                # Remove all edges in this clique from current_edges
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        edge_to_remove = (min(a, b), max(a, b))
                        if edge_to_remove in current_edges:
                            current_edges.remove(edge_to_remove)

            if len(partition) < best_count:
                best_count = len(partition)
                best_partition = partition

        # Fallback if no valid partition was found
        if best_partition is None:
            best_partition = [[u, v] for u, v in edges]

        return best_partition

    # Solve both instances with time budget split
    result1 = solve_instance(32, 0.6, 601, num_trials=30, time_limit=50)
    result2 = solve_instance(64, 0.55, 602, num_trials=50, time_limit=60)

    return [result1, result2]
