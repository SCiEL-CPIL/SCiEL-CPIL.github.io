def construct():
    import time
    import random

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, num_trials=100, time_limit=55):
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

                # Compute degrees in current_edges
                deg = [0] * n
                for u, v in current_edges:
                    deg[u] += 1
                    deg[v] += 1

                # Strategy: Pick a random edge, but weight by degree sum
                # To improve quality, we try a few random edges and pick the one that
                # yields the largest initial clique potential (common neighbors)
                
                # For efficiency, we just pick one random edge but use a better expansion
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
                # We use a heuristic: pick the candidate that maximizes the number of
                # edges it covers within the current candidate set (or just max degree)
                # Let's stick to max degree in current_edges for speed, but with a twist:
                # If there's a tie, pick randomly.
                
                while cand:
                    # Find the best candidate
                    # We want to pick w that is connected to as many other candidates as possible
                    # to keep the candidate set large? Or just max degree?
                    # Max degree in current_edges is a good proxy for "central" vertex.
                    
                    best_w = None
                    max_score = -1
                    
                    # To speed up, we can sample if cand is large, but n<=64 so it's small.
                    for w in cand:
                        # Score: degree in current_edges
                        score = deg[w]
                        # Tie-breaking: random
                        if score > max_score:
                            max_score = score
                            best_w = w
                        elif score == max_score and best_w is not None:
                            # Random tie-break
                            if trial_rng.random() < 0.5:
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
                        # This shouldn't happen if cand is maintained correctly,
                        # but if it does, remove and try next
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
    # Total time limit is 150s. We leave 10s buffer. So 140s available.
    # Instance 1 (n=32) is smaller, give it less time.
    # Instance 2 (n=64) is larger, give it more time.
    result1 = solve_instance(32, 0.6, 601, num_trials=200, time_limit=30)
    result2 = solve_instance(64, 0.55, 602, num_trials=400, time_limit=100)

    return [result1, result2]
