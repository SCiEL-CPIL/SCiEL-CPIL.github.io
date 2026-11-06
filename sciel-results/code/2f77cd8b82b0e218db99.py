<<<<<<< SEARCH
def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:5] + vertices[:5]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Greedy longest path from start: always extend to unvisited neighbor
        # with highest degree (keeps options open for closing the cycle)
        path = [start]
        visited = {start}

        while True:
            u = path[-1]
            candidates = [(len(adj[v]), v) for v in adj[u] if v not in visited]
            if not candidates:
                break
            candidates.sort(reverse=True)
            # Randomize among top-3 to add diversity
            top_k = min(3, len(candidates))
            _, next_v = rng.choice(candidates[:top_k])
            path.append(next_v)
            visited.add(next_v)

        # Check if we can close the cycle directly
        if len(path) >= 3 and start in adj[path[-1]]:
            if len(path) > len(best_cycle):
                best_cycle = list(path)

        # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
        # Take the longest such cycle
        for i in range(len(path) - 2):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(len(path) - 1, i + 1, -1):
                if path[i] in adj[path[j]]:
                    cycle = path[i:j + 1]
                    if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                        best_cycle = list(cycle)
                    break  # Found the longest cycle starting at i

    return best_cycle
=======
def _find_longest_cycle_greedy(adj, n, seed, deadline=None):
    """
    Find a long cycle using greedy longest-path extension with backtracking.
    Returns a list of vertices forming a cycle, or empty list if none found.
    """
    if not adj:
        return []

    best_cycle = []
    vertices = [v for v in adj if adj[v]]
    if not vertices:
        return []

    rng = random.Random(seed)
    rng.shuffle(vertices)

    # Try multiple starting vertices (top by degree + random)
    deg_sorted = sorted(vertices, key=lambda v: -len(adj[v]))
    starts = list(set(deg_sorted[:8] + vertices[:8]))
    rng.shuffle(starts)

    for start in starts:
        if deadline is not None and time.time() > deadline:
            break
        if not adj[start]:
            continue

        # Multiple path extensions from start with different strategies
        for strategy in range(4):
            if deadline is not None and time.time() > deadline:
                break

            path = [start]
            visited = {start}

            while True:
                u = path[-1]
                candidates = [v for v in adj[u] if v not in visited]
                if not candidates:
                    break

                if strategy == 0:
                    # Highest degree first
                    candidates.sort(key=lambda v: -len(adj[v]))
                    next_v = candidates[0]
                elif strategy == 1:
                    # Lowest degree first (force through bottlenecks)
                    candidates.sort(key=lambda v: len(adj[v]))
                    next_v = candidates[0]
                elif strategy == 2:
                    # Random choice
                    next_v = rng.choice(candidates)
                else:
                    # Weighted: prefer higher degree but with randomness
                    weights = [len(adj[v]) + 1 for v in candidates]
                    total = sum(weights)
                    r = rng.random() * total
                    cumulative = 0
                    next_v = candidates[-1]
                    for v, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_v = v
                            break

                path.append(next_v)
                visited.add(next_v)

            # Check if we can close the cycle directly
            if len(path) >= 3 and start in adj[path[-1]]:
                if len(path) > len(best_cycle):
                    best_cycle = list(path)

            # Look for any back-edge (path[i], path[j]) with i < j-1 to form a cycle
            # Take the longest such cycle
            for i in range(len(path) - 2):
                if deadline is not None and time.time() > deadline:
                    break
                for j in range(len(path) - 1, i + 1, -1):
                    if path[i] in adj[path[j]]:
                        cycle = path[i:j + 1]
                        if len(cycle) >= 3 and len(cycle) > len(best_cycle):
                            best_cycle = list(cycle)
                        break  # Found the longest cycle starting at i

    return best_cycle
>>>>>>> REPLACE

