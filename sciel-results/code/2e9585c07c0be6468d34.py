import time
import random

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid(adj, deg):
        """Check validity using adjacency lists and degrees."""
        for u in range(n):
            for v in adj[u]:
                if v > u:  # Each edge once
                    # For edge (u,v), check if there exist s in N(u)\{v}, t in N(v)\{u}
                    # with s != t and deg[s] == deg[t]
                    # Group N(u)\{v} by degree
                    deg_to_s = {}
                    for s in adj[u]:
                        if s != v:
                            d = deg[s]
                            if d not in deg_to_s:
                                deg_to_s[d] = []
                            deg_to_s[d].append(s)
                    # Check each t in N(v)\{u}
                    for t in adj[v]:
                        if t != u:
                            dt = deg[t]
                            if dt in deg_to_s:
                                s_list = deg_to_s[dt]
                                if len(s_list) >= 2:
                                    return False
                                elif len(s_list) == 1:
                                    if s_list[0] != t:
                                        return False
        return True

    def build_adj_deg(edge_set):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    # Start with K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b) if a < b else (b, a))

    adj, deg = build_adj_deg(base_edges)
    best_set = set(base_edges)
    best_count = len(best_set)

    # All possible intra-partition edges
    candidates = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidates.append((A[i], A[j]))
    for i in range(7):
        for j in range(i + 1, 15):
            candidates.append((B[i], B[j]))

    # Greedy: try to add each candidate
    current_set = set(base_edges)
    current_adj, current_deg = build_adj_deg(current_set)

    random.shuffle(candidates)
    for e in candidates:
        if time.time() > deadline:
            break
        u, v = e
        if u > v:
            u, v = v, u
        if (u, v) in current_set:
            continue
        # Try adding
        new_set = set(current_set)
        new_set.add((u, v))
        new_adj, new_deg = build_adj_deg(new_set)
        if is_valid(new_adj, new_deg):
            current_set = new_set
            current_adj, current_deg = new_adj, new_deg

    if len(current_set) > best_count:
        best_set = current_set
        best_count = len(current_set)

    # Local search: try adding edges that weren't added
    not_in_current = [e for e in candidates if (min(e), max(e)) not in current_set]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(not_in_current)
        for e in not_in_current:
            if time.time() > deadline:
                break
            u, v = e
            if u > v:
                u, v = v, u
            if (u, v) in current_set:
                continue
            new_set = set(current_set)
            new_set.add((u, v))
            new_adj, new_deg = build_adj_deg(new_set)
            if is_valid(new_adj, new_deg):
                current_set = new_set
                not_in_current.remove(e)
                improved = True
                if len(current_set) > best_count:
                    best_set = current_set
                    best_count = len(current_set)
                break

    # Try multiple random restarts with different orderings
    for restart in range(20):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        current_set = set(base_edges)
        current_adj, current_deg = build_adj_deg(current_set)

        for e in candidates:
            if time.time() > deadline:
                break
            u, v = e
            if u > v:
                u, v = v, u
            if (u, v) in current_set:
                continue
            new_set = set(current_set)
            new_set.add((u, v))
            new_adj, new_deg = build_adj_deg(new_set)
            if is_valid(new_adj, new_deg):
                current_set = new_set
                current_adj, current_deg = new_adj, new_deg

        if len(current_set) > best_count:
            best_set = current_set
            best_count = len(current_set)
            # Local search
            not_in_current = [e for e in candidates if (min(e), max(e)) not in current_set]
            improved = True
            while improved and time.time() < deadline:
                improved = False
                random.shuffle(not_in_current)
                for e in not_in_current:
                    if time.time() > deadline:
                        break
                    u, v = e
                    if u > v:
                        u, v = v, u
                    if (u, v) in current_set:
                        continue
                    new_set = set(current_set)
                    new_set.add((u, v))
                    new_adj, new_deg = build_adj_deg(new_set)
                    if is_valid(new_adj, new_deg):
                        current_set = new_set
                        not_in_current.remove(e)
                        improved = True
                        if len(current_set) > best_count:
                            best_set = current_set
                            best_count = len(current_set)
                        break

    # Final validation
    final_adj, final_deg = build_adj_deg(best_set)
    if not is_valid(final_adj, final_deg):
        # Fallback to base
        best_set = set(base_edges)

    result = [list(e) for e in best_set]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])

    return result
