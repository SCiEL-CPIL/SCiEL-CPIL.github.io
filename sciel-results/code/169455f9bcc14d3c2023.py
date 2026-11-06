"""Rewritten program: maximize edges of a triangle-free, non-3-colorable graph on 16 vertices."""
import time
import random

def construct():
    deadline = time.time() + 145.0

    N = 16
    ALL_MASK = (1 << N) - 1

    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    _colorable_cache = {}

    def is_3_colorable(adj):
        """Fast 3-colorability check using bitmask backtracking with dynamic vertex ordering."""
        key = tuple(adj)
        if key in _colorable_cache:
            return _colorable_cache[key]

        degrees = [bin(adj[i]).count('1') for i in range(N)]

        # Dynamic most-constrained-first: pick vertex with fewest available colors
        # Use a recursive approach that picks the best vertex at each step
        color_masks = [0, 0, 0]
        remaining = ALL_MASK

        def backtrack(rem):
            if rem == 0:
                return True

            # Find vertex with fewest available colors (most constrained)
            best_v = -1
            best_count = 99
            for v in range(N):
                if not (rem & (1 << v)):
                    continue
                nmask = adj[v]
                forbidden = 0
                if nmask & color_masks[0]:
                    forbidden |= 1
                if nmask & color_masks[1]:
                    forbidden |= 2
                if nmask & color_masks[2]:
                    forbidden |= 4
                avail = 3 - bin(forbidden).count('1')
                if avail < best_count:
                    best_count = avail
                    best_v = v
                    if avail == 0:
                        break

            if best_v == -1:
                return True
            if best_count == 0:
                return False

            v = best_v
            nmask = adj[v]
            forbidden = 0
            if nmask & color_masks[0]:
                forbidden |= 1
            if nmask & color_masks[1]:
                forbidden |= 2
            if nmask & color_masks[2]:
                forbidden |= 4

            # Try colors
            if not (forbidden & 1):
                color_masks[0] |= (1 << v)
                if backtrack(rem & ~(1 << v)):
                    return True
                color_masks[0] &= ~(1 << v)

            if not (forbidden & 2):
                color_masks[1] |= (1 << v)
                if backtrack(rem & ~(1 << v)):
                    return True
                color_masks[1] &= ~(1 << v)

            if not (forbidden & 4):
                color_masks[2] |= (1 << v)
                if backtrack(rem & ~(1 << v)):
                    return True
                color_masks[2] &= ~(1 << v)

            return False

        result = backtrack(ALL_MASK)
        _colorable_cache[key] = result
        return result

    def count_edges(adj):
        total = 0
        for i in range(N):
            total += bin(adj[i]).count('1')
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def is_triangle_free_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        return (adj[u] & adj[v]) == 0

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_degree(adj, v):
        return bin(adj[v]).count('1')

    # Build the Grötzsch graph (Mycielski of C5) on 11 vertices, rest isolated
    def make_grotzsch():
        adj = [0] * N
        # C5: vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        # Mycielski: vertices 5-9 are copies, vertex 10 is the apex
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        return adj

    # Greedy edge addition: try to add edges in a smart order
    def greedy_add(adj, max_iter=None):
        """Greedily add edges that maintain triangle-free and non-3-colorable."""
        cnt = count_edges(adj)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort candidate edges by combined degree descending
            candidates = []
            for i, j in all_pairs:
                if not (adj[i] & (1 << j)):
                    if is_triangle_free_add(adj, i, j):
                        deg_sum = get_degree(adj, i) + get_degree(adj, j)
                        candidates.append((deg_sum, i, j))
            candidates.sort(key=lambda x: -x[0])

            for deg_sum, u, v in candidates:
                if time.time() > deadline:
                    return adj, cnt
                if adj[u] & (1 << v):
                    continue
                if not is_triangle_free_add(adj, u, v):
                    continue
                add_edge(adj, u, v)
                if not is_3_colorable(adj):
                    cnt += 1
                    improved = True
                else:
                    remove_edge(adj, u, v)
        return adj, cnt

    # Local search: add + 2-swap moves (remove 2 edges, add 2 edges)
    def local_search(adj, cnt, max_no_improve=2000):
        """Local search with add and 2-swap moves."""
        best_adj = adj[:]
        best_cnt = cnt
        no_improve = 0

        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            improved = False

            # Try adding edges
            candidates = []
            for i, j in all_pairs:
                if not (adj[i] & (1 << j)):
                    if is_triangle_free_add(adj, i, j):
                        deg_sum = get_degree(adj, i) + get_degree(adj, j)
                        candidates.append((deg_sum, i, j))
            candidates.sort(key=lambda x: -x[0])

            for deg_sum, u, v in candidates:
                if time.time() > deadline - 5.0:
                    break
                if adj[u] & (1 << v):
                    continue
                if not is_triangle_free_add(adj, u, v):
                    continue
                add_edge(adj, u, v)
                if not is_3_colorable(adj):
                    cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_adj = adj[:]
                    improved = True
                else:
                    remove_edge(adj, u, v)

            if improved:
                no_improve = 0
                continue

            no_improve += 1

            # Try 2-swap moves: remove two edges, add two edges
            existing = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            if len(existing) < 2:
                break
            random.shuffle(existing)

            found_swap = False
            # Limit number of swap attempts per iteration
            max_swap_attempts = min(30, len(existing) * (len(existing) - 1) // 2)
            attempts = 0

            for idx1 in range(min(len(existing), 15)):
                if time.time() > deadline - 5.0 or attempts >= max_swap_attempts:
                    break
                e1 = existing[idx1]
                remove_edge(adj, e1[0], e1[1])

                for idx2 in range(idx1 + 1, min(len(existing), 15)):
                    if time.time() > deadline - 5.0 or attempts >= max_swap_attempts:
                        break
                    e2 = existing[idx2]
                    if adj[e2[0]] & (1 << e2[1]):
                        continue  # already removed by e1 removal? shouldn't happen
                    remove_edge(adj, e2[0], e2[1])
                    attempts += 1

                    # Try to add two new edges
                    new_candidates = []
                    for a, b in all_pairs:
                        if (a, b) == e1 or (a, b) == e2:
                            continue
                        if adj[a] & (1 << b):
                            continue
                        if not is_triangle_free_add(adj, a, b):
                            continue
                        deg_sum = get_degree(adj, a) + get_degree(adj, b)
                        new_candidates.append((deg_sum, a, b))
                    new_candidates.sort(key=lambda x: -x[0])

                    # Try adding one edge, then another
                    for deg1, a1, b1 in new_candidates[:20]:
                        if time.time() > deadline - 5.0:
                            break
                        add_edge(adj, a1, b1)
                        if not is_3_colorable(adj):
                            # Try adding one more
                            for deg2, a2, b2 in new_candidates[:20]:
                                if time.time() > deadline - 5.0:
                                    break
                                if (a2, b2) == (a1, b1):
                                    continue
                                if adj[a2] & (1 << b2):
                                    continue
                                if not is_triangle_free_add(adj, a2, b2):
                                    continue
                                add_edge(adj, a2, b2)
                                if not is_3_colorable(adj):
                                    cnt += 1  # net +1 (removed 2, added 3? no, added 2 net 0... wait)
                                    # Actually: removed 2, added 2 = same count
                                    # But we need cnt to reflect reality
                                    cnt = count_edges(adj)
                                    if cnt > best_cnt:
                                        best_cnt = cnt
                                        best_adj = adj[:]
                                    found_swap = True
                                    break
                                else:
                                    remove_edge(adj, a2, b2)
                            if found_swap:
                                break
                            else:
                                remove_edge(adj, a1, b1)
                        else:
                            remove_edge(adj, a1, b1)

                        if found_swap:
                            break

                    if found_swap:
                        break
                    else:
                        add_edge(adj, e2[0], e2[1])

                if found_swap:
                    break
                else:
                    add_edge(adj, e1[0], e1[1])

            if not found_swap:
                no_improve += 1

        return best_adj, best_cnt

    # Phase 1: Generate diverse initial solutions
    best_adj = None
    best_cnt = 0

    # Start from Grötzsch graph
    adj0 = make_grotzsch()
    adj0, cnt0 = greedy_add(adj0)
    if cnt0 > best_cnt and not is_3_colorable(adj0):
        best_adj = adj0[:]
        best_cnt = cnt0

    # Random triangle-free starts
    for attempt in range(30):
        if time.time() > deadline - 20.0:
            break
        adj = [0] * N
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_triangle_free_add(adj, u, v):
                add_edge(adj, u, v)
        # Greedy add
        adj, cnt = greedy_add(adj)
        if cnt > best_cnt and not is_3_colorable(adj):
            best_adj = adj[:]
            best_cnt = cnt

    # Phase 2: Local search from best
    if best_adj is not None and time.time() < deadline - 10.0:
        best_adj, best_cnt = local_search(best_adj, best_cnt, max_no_improve=1500)

    # Phase 3: More restarts with perturbation
    if best_adj is not None and time.time() < deadline - 10.0:
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            adj = best_adj[:]
            # Perturb: remove 2-5 random edges
            existing = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            random.shuffle(existing)
            num_remove = random.randint(2, 5)
            for u, v in existing[:num_remove]:
                remove_edge(adj, u, v)

            adj, cnt = greedy_add(adj)
            if cnt > best_cnt and not is_3_colorable(adj):
                best_adj = adj[:]
                best_cnt = cnt

            # Quick local search
            if time.time() < deadline - 10.0:
                adj2, cnt2 = local_search(adj, cnt, max_no_improve=300)
                if cnt2 > best_cnt and not is_3_colorable(adj2):
                    best_adj = adj2[:]
                    best_cnt = cnt2

    # Final: try to add any remaining edges
    if best_adj is not None and time.time() < deadline - 3.0:
        candidates = []
        for i, j in all_pairs:
            if not (best_adj[i] & (1 << j)):
                if is_triangle_free_add(best_adj, i, j):
                    deg_sum = get_degree(best_adj, i) + get_degree(best_adj, j)
                    candidates.append((deg_sum, i, j))
        candidates.sort(key=lambda x: -x[0])
        for deg_sum, u, v in candidates:
            if time.time() > deadline - 3.0:
                break
            if best_adj[u] & (1 << v):
                continue
            if not is_triangle_free_add(best_adj, u, v):
                continue
            add_edge(best_adj, u, v)
            if not is_3_colorable(best_adj):
                best_cnt += 1
            else:
                remove_edge(best_adj, u, v)

    if best_adj is None:
        # Fallback: Grötzsch graph
        adj = make_grotzsch()
        return adj_to_edges(adj)

    return adj_to_edges(best_adj)
