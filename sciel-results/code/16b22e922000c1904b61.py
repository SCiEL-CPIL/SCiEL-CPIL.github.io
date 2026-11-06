"""Rewritten program: maximize edges of a triangle-free, non-3-colorable graph on 16 vertices."""
import time
import random

def construct():
    deadline = time.time() + 145.0

    N = 16
    ALL_MASK = (1 << N) - 1

    # Precompute all possible edges
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_pairs_mask = {}
    for i, j in all_pairs:
        all_pairs_mask[(i, j)] = (1 << j) | (1 << i)

    _colorable_cache = {}

    def is_3_colorable(adj):
        """Fast 3-colorability check using bitmask backtracking with caching."""
        key = tuple(adj)
        if key in _colorable_cache:
            return _colorable_cache[key]

        # Order vertices by degree descending (more constrained first)
        degrees = [bin(adj[i]).count('1') for i in range(N)]
        order = sorted(range(N), key=lambda v: -degrees[v])

        color_masks = [0, 0, 0]

        def backtrack(pos):
            if pos == N:
                return True

            v = order[pos]
            nmask = adj[v]

            # Determine forbidden colors
            forbidden = 0
            if nmask & color_masks[0]:
                forbidden |= 1
            if nmask & color_masks[1]:
                forbidden |= 2
            if nmask & color_masks[2]:
                forbidden |= 4

            if forbidden == 7:
                return False

            # Try colors in order of least constrained (fewest conflicts)
            # Try color 0 first if available
            if not (forbidden & 1):
                color_masks[0] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[0] &= ~(1 << v)

            if not (forbidden & 2):
                color_masks[1] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[1] &= ~(1 << v)

            if not (forbidden & 4):
                color_masks[2] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[2] &= ~(1 << v)

            return False

        result = backtrack(0)
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
        # Sort candidate edges by combined degree descending
        candidates = []
        for i, j in all_pairs:
            if not (adj[i] & (1 << j)):
                deg_sum = bin(adj[i]).count('1') + bin(adj[j]).count('1')
                candidates.append((deg_sum, i, j))
        candidates.sort(key=lambda x: -x[0])

        for deg_sum, u, v in candidates:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not is_triangle_free_add(adj, u, v):
                continue
            add_edge(adj, u, v)
            if not is_3_colorable(adj):
                cnt += 1
            else:
                remove_edge(adj, u, v)
        return adj, cnt

    # Local search: add + swap moves
    def local_search(adj, cnt, max_no_improve=3000):
        """Local search with add and swap moves."""
        best_adj = adj[:]
        best_cnt = cnt
        no_improve = 0

        while time.time() < deadline and no_improve < max_no_improve:
            improved = False

            # Try adding edges
            candidates = []
            for i, j in all_pairs:
                if not (adj[i] & (1 << j)):
                    if is_triangle_free_add(adj, i, j):
                        deg_sum = bin(adj[i]).count('1') + bin(adj[j]).count('1')
                        candidates.append((deg_sum, i, j))
            candidates.sort(key=lambda x: -x[0])

            for deg_sum, u, v in candidates:
                if time.time() > deadline:
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

            # Try swap moves: remove one edge, add one (or two)
            existing = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            random.shuffle(existing)

            found_swap = False
            for u, v in existing:
                if time.time() > deadline:
                    break
                if len(existing) > 200:
                    break  # limit swap attempts

                remove_edge(adj, u, v)

                # Try to add one edge
                for a, b in all_pairs:
                    if time.time() > deadline:
                        break
                    if (a, b) == (u, v):
                        continue
                    if adj[a] & (1 << b):
                        continue
                    if not is_triangle_free_add(adj, a, b):
                        continue
                    add_edge(adj, a, b)
                    if not is_3_colorable(adj):
                        # Net same edge count, but structure changed
                        # Try to add one more
                        for c, d in all_pairs:
                            if time.time() > deadline:
                                break
                            if (c, d) == (a, b) or (c, d) == (u, v):
                                continue
                            if adj[c] & (1 << d):
                                continue
                            if not is_triangle_free_add(adj, c, d):
                                continue
                            add_edge(adj, c, d)
                            if not is_3_colorable(adj):
                                cnt += 1
                                if cnt > best_cnt:
                                    best_cnt = cnt
                                    best_adj = adj[:]
                                found_swap = True
                                break
                            else:
                                remove_edge(adj, c, d)
                        if found_swap:
                            no_improve = 0
                            break
                        else:
                            remove_edge(adj, a, b)
                    else:
                        remove_edge(adj, a, b)

                if found_swap:
                    break
                else:
                    add_edge(adj, u, v)

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
    for attempt in range(40):
        if time.time() > deadline:
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
    if best_adj is not None:
        best_adj, best_cnt = local_search(best_adj, best_cnt, max_no_improve=2000)

    # Phase 3: More restarts with perturbation
    if best_adj is not None and time.time() < deadline:
        for restart in range(100):
            if time.time() > deadline:
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
            if time.time() < deadline:
                adj2, cnt2 = local_search(adj, cnt, max_no_improve=500)
                if cnt2 > best_cnt and not is_3_colorable(adj2):
                    best_adj = adj2[:]
                    best_cnt = cnt2

    # Final: try to add any remaining edges
    if best_adj is not None:
        for i, j in all_pairs:
            if time.time() > deadline:
                break
            if not (best_adj[i] & (1 << j)):
                if is_triangle_free_add(best_adj, i, j):
                    add_edge(best_adj, i, j)
                    if not is_3_colorable(best_adj):
                        best_cnt += 1
                    else:
                        remove_edge(best_adj, i, j)

    if best_adj is None:
        # Fallback: Grötzsch graph
        adj = make_grotzsch()
        return adj_to_edges(adj)

    return adj_to_edges(best_adj)
