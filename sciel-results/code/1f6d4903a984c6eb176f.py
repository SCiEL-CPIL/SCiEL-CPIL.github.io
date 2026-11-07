"""Improved packing: large-first backtracking with better heuristics, more restarts,
and a dedicated high-precision polish pass."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    trees = instance_trees(20, 611)
    n = 20

    # Precompute tree adjacency, BFS orders, and degrees
    tree_info = []
    for idx in range(n):
        tree = trees[idx]
        k = len(tree) + 1
        if k == 1:
            tree_info.append((tree, [], [], [0], [0]))
            continue
        adj = [[] for _ in range(k)]
        for u, v in tree:
            adj[u].append(v)
            adj[v].append(u)
        # BFS from vertex 0
        parent = [-1] * k
        order = [0]
        visited = [False] * k
        visited[0] = True
        queue = [0]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        deg = [len(adj[i]) for i in range(k)]
        tree_info.append((tree, adj, parent, order, deg))

    # Sort trees by size descending for packing order
    tree_order = list(range(n))
    tree_order.sort(key=lambda i: len(trees[i]), reverse=True)

    def find_embedding(tree_idx, used_edges, rng, strict=False):
        """Find injective embedding of tree tree_idx into K20 avoiding used_edges."""
        tree, adj, parent, order, deg = tree_info[tree_idx]
        k = len(order)
        if k == 1:
            for hv in range(n):
                return [hv]
            return None

        host = [-1] * k
        used_hosts = set()

        def get_candidates(tv, p_tv, p_hv):
            cands = []
            for hv in range(n):
                if hv in used_hosts:
                    continue
                if p_tv != -1:
                    e = (min(hv, p_hv), max(hv, p_hv))
                    if e in used_edges:
                        continue
                cands.append(hv)
            return cands

        def future_options_score(hv):
            """Lower score = fewer options = more constrained (prefer these)."""
            cnt = 0
            for nh in range(n):
                if nh in used_hosts or nh == hv:
                    continue
                e = (min(hv, nh), max(hv, nh))
                if e not in used_edges:
                    cnt += 1
            return cnt

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True

            tv = order[idx]
            p_tv = parent[tv]
            p_hv = host[p_tv] if p_tv != -1 else -1

            cands = get_candidates(tv, p_tv, p_hv)
            if not cands:
                return False

            # Sort by future options (MRV heuristic)
            if len(cands) > 2:
                cands.sort(key=future_options_score)

            # Add randomness to break symmetry (unless in strict mode)
            if not strict and rng.random() < 0.25 and len(cands) > 3:
                i = rng.randrange(len(cands) - 1)
                cands[i], cands[i+1] = cands[i+1], cands[i]

            for hv in cands:
                host[tv] = hv
                used_hosts.add(hv)
                if backtrack(idx + 1):
                    return True
                used_hosts.remove(hv)
                host[tv] = -1

            return False

        if backtrack(0):
            return host[:]
        return None

    def solve_with_backtracking(seed, order, max_attempts=15):
        """
        Recursive backtracking over tree assignments.
        Tries to embed trees in given order; if a tree fails,
        backtracks to try different embedding for a previous tree.
        """
        rng = Random(seed)
        used = set()
        res = [None] * n

        def try_place(pos):
            if time.time() > deadline:
                return False
            if pos == len(order):
                return True

            tree_idx = order[pos]
            tree, _, _, _, _ = tree_info[tree_idx]
            k = len(tree) + 1

            # Try multiple embeddings for this tree
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return False
                embedding = find_embedding(tree_idx, used, rng)
                if embedding is None:
                    break

                # Check if this embedding is valid (all edges unused)
                valid = True
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used:
                        valid = False
                        break
                if not valid:
                    break

                # Place this tree
                res[tree_idx] = embedding
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.add(e)

                # Recurse
                if try_place(pos + 1):
                    return True

                # Undo
                for u, v in tree:
                    hu, hv = embedding[u], embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    used.discard(e)
                res[tree_idx] = None

            return False

        if try_place(0):
            return res, sum(len(trees[i]) for i in range(n) if res[i] is not None)
        else:
            partial_score = sum(len(trees[i]) for i in range(n) if res[i] is not None)
            return res, partial_score

    def polish_partial(res, used, order):
        """
        Try to improve a partial packing by re-embedding already-placed trees
        in a different order, using strict (deterministic) embedding.
        Returns (improved_res, improved_score).
        """
        placed_indices = [i for i in range(n) if res[i] is not None]
        if not placed_indices:
            return res, sum(len(trees[i]) for i in range(n) if res[i] is not None)

        # Recompute used edges from current res
        used = set()
        for i in range(n):
            if res[i] is not None:
                for u, v in trees[i]:
                    hu, hv = res[i][u], res[i][v]
                    e = (min(hu, hv), max(hu, hv))
                    used.add(e)

        # Try to fill in unplaced trees first, then attempt full packing
        unplaced = [i for i in range(n) if res[i] is None]
        if not unplaced:
            return res, sum(len(trees[i]) for i in range(n) if res[i] is not None)

        # Create a new ordering: unplaced first (by size desc), then placed
        unplaced.sort(key=lambda i: -len(trees[i]))
        placed_sorted = sorted(placed_indices, key=lambda i: -len(trees[i]))
        full_order = unplaced + placed_sorted

        # Clear all and re-embed everything in the new order
        new_res = [None] * n
        new_used = set()

        # Use strict deterministic embedding for polish
        polish_rng = Random(0)

        def polish_place(pos):
            if time.time() > deadline:
                return False
            if pos == len(full_order):
                return True

            tree_idx = full_order[pos]
            tree = trees[tree_idx]

            embedding = find_embedding(tree_idx, new_used, polish_rng, strict=True)
            if embedding is None:
                return False

            # Validate
            valid = True
            for u, v in tree:
                hu, hv = embedding[u], embedding[v]
                e = (min(hu, hv), max(hu, hv))
                if e in new_used:
                    valid = False
                    break
            if not valid:
                return False

            new_res[tree_idx] = embedding
            for u, v in tree:
                hu, hv = embedding[u], embedding[v]
                e = (min(hu, hv), max(hu, hv))
                new_used.add(e)

            if polish_place(pos + 1):
                return True

            # Undo
            for u, v in tree:
                hu, hv = embedding[u], embedding[v]
                e = (min(hu, hv), max(hu, hv))
                new_used.discard(e)
            new_res[tree_idx] = None
            return False

        if polish_place(0):
            return new_res, sum(len(trees[i]) for i in range(n) if new_res[i] is not None)
        else:
            return res, sum(len(trees[i]) for i in range(n) if res[i] is not None)

    best_res = None
    best_score = 0

    # Try multiple seeds and orders
    seed_counter = 0
    for attempt in range(800):
        if time.time() > deadline:
            break
        seed_counter += 1

        # Alternate between different orderings
        r = attempt % 11
        if r == 0:
            order = tree_order  # descending
        elif r == 1:
            order = tree_order[::-1]  # ascending
        elif r == 2:
            # Random shuffle
            rng_tmp = Random(attempt)
            order = list(range(n))
            rng_tmp.shuffle(order)
        elif r == 3:
            # Descending but with small trees first among large ones
            order = sorted(range(n), key=lambda i: -len(trees[i]))
            rng_tmp = Random(attempt)
            for i in range(0, n, 3):
                group = order[i:i+3]
                rng_tmp.shuffle(group)
                order[i:i+3] = group
        elif r == 4:
            # Random but biased toward large trees
            order = list(range(n))
            rng_tmp = Random(attempt)
            rng_tmp.shuffle(order)
            if rng_tmp.random() < 0.6:
                max_idx = order.index(max(range(n), key=lambda i: len(trees[i])))
                order[0], order[max_idx] = order[max_idx], order[0]
        elif r == 5:
            # Largest two first, then rest random
            order = list(range(n))
            rng_tmp = Random(attempt)
            rng_tmp.shuffle(order)
            top2 = sorted(range(n), key=lambda i: -len(trees[i]))[:2]
            rest = [i for i in order if i not in top2]
            order = top2 + rest
        elif r == 6:
            # Reverse of tree_order with slight shuffle in middle
            order = tree_order[::-1][:]
            rng_tmp = Random(attempt)
            mid_start = n // 4
            mid_end = 3 * n // 4
            mid = order[mid_start:mid_end]
            rng_tmp.shuffle(mid)
            order[mid_start:mid_end] = mid
        elif r == 7:
            # Group by size parity
            even = [i for i in range(n) if len(trees[i]) % 2 == 0]
            odd = [i for i in range(n) if len(trees[i]) % 2 == 1]
            even.sort(key=lambda i: -len(trees[i]))
            odd.sort(key=lambda i: -len(trees[i]))
            order = odd + even
        elif r == 8:
            # Fully random
            rng_tmp = Random(attempt * 3 + 7)
            order = list(range(n))
            rng_tmp.shuffle(order)
        elif r == 9:
            # Large trees in random positions, small trees fill gaps
            order = list(range(n))
            rng_tmp = Random(attempt)
            rng_tmp.shuffle(order)
            # Keep largest 5 in their shuffled positions
        else:
            # Descending with random ties broken differently
            order = sorted(range(n), key=lambda i: (-len(trees[i]), rng_tmp.randbits(32) if False else i))
            rng_tmp = Random(attempt)
            for i in range(0, n, 4):
                group = order[i:i+4]
                rng_tmp.shuffle(group)
                order[i:i+4] = group

        res, score = solve_with_backtracking(seed_counter, order)

        if score > best_score:
            best_score = score
            best_res = [r if r is not None else None for r in res]

        if best_score == 190:
            break

        # If we have a near-complete packing, try polish
        if best_score >= 180 and attempt % 5 == 0:
            polished_res, polished_score = polish_partial(best_res, set(), order)
            if polished_score > best_score:
                best_score = polished_score
                best_res = polished_res

    # Final polish attempt if we're close
    if best_res is not None and best_score < 190:
        polished_res, polished_score = polish_partial(best_res, set(), tree_order)
        if polished_score > best_score:
            best_score = polished_score
            best_res = polished_res

    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])

    return final_result
