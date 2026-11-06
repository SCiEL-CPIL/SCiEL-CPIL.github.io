"""
Snake-in-the-box in Q_9: Longest induced path via optimized backtracking + local search.

Strategy:
1. Build initial snake using bitset-accelerated backtracking (greedy with restarts).
2. Improve via local search: find longer detours between non-adjacent pairs.
3. Use bitset operations (Python int as bitset of 512 vertices) for fast validity checks.
"""

import time
import random

N = 9
V = 1 << N  # 512 vertices

# Precompute neighbors for each vertex
_NEIGHBORS = [[] for _ in range(V)]
for _v in range(V):
    for _b in range(N):
        _NEIGHBORS[_v].append(_v ^ (1 << _b))

# Precompute bitmask for each vertex (for bitset operations)
_VMASK = [1 << v for v in range(V)]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    def bitset_add(mask, v):
        return mask | _VMASK[v]

    def bitset_has(mask, v):
        return mask & _VMASK[v]

    def count_bits(mask):
        return mask.bit_count()

    def valid_extension(path, used_mask, candidate):
        """Check if candidate can be appended to path.
        - candidate not in used_mask
        - candidate not adjacent to any vertex in path[:-1]
        """
        if bitset_has(used_mask, candidate):
            return False
        # Check adjacency to path[:-1] (all except last)
        for v in path[:-1]:
            if (candidate ^ v).bit_count() == 1:
                return False
        return True

    def greedy_extend(start, time_limit=5.0):
        """Greedy extension from start vertex with random tie-breaking."""
        path = [start]
        used_mask = _VMASK[start]
        t0 = time.time()

        while time.time() - t0 < time_limit:
            cur = path[-1]
            candidates = []
            for nb in _NEIGHBORS[cur]:
                if not bitset_has(used_mask, nb):
                    # Check validity
                    valid = True
                    for v in path[:-1]:
                        if (nb ^ v).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(nb)

            if not candidates:
                break

            # Score candidates: prefer those with more future options
            def score(c):
                # Count valid next extensions from c
                cnt = 0
                for nb2 in _NEIGHBORS[c]:
                    if not bitset_has(used_mask, nb2):
                        ok = True
                        for v in path[:-1]:
                            if (nb2 ^ v).bit_count() == 1:
                                ok = False
                                break
                        if ok:
                            cnt += 1
                return cnt

            if len(candidates) <= 3:
                # Just pick best
                best = max(candidates, key=score)
            else:
                # Softmax-like selection: pick top-3 randomly weighted
                scored = [(score(c), c) for c in candidates]
                scored.sort(reverse=True)
                top = scored[:min(3, len(scored))]
                # Weighted random from top
                weights = [s + 1 for s, _ in top]
                r = random.random() * sum(weights)
                acc = 0
                chosen = top[-1][1]
                for w, c in top:
                    acc += w
                    if r <= acc:
                        chosen = c
                        break
                best = chosen

            path.append(best)
            used_mask = bitset_add(used_mask, best)

        return path

    def find_detour(path, i, j, max_depth):
        """Find a longer path from path[i] to path[j] through unused vertices.
        Returns the detour path (including endpoints) or None if no longer path found.
        """
        start_v = path[i]
        end_v = path[j]
        seg_len = j - i  # number of edges in original segment

        # Vertices in the segment (exclusive of endpoints for the "forbidden" set)
        # Actually, we want a path from start_v to end_v that doesn't use any vertex in path except endpoints
        forbidden_mask = 0
        for v in path:
            if v != start_v and v != end_v:
                forbidden_mask = bitset_add(forbidden_mask, v)

        # DFS with pruning
        best_detour = None
        found_any = [False]

        def dfs(cur, path_so_far, used_detour_mask):
            if time.time() > deadline:
                return
            if found_any[0] and best_detour is not None and len(path_so_far) <= len(best_detour):
                # Prune: we already have a longer one
                return
            if len(path_so_far) > max_depth:
                return

            if cur == end_v and len(path_so_far) > seg_len + 1:
                if best_detour is None or len(path_so_far) > len(best_detour):
                    best_detour = list(path_so_far)
                    found_any[0] = True
                return

            # Try extending
            for nb in _NEIGHBORS[cur]:
                if nb == start_v and len(path_so_far) > 1:
                    continue  # Don't go back to start
                if bitset_has(forbidden_mask, nb):
                    continue
                if bitset_has(used_detour_mask, nb):
                    continue
                # Check: nb must not be adjacent to any vertex in path_so_far[:-1]
                # (to maintain induced path property within the detour)
                valid = True
                for v in path_so_far[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    continue

                path_so_far.append(nb)
                new_used = bitset_add(used_detour_mask, nb)
                dfs(nb, path_so_far, new_used)
                path_so_far.pop()

        dfs(start_v, [start_v], _VMASK[start_v])
        return best_detour

    def local_search(path, max_rounds=200):
        """Improve path by finding longer detours."""
        best_path = list(path)
        best_len = len(best_path)
        no_improve = 0

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break

            # Pick a random pair (i, j) with j > i+1
            L = len(best_path)
            if L < 4:
                break

            # Focus on shorter segments for faster detour search
            max_seg = min(L - 2, 8)
            seg_len = random.randint(2, max_seg)
            i = random.randint(0, L - seg_len - 1)
            j = i + seg_len

            # Search for detour with depth limit
            depth_limit = seg_len + random.randint(2, 5)
            detour = find_detour(best_path, i, j, depth_limit)

            if detour is not None and len(detour) > seg_len + 1:
                new_path = best_path[:i] + detour + best_path[j:]
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(new_path)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve > 30:
                break

        return best_path

    def try_extend_both_ends(path):
        """Try to extend the snake at both ends."""
        best = list(path)

        # Extend at the end
        cur = best[-1]
        for nb in _NEIGHBORS[cur]:
            if not bitset_has(_build_mask(best), nb):
                valid = True
                for v in best[:-1]:
                    if (nb ^ v).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    best.append(nb)
                    break

        # Extend at the beginning
        cur = best[0]
        for nb in _NEIGHBORS[cur]:
            if not bitset_has(_build_mask(best), nb):
                valid = True
                for v in best[1:]:
                    if (nb ^ v).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    best.insert(0, nb)
                    break

        return best

    def _build_mask(path):
        m = 0
        for v in path:
            m |= _VMASK[v]
        return m

    # --- Main search ---
    best_path = []

    # Phase 1: Multiple greedy starts
    starts = list(range(V))
    random.shuffle(starts)

    for s in starts:
        if time_left() < 5.0:
            break
        p = greedy_extend(s, time_limit=3.0)
        if len(p) > len(best_path):
            best_path = p

    # Phase 2: Local search on best
    if time_left() > 3.0:
        improved = local_search(best_path, max_rounds=100)
        if len(improved) > len(best_path):
            best_path = improved

    # Phase 3: Extend both ends
    if time_left() > 2.0:
        extended = try_extend_both_ends(best_path)
        if len(extended) > len(best_path):
            best_path = extended
            # Local search again
            if time_left() > 2.0:
                improved2 = local_search(best_path, max_rounds=50)
                if len(improved2) > len(best_path):
                    best_path = improved2

    # Phase 4: More restarts if time remains
    while time_left() > 10.0:
        s = random.randint(0, V - 1)
        p = greedy_extend(s, time_limit=3.0)
        if len(p) > len(best_path):
            best_path = p
            improved = local_search(best_path, max_rounds=50)
            if len(improved) > len(best_path):
                best_path = improved

    # Phase 5: Final extension attempts
    if time_left() > 1.0:
        for _ in range(3):
            if time_left() < 0.5:
                break
            ext = try_extend_both_ends(best_path)
            if len(ext) > len(best_path):
                best_path = ext

    return best_path
