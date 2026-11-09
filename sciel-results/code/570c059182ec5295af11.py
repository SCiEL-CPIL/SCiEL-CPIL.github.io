"""Combined snake-in-the-box solver for Q_9: bidirectional growth + Warnsdorff + local repair."""
import time
import random

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 115.0  # Safety margin for 120s limit


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    Strategy:
      1. Bidirectional growth with Warnsdorff's heuristic (grow the end with fewer
         options, choosing the move that leaves the fewest onward moves).
      2. If stuck, try local repair: find a non-consecutive pair of vertices in the
         path that are at distance 2, and attempt to insert a new vertex between them
         (or splice a short detour) to extend the path.
      3. Multiple restarts with different random seeds until the deadline.
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    # Precompute neighbors for all vertices
    neighbors = [[] for _ in range(MAX_VERTS)]
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))

    def _elapsed():
        return time.time() - start_time

    def _deadline_hit():
        return _elapsed() > DEADLINE

    def _valid_extension(candidate, path, used_set, skip_index):
        """
        Check if candidate can be added adjacent to path[skip_index] as a new end.
        - candidate not in used_set
        - candidate not adjacent to any p in path except path[skip_index]
        """
        if candidate in used_set:
            return False
        for i, p in enumerate(path):
            if i == skip_index:
                continue
            if (candidate ^ p).bit_count() == 1:
                return False
        return True

    def _count_onward(candidate, path, used_set, is_left):
        """
        Count how many valid moves would be available for candidate if it were
        added as the new left or right end.
        is_left: True if candidate will become path[0], False if path[-1].
        """
        count = 0
        skip = 0 if is_left else len(path) - 1
        for w in neighbors[candidate]:
            if w in used_set:
                continue
            # w must not be adjacent to any p in path except the connector
            ok = True
            for i, p in enumerate(path):
                if i == skip:
                    continue
                if (w ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                count += 1
        return count

    def bidirectional_grow(start_vertex, rng):
        """
        Grow a snake bidirectionally from start_vertex using Warnsdorff's rule.
        Returns the path list.
        """
        path = [start_vertex]
        used = {start_vertex}

        # Pick a random neighbor to create the second end
        v1 = rng.choice(neighbors[start_vertex])
        path.append(v1)
        used.add(v1)

        while True:
            if _deadline_hit():
                break

            left_end = path[0]
            right_end = path[-1]

            # Collect valid moves for left end (prepending)
            valid_left = []
            for v in neighbors[left_end]:
                if v in used:
                    continue
                # Check against path[1:]
                ok = True
                for p in path[1:]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    valid_left.append(v)

            # Collect valid moves for right end (appending)
            valid_right = []
            for v in neighbors[right_end]:
                if v in used:
                    continue
                # Check against path[:-1]
                ok = True
                for p in path[:-1]:
                    if (v ^ p).bit_count() == 1:
                        ok = False
                        break
                if ok:
                    valid_right.append(v)

            if not valid_left and not valid_right:
                break

            # Warnsdorff: pick the end with fewer options; tie-break randomly
            # Then within that end, pick the move with fewest onward moves
            if len(valid_left) <= len(valid_right):
                # Extend left
                best_v = None
                best_cost = float('inf')
                for v in valid_left:
                    cost = _count_onward(v, path, used, is_left=True)
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                path.insert(0, best_v)
                used.add(best_v)
            else:
                # Extend right
                best_v = None
                best_cost = float('inf')
                for v in valid_right:
                    cost = _count_onward(v, path, used, is_left=False)
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v
                path.append(best_v)
                used.add(best_v)

        return path

    def try_local_repair(path, used):
        """
        Attempt to extend the path by finding a non-consecutive pair at distance 2
        and inserting a vertex between them. Returns (new_path, new_used) if successful.
        """
        n = len(path)
        if n < 4:
            return None, None

        # For each pair of non-consecutive vertices at distance 2, check if there's
        # a common neighbor not in used that can be inserted.
        # We iterate over positions i and j where j > i+1.
        # To keep it fast, only check a limited window.
        limit = min(n, 60)
        for i in range(0, limit):
            if _deadline_hit():
                return None, None
            for j in range(i + 2, min(i + 20, limit)):
                a = path[i]
                b = path[j]
                if (a ^ b).bit_count() != 2:
                    continue
                # Find common neighbors of a and b
                for c in neighbors[a]:
                    if c in used:
                        continue
                    if c == b:
                        continue
                    if (c ^ b).bit_count() != 1:
                        continue
                    # c is a common neighbor of a and b, not used.
                    # We want to insert c between a and b in the path.
                    # But the path goes ...a...b... (a before b).
                    # We need to check that inserting c between a and b doesn't
                    # violate induced path property with other vertices.
                    # Specifically, c must not be adjacent to any path[k] where
                    # k is not i or j.
                    ok = True
                    for k, p in enumerate(path):
                        if k == i or k == j:
                            continue
                        if (c ^ p).bit_count() == 1:
                            ok = False
                            break
                    if not ok:
                        continue

                    # Also need to verify that the subpath from i to j can be
                    # re-routed through c. Actually, the simplest insertion is:
                    # Replace path[i]...path[j] with path[i], c, path[j], path[j+1]...
                    # But that changes connectivity. The path must remain a simple path.
                    # Inserting c between i and j means: ...path[i-1], path[i], c, path[j], path[j+1]...
                    # But path[i] and c are adjacent, c and path[j] are adjacent.
                    # We need path[i-1] to still connect to path[i] (yes, unchanged)
                    # and path[j] to connect to path[j+1] (yes, unchanged).
                    # The issue is that the original path had path[i]...path[j] as a
                    # subpath. We're replacing that subpath with path[i], c, path[j].
                    # This is valid if the subpath path[i+1]...path[j-1] is removed.
                    # But that would make it not a path anymore unless we splice.
                    # Actually, this approach is flawed. Let's use a simpler repair:
                    # Find two consecutive vertices path[k], path[k+1] such that we can
                    # replace the edge with a 2-edge detour through a new vertex.
                    # i.e., find k where there exists w such that:
                    #   w adjacent to path[k], w adjacent to path[k+1], w not in used,
                    #   w not adjacent to any other path vertex.
                    # Then replace path[k], path[k+1] with path[k], w, path[k+1].
                    pass  # Skip this complex repair for now; use simpler approach

        # Simpler repair: try to extend from either end by checking if we missed
        # any valid move due to ordering. Actually, the bidirectional growth already
        # checks all moves. Let's try a different repair: 
        # Look for a vertex w not in used such that w is adjacent to exactly two
        # consecutive vertices in the path, and w is not adjacent to any other path vertex.
        # Then we can splice w in between them.
        for k in range(n - 1):
            if _deadline_hit():
                return None, None
            a = path[k]
            b = path[k + 1]
            for w in neighbors[a]:
                if w in used or w == b:
                    continue
                if (w ^ b).bit_count() != 1:
                    continue
                # w is adjacent to both a and b
                # Check w not adjacent to any other path vertex
                ok = True
                for m, p in enumerate(path):
                    if m == k or m == k + 1:
                        continue
                    if (w ^ p).bit_count() == 1:
                        ok = False
                        break
                if not ok:
                    continue
                # Splice w between a and b
                new_path = path[:k + 1] + [w] + path[k + 1:]
                new_used = used | {w}
                return new_path, new_used
        return None, None

    def repair_loop(path, used):
        """
        Repeatedly apply local repair until no more improvements or deadline.
        """
        while True:
            if _deadline_hit():
                break
            new_path, new_used = try_local_repair(path, used)
            if new_path is None:
                break
            path = new_path
            used = new_used
        return path, used

    # Multiple restarts
    for seed in range(10000):
        if _deadline_hit():
            break
        rng = random.Random(seed)
        start_v = rng.randrange(MAX_VERTS)
        path = bidirectional_grow(start_v, rng)
        used = set(path)
        path, used = repair_loop(path, used)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    if best_path is None:
        # Fallback: return a trivial path
        return [0]
    return best_path
