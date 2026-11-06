"""Snake-in-the-box in Q_9: bitmask backtracking + local search (2-opt / splice)."""
import time
import random

N = 9
MASK = (1 << N) - 1
DEADLINE_MARGIN = 2.0  # seconds before the 150s hard limit


def _time_ok(deadline):
    return time.time() < deadline


def _greedy_start():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _build_forbidden(path):
    """Return a set of vertices that cannot be inserted adjacent to any existing vertex
    in a way that would create a shortcut (i.e., differ by exactly 1 bit from a non-adjacent vertex)."""
    # A vertex v is "forbidden" if there exists p in path such that popcount(v^p)==1
    # and p is not a neighbor of v in the path (i.e., v is not path[i-1] or path[i+1] for the index of p... 
    # Actually simpler: v is forbidden if it's adjacent (diff 1) to any vertex in path that is NOT 
    # its immediate predecessor/successor in the path. But for insertion we need:
    # If we insert v between path[i] and path[i+1], then v must not be adjacent to any path[j] 
    # where j is not i-1, i, i+1, i+2. 
    # For now, just precompute adjacency conflicts globally for fast rejection.
    
    # For 2-opt: we reverse a segment. The new endpoints are path[i] and path[j+1] (or similar).
    # We need to check that the new connections don't create shortcuts.
    
    pass


def _is_valid_snake(path):
    """Check if path is a valid induced path."""
    if len(path) < 2:
        return True
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    # Check non-consecutive vertices differ by at least 2
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _local_search(path, deadline):
    """Improve the snake using 2-opt style moves: reverse segments."""
    path = list(path)
    n = len(path)
    improved = True
    while improved and _time_ok(deadline):
        improved = False
        # Try all pairs (i, j) with j - i >= 2
        for i in range(n - 2):
            if not _time_ok(deadline):
                return path
            for j in range(i + 2, n):
                # Reverse path[i..j]
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                # Check validity of the new connections
                # The only new edges are (i-1, i) and (j, j+1) in the new path
                # But actually, reversing changes the internal edges too? No, reversal of a path segment
                # keeps the same set of edges, just reversed. So the only new adjacencies are at the boundaries.
                # Wait: if we reverse path[i..j], the edge between i-1 and i becomes i-1 and j (the new element at position i).
                # And the edge between j and j+1 becomes i (the new element at position j) and j+1.
                # We need to check:
                # 1. (i-1, new_i) differ by 1 bit
                # 2. (new_j, j+1) differ by 1 bit
                # 3. No shortcuts created: new_i must not be adjacent to any non-consecutive vertex
                #    new_j must not be adjacent to any non-consecutive vertex
                
                valid = True
                # Check boundary edges
                if i > 0:
                    if (path[i-1] ^ path[j]).bit_count() != 1:
                        valid = False
                if j < n - 1:
                    if (path[i] ^ path[j+1]).bit_count() != 1:
                        valid = False
                if not valid:
                    continue
                
                # Check for shortcuts
                # new_i = path[j], new_j = path[i]
                # path[j] is now at index i. Its neighbors are path[i-1] (if i>0) and path[j-1] (if j>i).
                # Actually, the neighbors of the element at index i in the new path are:
                #   left: path[i-1] if i > 0
                #   right: path[j-1] if j > i (which is the element that was at j-1, now at i+1)
                # We need to ensure path[j] is not adjacent to any other vertex in the path except its two neighbors.
                # Similarly for path[i] at index j.
                
                # For simplicity and speed, do a full check only if boundary checks pass.
                # But full check is O(n^2) which is too slow for many iterations.
                # Let's do a targeted check: 
                # path[j] (new position i) must not be adjacent to any path[k] where k is not i-1, i+1.
                # path[i] (new position j) must not be adjacent to any path[k] where k is not j-1, j+1.
                
                # Precompute neighbors in the new path for the two moved elements
                # This is still O(n) per check, and we have O(n^2) checks, so O(n^3) total per improvement pass.
                # For n~100, that's 10^6, which might be okay in C but slow in Python.
                # Let's try a faster heuristic: only check if the new endpoints are "close" in the path.
                
                # Actually, let's just do the full validation for correctness first, then optimize.
                if _is_valid_snake(new_path):
                    path = new_path
                    improved = True
                    break
            if improved:
                break
    return path


def _extend_from_both_ends(path, deadline):
    """Try to extend the snake from both ends."""
    path = list(path)
    improved = True
    while improved and _time_ok(deadline):
        improved = False
        # Try extending from the start
        for b in range(N):
            v = path[0] ^ (1 << b)
            if v in path:
                continue
            # Check if v is adjacent to any non-first vertex
            if any((v ^ p).bit_count() == 1 for p in path[2:]):
                continue
            path.insert(0, v)
            improved = True
            break
        if improved:
            continue
        # Try extending from the end
        for b in range(N):
            v = path[-1] ^ (1 << b)
            if v in path:
                continue
            if any((v ^ p).bit_count() == 1 for p in path[:-2]):
                continue
            path.append(v)
            improved = True
            break
    return path


def construct():
    deadline = time.time() + 150.0 - DEADLINE_MARGIN
    
    # Start with greedy
    best = _greedy_start()
    best_len = len(best)
    
    # Local search
    improved = _local_search(best, deadline)
    if len(improved) > best_len:
        best = improved
        best_len = len(best)
    
    # Extend from both ends
    extended = _extend_from_both_ends(best, deadline)
    if len(extended) > best_len:
        best = extended
        best_len = len(extended)
    
    # More local search
    improved = _local_search(best, deadline)
    if len(improved) > best_len:
        best = improved
        best_len = len(best)
    
    return best
