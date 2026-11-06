"""Snake-in-the-box in Q_9: build the longest induced path (snake)."""
import time
import random

N = 9
TOTAL = 1 << N  # 512 vertices
# Precompute neighbors for each vertex
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(TOTAL)]

# Precompute bit_count for all pairs? That's 512*512 = 262144, fine.
# But we can compute on the fly with int.bit_count() which is fast in Python 3.10+.

def _is_valid_extension(path, used_set, candidate):
    """Check if candidate can be appended to path as an induced path extension.
    - candidate not in used_set
    - candidate differs from all non-consecutive vertices (i.e., all except path[-1]) by >= 2 bits
    - candidate differs from path[-1] by exactly 1 bit (guaranteed by construction if we only consider neighbors)
    """
    if candidate in used_set:
        return False
    # Check against all vertices in path except the last one (path[-1])
    # Non-consecutive means index != len(path)-1 and index != len(path)-2? 
    # Actually: consecutive vertices are (path[i], path[i+1]).
    # New vertex is path[-1] (current end). It is consecutive to path[-2].
    # So we must check that new vertex differs from path[0..-3] by >= 2 bits.
    # It already differs from path[-1] by 1 bit (by construction as a neighbor).
    # And it must differ from path[-2] by >= 2 bits? No, path[-2] is consecutive to path[-1], not to the new vertex.
    # Wait: the new vertex is adjacent to path[-1]. Is it adjacent to path[-2]?
    # If it were, then path[-2] and new vertex would be consecutive in the path? No, they are not consecutive in the sequence.
    # The sequence is ... path[-2], path[-1], new_vertex.
    # Consecutive pairs are (path[-2], path[-1]) and (path[-1], new_vertex).
    # Non-consecutive pairs involving new_vertex are (new_vertex, path[i]) for i <= len(path)-3.
    # So we need Hamming distance >= 2 for all i <= len(path)-3.
    # Also, new_vertex must not be adjacent to path[-2]? No, path[-2] is at distance 2 from new_vertex in the path sequence,
    # but they could be Hamming distance 1. If they are Hamming distance 1, that creates a chord, making it not induced.
    # So yes, we need Hamming distance >= 2 for ALL previous vertices except path[-1].
    # path[-1] is the current end. The new vertex is a neighbor of path[-1].
    # So we check all vertices in path except the last one.
    for v in path[:-1]:
        if (v ^ candidate).bit_count() == 1:
            return False
    return True


def _greedy_construct(start=0):
    """Greedy snake construction with Warnsdorff-like heuristic."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced property
            valid = True
            for p in path[:-1]:
                if (p ^ v).bit_count() == 1:
                    valid = False
                    break
            if valid:
                # Count how many further options this candidate would leave
                # (Warnsdorff: prefer moves with fewer onward options)
                onward = 0
                for b2 in range(N):
                    w = v ^ (1 << b2)
                    if w in used:
                        continue
                    # Quick check: w must not be adjacent to any vertex in path[:-1]
                    # But this is expensive. For now, just count unvisited neighbors.
                    onward += 1
                candidates.append((onward, b, v))
        if not candidates:
            return path
        # Sort by onward count (ascending) to prefer tightest moves
        candidates.sort(key=lambda x: x[0])
        _, _, nxt = candidates[0]
        path.append(nxt)
        used.add(nxt)
    return path


def _try_splice(path):
    """Attempt to improve the snake by local splicing/mutations.
    Try flipping a segment or inserting a vertex.
    This is a heuristic local search."""
    # Simple approach: try to insert a vertex between two consecutive vertices
    # where it's not already adjacent to other vertices
    improved = False
    n = len(path)
    if n < 3:
        return path, False
    
    used = set(path)
    
    # Try inserting a vertex at each position
    for i in range(1, n - 1):
        prev_v = path[i - 1]
        next_v = path[i]
        # Find common neighbors of prev_v and next_v that are not in used
        # and not adjacent to any other vertex in the path
        common_neighbors = []
        for b in range(N):
            cand = prev_v ^ (1 << b)
            if cand == next_v or cand in used:
                continue
            # Check if cand is adjacent to next_v
            if (cand ^ next_v).bit_count() != 1:
                continue
            # Check induced property with all other vertices
            valid = True
            for j, p in enumerate(path):
                if j == i - 1 or j == i:
                    continue
                if (p ^ cand).bit_count() == 1:
                    valid = False
                    break
            if valid:
                common_neighbors.append(cand)
        
        if common_neighbors:
            # Try inserting the first valid candidate
            cand = common_neighbors[0]
            new_path = path[:i] + [cand] + path[i:]
            # Verify the entire path is still valid
            if _verify_snake(new_path):
                return new_path, True
    
    return path, False


def _verify_snake(path):
    """Verify that path is a valid induced path in the hypercube."""
    n = len(path)
    if n == 0:
        return True
    if len(set(path)) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_path = []
    
    # Try greedy from multiple starting points
    random.seed(42)
    starts = list(range(TOTAL))
    random.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        
        path = _greedy_construct(start)
        
        if len(path) > len(best_path):
            best_path = path
        
        # Try local improvement via splicing
        if time.time() > deadline:
            break
        
        improved_path, improved = _try_splice(path)
        if improved and len(improved_path) > len(best_path):
            best_path = improved_path
    
    # Additional pass: try to extend the best path found
    if best_path and time.time() < deadline:
        # Try to extend from the end
        path = list(best_path)
        used = set(path)
        while time.time() < deadline:
            cur = path[-1]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[:-1]:
                    if (p ^ v).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.append(v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        
        if len(path) > len(best_path):
            best_path = path
    
    # Try to extend from the beginning
    if best_path and time.time() < deadline:
        path = list(best_path)
        used = set(path)
        while time.time() < deadline:
            cur = path[0]
            found = False
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                valid = True
                for p in path[1:]:
                    if (p ^ v).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    path.insert(0, v)
                    used.add(v)
                    found = True
                    break
            if not found:
                break
        
        if len(path) > len(best_path):
            best_path = path
    
    return best_path
