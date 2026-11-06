"""Snake-in-the-box in Q_9: constructive greedy + destroy-reinsert local search."""
import time
import random
import sys

N = 9
MASK = (1 << N) - 1
TOTAL_VERTS = 1 << N  # 512
DEADLINE = 142.0  # safety margin before 150s kill

# Precompute neighbors
_NEIGHBORS = []
for v in range(TOTAL_VERTS):
    nbrs = [v ^ (1 << b) for b in range(N)]
    _NEIGHBORS.append(nbrs)

# Precompute adjacency bitmask: adj_mask[v] has bit w set iff v~w
_ADJ_MASK = []
for v in range(TOTAL_VERTS):
    m = 0
    for w in _NEIGHBORS[v]:
        m |= (1 << w)
    _ADJ_MASK.append(m)


def _is_valid_snake_full(path):
    """Full validation: consecutive differ by 1, non-consecutive differ by >= 2."""
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


def _build_used_set(path):
    s = set(path)
    return s


def _greedy_extend(path, used_set, rng, max_steps=None):
    """Greedy extend with randomized tie-breaking among valid candidates."""
    if max_steps is None:
        max_steps = 10000
    steps = 0
    while steps < max_steps:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # v must not be adjacent to any used vertex except cur
            ok = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if w in used_set:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        if not candidates:
            break
        v = rng.choice(candidates)
        path.append(v)
        used_set.add(v)
        steps += 1
    return path, used_set


def _greedy_prepend(path, used_set, rng, max_steps=None):
    """Greedy prepend with randomized tie-breaking."""
    if max_steps is None:
        max_steps = 10000
    steps = 0
    while steps < max_steps:
        cur = path[0]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            ok = True
            for w in _NEIGHBORS[v]:
                if w == cur:
                    continue
                if w in used_set:
                    ok = False
                    break
            if ok:
                candidates.append(v)
        if not candidates:
            break
        v = rng.choice(candidates)
        path.insert(0, v)
        used_set.add(v)
        steps += 1
    return path, used_set


def _try_insert_vertex(path, used_set, v):
    """Try to insert vertex v at any position in the path such that the result is a valid snake.
    Returns True and modifies path/used_set if successful."""
    n = len(path)
    # Try inserting at position i (between path[i-1] and path[i], with i=0 meaning prepend, i=n meaning append)
    for i in range(n + 1):
        prev = path[i - 1] if i > 0 else None
        nxt = path[i] if i < n else None
        # Check v is not adjacent to prev or nxt (they must differ by exactly 1)
        if prev is not None:
            if (v ^ prev).bit_count() != 1:
                continue
        if nxt is not None:
            if (v ^ nxt).bit_count() != 1:
                continue
        # Check v is not adjacent to any other vertex in path except prev and nxt
        ok = True
        for j, w in enumerate(path):
            if j == i - 1 or j == i:
                continue
            if (v ^ w).bit_count() == 1:
                ok = False
                break
        if not ok:
            continue
        # Insert
        path.insert(i, v)
        used_set.add(v)
        return True
    return False


def _find_insertable_vertices(path, used_set, all_vertices):
    """Find vertices not in used_set that can be inserted somewhere in the path."""
    insertable = []
    n = len(path)
    for v in all_vertices:
        if v in used_set:
            continue
        # Quick check: v must be adjacent to at least one path vertex
        # (since it would need to connect to prev and/or nxt)
        has_adj = False
        for w in path:
            if (v ^ w).bit_count() == 1:
                has_adj = True
                break
        if not has_adj:
            continue
        # Try to insert
        for i in range(n + 1):
            prev = path[i - 1] if i > 0 else None
            nxt = path[i] if i < n else None
            if prev is not None and (v ^ prev).bit_count() != 1:
                continue
            if nxt is not None and (v ^ nxt).bit_count() != 1:
                continue
            ok = True
            for j, w in enumerate(path):
                if j == i - 1 or j == i:
                    continue
                if (v ^ w).bit_count() == 1:
                    ok = False
                    break
            if ok:
                insertable.append((v, i))
                break
    return insertable


def _destroy_reinsert_search(path, used_set, rng, time_limit):
    """
    Local search: repeatedly remove a random vertex from the path, then try to
    re-insert it (or other vertices) to grow the path. Uses simulated annealing
    style: accept temporary length decreases if they enable later growth.
    """
    start_time = time.time()
    best_path = list(path)
    best_used = set(used_set)
    best_len = len(path)
    
    # Current state
    cur_path = list(path)
    cur_used = set(used_set)
    cur_len = len(path)
    
    # Temperature for simulated annealing
    temperature = 5.0
    cooling = 0.995
    
    all_vertices = list(range(TOTAL_VERTS))
    
    while time.time() - start_time < time_limit:
        if temperature < 0.1:
            break
        
        # Choose a random vertex to remove (not the endpoints, to keep path connected)
        if len(cur_path) < 4:
            break
        
        # Remove a random interior vertex
        idx = rng.randint(1, len(cur_path) - 2)
        removed_v = cur_path[idx]
        cur_path.pop(idx)
        cur_used.remove(removed_v)
        
        # Now try to re-insert removed_v or find other insertable vertices
        # First try to re-insert the removed vertex
        reinserted = _try_insert_vertex(cur_path, cur_used, removed_v)
        
        if reinserted:
            new_len = len(cur_path)
            # Accept if same length (re-inserted) or better
            if new_len >= cur_len:
                cur_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_path = list(cur_path)
                    best_used = set(cur_used)
            else:
                # Rejected, undo
                # Remove the re-inserted vertex
                cur_path.remove(removed_v)
                cur_used.remove(removed_v)
        else:
            # Try to find any insertable vertex to grow
            insertable = _find_insertable_vertices(cur_path, cur_used, all_vertices)
            if insertable:
                # Pick a random insertable vertex
                v, pos = rng.choice(insertable)
                cur_path.insert(pos, v)
                cur_used.add(v)
                new_len = len(cur_path)
                if new_len >= cur_len:
                    cur_len = new_len
                    if new_len > best_len:
                        best_len = new_len
                        best_path = list(cur_path)
                        best_used = set(cur_used)
                else:
                    # Reject, undo
                    cur_path.remove(v)
                    cur_used.remove(v)
            else:
                # No insertable vertex, try to re-insert removed_v with different approach
                # Just put it back
                # Find a position where it fits
                n = len(cur_path)
                inserted = False
                for i in range(n + 1):
                    prev = cur_path[i - 1] if i > 0 else None
                    nxt = cur_path[i] if i < n else None
                    if prev is not None and (removed_v ^ prev).bit_count() != 1:
                        continue
                    if nxt is not None and (removed_v ^ nxt).bit_count() != 1:
                        continue
                    ok = True
                    for j, w in enumerate(cur_path):
                        if j == i - 1 or j == i:
                            continue
                        if (removed_v ^ w).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        cur_path.insert(i, removed_v)
                        cur_used.add(removed_v)
                        inserted = True
                        break
                if not inserted:
                    # Can't re-insert, try a different removal
                    # This is a dead end, restart from best
                    cur_path = list(best_path)
                    cur_used = set(best_used)
                    cur_len = best_len
                    temperature *= cooling
                    continue
        
        temperature *= cooling
    
    return best_path, best_used


def _multi_start_constructive(rng, time_limit):
    """
    Multi-start constructive heuristic with destroy-reinsert local search.
    """
    start_time = time.time()
    best_path = []
    best_len = 0
    
    for seed in range(500):
        if time.time() - start_time > time_limit:
            break
        
        rng_local = random.Random(seed * 12345 + 67890)
        
        # Random start vertex
        start_v = rng_local.randint(0, TOTAL_VERTS - 1)
        
        # Phase 1: Constructive greedy (extend both ends)
        path = [start_v]
        used_set = {start_v}
        
        # Extend forward
        path, used_set = _greedy_extend(path, used_set, rng_local)
        # Extend backward
        path, used_set = _greedy_prepend(path, used_set, rng_local)
        # Extend forward again
        path, used_set = _greedy_extend(path, used_set, rng_local)
        
        cur_len = len(path)
        
        if cur_len > best_len:
            best_len = cur_len
            best_path = list(path)
        
        # Phase 2: Destroy-reinsert local search
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        # Allocate time for local search based on path quality
        search_time = min(remaining_time * 0.3, 8.0)
        if cur_len > best_len - 5:
            search_time = min(remaining_time * 0.5, 15.0)
        
        improved_path, improved_used = _destroy_reinsert_search(
            path, used_set, rng_local, search_time
        )
        
        if len(improved_path) > best_len:
            best_len = len(improved_path)
            best_path = list(improved_path)
    
    return best_path, best_len


def _dfs_extend(path, used_mask, target_len, deadline, rng, bit_order_cache):
    """
    Depth-first search trying to extend path to reach target_len.
    Returns True if a path of length >= target_len was found.
    """
    if len(path) >= target_len:
        return True
    
    if time.time() > deadline:
        return False
    
    cur = path[-1]
    
    if cur in bit_order_cache:
        bits = bit_order_cache[cur]
    else:
        bits = list(range(N))
        rng.shuffle(bits)
        bit_order_cache[cur] = bits
    
    for b in bits:
        v = cur ^ (1 << b)
        
        if (used_mask >> v) & 1:
            continue
        
        valid = True
        for w in _NEIGHBORS[v]:
            if w == cur:
                continue
            if (used_mask >> w) & 1:
                valid = False
                break
        
        if not valid:
            continue
        
        path.append(v)
        used_mask |= (1 << v)
        
        if _dfs_extend(path, used_mask, target_len, deadline, rng, bit_order_cache):
            return True
        
        path.pop()
        used_mask ^= (1 << v)
        
        if time.time() > deadline:
            return False
    
    return False


def _iterative_deepening_search(start_v, max_depth, deadline, rng):
    """Iterative deepening: try to find a snake of length max_depth."""
    path = [start_v]
    used_mask = 1 << start_v
    bit_order_cache = {}
    
    if _dfs_extend(path, used_mask, max_depth, deadline, rng, bit_order_cache):
        return path
    
    return None


def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multi-start constructive + destroy-reinsert local search
    # This is the main algorithm
    phase1_time = 90.0
    path1, len1 = _multi_start_constructive(rng, phase1_time)
    if len1 > best_len:
        best_len = len1
        best_path = path1
    
    # Phase 2: DFS iterative deepening to try to extend beyond current best
    if time.time() < DEADLINE - 10 and best_path:
        current_best_start = best_path[0]
        target = best_len + 1
        max_target = min(best_len + 30, TOTAL_VERTS)
        
        # Try a few starting vertices
        start_vertices = [current_best_start]
        for _ in range(5):
            start_vertices.append(rng.randint(0, TOTAL_VERTS - 1))
        
        for sv in start_vertices:
            if time.time() > DEADLINE - 5:
                break
            
            target = best_len + 1
            while target <= max_target and time.time() < DEADLINE - 5:
                path = _iterative_deepening_search(sv, target, DEADLINE - 5, rng)
                if path is not None:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                        target += 1
                    else:
                        break
                else:
                    break
    
    # Phase 3: Final local search on the best path found
    if time.time() < DEADLINE - 5 and best_path:
        used_set = set(best_path)
        remaining = DEADLINE - 5 - time.time()
        if remaining > 1.0:
            improved_path, improved_used = _destroy_reinsert_search(
                best_path, used_set, rng, remaining
            )
            if len(improved_path) > best_len:
                best_len = len(improved_path)
                best_path = list(improved_path)
    
    # Final validation
    if best_path and not _is_valid_snake_full(best_path):
        # Fallback: simple deterministic greedy
        path = [0]
        used_set = {0}
        path, used_set = _greedy_extend(path, used_set, random.Random(0))
        best_path = path
        best_len = len(path)
    
    # Ensure minimum length
    if len(best_path) < 104:
        path = [0]
        used_set = {0}
        path, used_set = _greedy_extend(path, used_set, random.Random(0))
        best_path = path
        best_len = len(path)
    
    return best_path


if __name__ == "__main__":
    result = construct()
    print(f"Snake length: {len(result)}")
