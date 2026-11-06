import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbor masks for fast adjacency checks
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

# Precompute bit positions for each vertex
BIT_POS = [0] * SIZE
for v in range(SIZE):
    bits = 0
    for b in range(N):
        if v & (1 << b):
            bits |= 1 << b
    BIT_POS[v] = bits

def construct():
    deadline = time.time() + 115.0
    rnd = random.Random(42)
    best_path = None
    best_len = 0

    def validate_and_score(path):
        """Check validity and return edge count (len-1) or -1 if invalid."""
        if not path:
            return 0
        n = len(path)
        seen = set()
        for i, v in enumerate(path):
            if v in seen:
                return -1
            seen.add(v)
            if i >= 1:
                for j in range(i - 1):
                    if (v ^ path[j]).bit_count() == 1:
                        return -1
                if (v ^ path[i-1]).bit_count() != 1:
                    return -1
        return n - 1

    def greedy_extend(path, priority):
        """Extend a partial path greedily using given bit-priority order."""
        used_mask = 0
        for v in path:
            used_mask |= (1 << v)
        cur = path[-1]
        while True:
            if time.time() > deadline - 2:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return path

    def random_greedy():
        """Generate a random greedy snake."""
        start = rnd.randint(0, SIZE - 1)
        priority = list(range(N))
        rnd.shuffle(priority)
        path = [start]
        used_mask = 1 << start
        cur = start
        while True:
            if time.time() > deadline - 2:
                break
            nxt = None
            for b in priority:
                v = cur ^ (1 << b)
                if used_mask & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_mask
                forbidden &= ~(1 << cur)
                if forbidden:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path.append(nxt)
            used_mask |= (1 << nxt)
            cur = nxt
        return path

    def count_available_neighbors(v, used_mask, exclude_cur=None):
        """Count how many unvisited neighbors v has (excluding cur if given)."""
        count = 0
        for b in range(N):
            nb = v ^ (1 << b)
            if nb == exclude_cur:
                continue
            if not (used_mask & (1 << nb)):
                count += 1
        return count

    def backtrack_search(start, time_budget):
        """
        Constraint-based backtracking with heuristic variable ordering.
        At each step, choose the next vertex from candidates that minimizes
        the future branching factor (minimum remaining values heuristic).
        """
        end_time = time.time() + time_budget
        path = [start]
        used_mask = 1 << start
        cur = start

        # For the backtracking search, we use a stack-based approach
        # where at each level we try candidates in order of "tightness"
        
        def _dfs(cur_v, used_m, path_list, depth):
            nonlocal best_path, best_len
            if time.time() > end_time:
                return False
            
            # Update best if current path is longer
            if len(path_list) > best_len:
                best_len = len(path_list) - 1
                best_path = list(path_list)
            
            # Get all valid candidates from current vertex
            candidates = []
            for b in range(N):
                v = cur_v ^ (1 << b)
                if used_m & (1 << v):
                    continue
                forbidden = NEIGHBOR_MASK[v] & used_m
                forbidden &= ~(1 << cur_v)
                if forbidden:
                    continue
                candidates.append(v)
            
            if not candidates:
                return False
            
            # Heuristic: order candidates by minimum remaining values
            # (i.e., prefer candidates that leave fewer options for future steps)
            scored_candidates = []
            for v in candidates:
                # Count how many unvisited neighbors v has (excluding cur_v)
                # Lower count means more constrained, so try first
                avail = count_available_neighbors(v, used_m | (1 << v), cur_v)
                scored_candidates.append((avail, rnd.random(), v))
            
            scored_candidates.sort(key=lambda x: (x[0], x[1]))
            
            for avail, _, v in scored_candidates:
                if time.time() > end_time:
                    return False
                
                new_used = used_m | (1 << v)
                path_list.append(v)
                
                if _dfs(v, new_used, path_list, depth + 1):
                    # Found a complete path or improved best
                    pass
                
                path_list.pop()
                
                # If we found a very good path, we can stop
                if best_len >= len(path_list) + 50:
                    return True
            
            return False
        
        _dfs(start, used_mask, path, 0)
        return best_path, best_len

    def local_search_improve(path):
        """
        Try to improve a path by local search:
        - Try removing a vertex and re-extending greedily
        """
        if not path:
            return path
        L = len(path)
        improved = True
        best_local = list(path)
        best_local_len = L
        
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing a vertex and re-extending
            for i in range(1, len(best_local) - 1):
                if time.time() > deadline - 5:
                    break
                candidate = best_local[:i] + best_local[i+1:]
                # Quick validity check
                valid = True
                for k in range(1, len(candidate)):
                    if (candidate[k] ^ candidate[k-1]).bit_count() != 1:
                        valid = False
                        break
                    for j in range(k - 1):
                        if (candidate[k] ^ candidate[j]).bit_count() == 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    extended = greedy_extend(list(candidate), list(range(N)))
                    if len(extended) > best_local_len:
                        best_local = extended
                        best_local_len = len(extended)
                        improved = True
                        break
        
        return best_local

    # Phase 0: Initial greedy with standard priority
    priority = list(range(N))
    path = greedy_extend([0], priority)
    L = validate_and_score(path)
    if L > 0:
        best_len = L
        best_path = path

    # Phase 1: Many random greedy restarts
    attempts = 0
    while time.time() < deadline - 30:
        p = random_greedy()
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p
        attempts += 1
        if attempts % 200 == 0 and time.time() > deadline - 35:
            break

    # Phase 2: Constraint-based backtracking with heuristic ordering
    # Try from multiple starting points with limited time budgets
    backtracking_starts = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19]
    for start in backtracking_starts:
        if time.time() > deadline - 15:
            break
        # Give each backtracking attempt a small time budget
        budget = min(8.0, max(2.0, deadline - time.time() - 15))
        old_best = best_len
        bt_path, bt_len = backtrack_search(start, budget)
        if bt_path and bt_len > best_len:
            best_len = bt_len
            best_path = bt_path
            # Try local improvement on the backtracking result
            p2 = local_search_improve(best_path)
            L2 = validate_and_score(p2)
            if L2 > best_len:
                best_len = L2
                best_path = p2

    # Phase 3: More random greedy with local search
    while time.time() < deadline - 8:
        p = random_greedy()
        L = validate_and_score(p)
        if L > best_len:
            best_len = L
            best_path = p
            # Try local improvement
            p2 = local_search_improve(best_path)
            L2 = validate_and_score(p2)
            if L2 > best_len:
                best_len = L2
                best_path = p2

    if best_path is None:
        return [0]
    return best_path
