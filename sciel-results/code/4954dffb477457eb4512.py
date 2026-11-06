import time
from collections import deque

N = 12
FULL_SET = (1 << N) - 1

# Precompute popcount for single-bit check
def _is_singleton(mask):
    return mask != 0 and (mask & (mask - 1)) == 0

def _bfs_reset_length_fast(a, b):
    """
    Fast BFS using precomputed transition tables for all subsets.
    """
    # Precompute transition tables for all 2^N subsets
    # trans_a[subset] = result of applying 'a' to all states in subset
    # trans_b[subset] = result of applying 'b' to all states in subset
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for subset in range(1, 1 << N):
        ta = 0
        tb = 0
        temp = subset
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            ta |= (1 << a[idx])
            tb |= (1 << b[idx])
            temp ^= lsb
        trans_a[subset] = ta
        trans_b[subset] = tb
    
    # BFS from full_set
    if _is_singleton(FULL_SET):
        return 0
    
    visited = [-1] * (1 << N)
    visited[FULL_SET] = 0
    queue = deque([FULL_SET])
    
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        
        if _is_singleton(curr):
            return dist
        
        # Apply 'a'
        next_a = trans_a[curr]
        if visited[next_a] == -1:
            if _is_singleton(next_a):
                return dist + 1
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = trans_b[curr]
        if visited[next_b] == -1:
            if _is_singleton(next_b):
                return dist + 1
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return -1  # Not synchronizing

def _local_search(a, b, timeout):
    """
    Hill-climbing local search on b transitions.
    Returns best (score, b_copy) found.
    """
    start_time = time.time()
    current_best = _bfs_reset_length_fast(a, b)
    best_b = b[:]
    
    if current_best < 0:
        return current_best, best_b
    
    while time.time() - start_time < timeout:
        improved = False
        for i in range(N):
            if time.time() - start_time > timeout:
                break
            old_val = b[i]
            for j in range(N):
                if j == old_val:
                    continue
                b[i] = j
                val = _bfs_reset_length_fast(a, b)
                if val > current_best:
                    current_best = val
                    best_b = b[:]
                    improved = True
                    break
            if not improved:
                b[i] = old_val
            if improved:
                break
                
        if not improved:
            break
            
    return current_best, best_b

def construct():
    """
    Systematically enumerate structural families and apply local search.
    """
    start_time = time.time()
    total_timeout = 85.0  # Safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Family 1: Pinwheel / cycle + break
    # a: cycle, b: mostly identity with one or two breaks
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i for i in range(N)]
        b[break_pos] = (break_pos + N - 1) % N  # break: send break_pos to previous
        score = _bfs_reset_length_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Family 2: Cycle + clamp
    # a: cycle, b: clamp to a range
    for lo in range(N):
        for hi in range(lo, N):
            a = [(i + 1) % N for i in range(N)]
            b = [min(max(i, lo), hi) for i in range(N)]
            score = _bfs_reset_length_fast(a, b)
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # Family 3: Vorel halving variants
    a = [(i + 1) % N for i in range(N)]
    b = [i // 2 for i in range(N)]
    score = _bfs_reset_length_fast(a, b)
    if score > best_score:
        best_score = score
        best_a = a[:]
        best_b = b[:]
    
    # Family 4: Rank-2 constructions
    # a: cycle, b: maps even to 0, odd to 1, others to themselves
    for mod in [2, 3, 4]:
        a = [(i + 1) % N for i in range(N)]
        b = []
        for i in range(N):
            if i % mod == 0:
                b.append(0)
            elif i % mod == 1:
                b.append(1)
            else:
                b.append(i)
        score = _bfs_reset_length_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Family 5: Systematic sweep of b for fixed a=cycle
    # Try many different b patterns
    a = [(i + 1) % N for i in range(N)]
    
    # b: all map to 0
    b = [0] * N
    score = _bfs_reset_length_fast(a, b)
    if score > best_score:
        best_score = score
        best_a = a[:]
        best_b = b[:]
    
    # b: identity except one state maps elsewhere
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            b = list(range(N))
            b[i] = j
            score = _bfs_reset_length_fast(a, b)
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # Family 6: Two-break pinwheel
    for i in range(N):
        for j in range(i+1, N):
            a = [(i + 1) % N for i in range(N)]
            b = [k for k in range(N)]
            b[i] = (i + N - 1) % N
            b[j] = (j + N - 1) % N
            score = _bfs_reset_length_fast(a, b)
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # Family 7: Vorel with offset
    for offset in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + offset) // 2 for i in range(N)]
        score = _bfs_reset_length_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Check time and do local search on best candidate
    elapsed = time.time() - start_time
    remaining = total_timeout - elapsed
    
    if remaining > 2.0 and best_score > 0:
        # Do local search on the best candidate found
        local_timeout = min(remaining - 1.0, 30.0)
        _, local_b = _local_search(best_a, best_b, local_timeout)
        local_score = _bfs_reset_length_fast(best_a, local_b)
        if local_score > best_score:
            best_score = local_score
            best_b = local_b
    
    # Final safety check
    if time.time() - start_time > total_timeout - 2.0:
        # Return best found so far
        pass
    
    if best_score < 0:
        # Fallback: return a known synchronizing automaton
        a = [(i + 1) % N for i in range(N)]
        b = [i // 2 for i in range(N)]
        return [a, b]
    
    return [best_a, best_b]
