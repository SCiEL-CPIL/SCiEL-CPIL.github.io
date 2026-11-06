"""Snake-in-the-box in Q_9: constraint-based backtracking with MRV heuristic."""
import time
import random
import sys

N = 9
SIZE = 1 << N
MASK = SIZE - 1

# Precompute neighbors and forbidden masks for every vertex
NEIGHBOR_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASK[v] = m

def _candidates_at(cur, used_mask):
    """Return list of valid next vertices from cur given used_mask.
    A candidate v is valid if:
      - v not in used_mask
      - v not adjacent to any vertex in used_mask except cur
    """
    cands = []
    nbrs = NEIGHBOR_MASK[cur]
    for v_bit in range(N):
        v = cur ^ (1 << v_bit)
        if used_mask & (1 << v):
            continue
        # v must not be adjacent to any used vertex except cur
        # NEIGHBOR_MASK[v] & used_mask should contain only cur
        conflict = NEIGHBOR_MASK[v] & used_mask
        conflict &= ~(1 << cur)
        if conflict == 0:
            cands.append(v)
    return cands

def _mrv_score(cand, cur, used_mask):
    """MRV heuristic: score a candidate by how few options it leaves for the next step.
    Lower score = more constrained = preferred (fail-fast).
    We count how many valid extensions exist from cand given the new used_mask.
    """
    new_used = used_mask | (1 << cand)
    count = 0
    nbrs = NEIGHBOR_MASK[cand]
    for v_bit in range(N):
        v2 = cand ^ (1 << v_bit)
        if new_used & (1 << v2):
            continue
        conflict = NEIGHBOR_MASK[v2] & new_used
        conflict &= ~(1 << cand)
        if conflict == 0:
            count += 1
            if count > 12:  # cap for speed
                break
    return count

def _backtrack_search(path, used_mask, deadline, max_depth, rng, 
                      use_mrv=True, mrv_window=3):
    """Constraint-based DFS with MRV ordering.
    Returns (path, used_mask, extended) where extended indicates if path grew.
    """
    if time.time() > deadline:
        return path, used_mask, False
    
    cur = path[-1]
    cands = _candidates_at(cur, used_mask)
    
    if not cands:
        return path, used_mask, False
    
    # Order candidates
    if use_mrv and len(cands) > 1:
        # Score all candidates with MRV (how constrained the next step is)
        scored = []
        for c in cands:
            s = _mrv_score(c, cur, used_mask)
            scored.append((s, c))
        scored.sort(key=lambda x: x[0])  # most constrained first
        cands = [c for _, c in scored]
    else:
        rng.shuffle(cands)
    
    # Try candidates in order
    for i, v in enumerate(cands):
        if time.time() > deadline:
            return path, used_mask, False
        
        new_path = path + [v]
        new_used = used_mask | (1 << v)
        
        # Recurse with limited depth to avoid infinite loops
        if max_depth > 1:
            # Try to extend further
            extended_path, extended_used, did_extend = _backtrack_search(
                new_path, new_used, deadline, max_depth - 1, rng,
                use_mrv, mrv_window
            )
            if did_extend:
                return extended_path, extended_used, True
        
        # Even without deep recursion, accept this extension
        return new_path, new_used, True
    
    return path, used_mask, False

def _greedy_extend(path, used_mask, bit_order, deadline):
    """Pure greedy: try bits in given order, take first valid."""
    cur = path[-1]
    for b in bit_order:
        if time.time() > deadline:
            break
        v = cur ^ (1 << b)
        if used_mask & (1 << v):
            continue
        conflict = NEIGHBOR_MASK[v] & used_mask
        conflict &= ~(1 << cur)
        if conflict == 0:
            path.append(v)
            used_mask |= (1 << v)
            return True
    return False

def _greedy_construct(seed, bit_order, deadline):
    """Build a greedy snake from seed with given bit order."""
    path = [seed]
    used_mask = 1 << seed
    while time.time() < deadline:
        if not _greedy_extend(path, used_mask, bit_order, deadline):
            break
    return path, used_mask

def _try_backtrack_from(path, used_mask, deadline, rng, 
                        max_depth=15, max_attempts=5):
    """Try to extend path using backtracking with MRV.
    If greedy is stuck, use backtracking to find a longer extension.
    """
    if time.time() > deadline:
        return path, used_mask
    
    cur = path[-1]
    cands = _candidates_at(cur, used_mask)
    
    if not cands:
        return path, used_mask
    
    # Order by MRV
    scored = []
    for c in cands:
        s = _mrv_score(c, cur, used_mask)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0])
    cands = [c for _, c in scored]
    
    best_path = path[:]
    best_used = used_mask
    best_len = len(path)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Pick a candidate (rotating through ordered list for diversity)
        idx = (attempt * len(cands)) % len(cands)
        v = cands[idx]
        
        new_path = path + [v]
        new_used = used_mask | (1 << v)
        
        # Now try to extend from here with backtracking
        ext_path, ext_used, did_extend = _backtrack_search(
            new_path, new_used, deadline, max_depth, rng,
            use_mrv=True, mrv_window=3
        )
        
        if len(ext_path) > best_len:
            best_len = len(ext_path)
            best_path = ext_path[:]
            best_used = ext_used
    
    return best_path, best_used

def _local_search_extend(path, used_mask, deadline, rng):
    """Try to extend a stuck path by backtracking the last few steps
    and choosing differently (constraint-based local search).
    """
    if time.time() > deadline:
        return path, used_mask
    
    # Try truncating the last k vertices and re-extending with MRV
    for k in range(1, min(6, len(path))):
        if time.time() > deadline:
            break
        
        trunc_path = path[:-k]
        # Rebuild used_mask from trunc_path
        trunc_used = 0
        for v in trunc_path:
            trunc_used |= (1 << v)
        
        # Try to extend from truncated path with backtracking
        ext_path, ext_used = _try_backtrack_from(
            trunc_path, trunc_used, deadline, rng,
            max_depth=10, max_attempts=3
        )
        
        if len(ext_path) > len(path):
            return ext_path, ext_used
    
    return path, used_mask

def _validate(path):
    """Validate that path is a valid induced path. Returns edge count or -1."""
    n = len(path)
    if n == 0:
        return 0
    if len(set(path)) != n:
        return -1
    used_mask = 0
    for i, v in enumerate(path):
        if used_mask & (1 << v):
            return -1
        if i >= 2:
            conflict = NEIGHBOR_MASK[v] & used_mask
            conflict &= ~(1 << path[i-1])
            if conflict:
                return -1
        used_mask |= (1 << v)
    for i in range(n - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return -1
    return n - 1

def construct():
    """Construct the longest induced path (snake) in Q_9.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s (or 150s) limit
    rng = random.Random(20260715)
    
    best_path = None
    best_len = 0
    
    def update_best(path):
        nonlocal best_path, best_len
        L = _validate(path)
        if L > best_len:
            best_len = L
            best_path = path[:]
    
    # Phase 1: Quick greedy baselines to establish a starting point
    for trial in range(50):
        if time.time() > deadline:
            break
        seed = rng.randint(0, SIZE - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path, used_mask = _greedy_construct(seed, bit_order, deadline)
        update_best(path)
    
    # Phase 2: Constraint-based backtracking from greedy endpoints
    # This is the core structural change: use MRV to navigate the search tree
    for trial in range(200):
        if time.time() > deadline:
            break
        
        seed = rng.randint(0, SIZE - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path, used_mask = _greedy_construct(seed, bit_order, deadline)
        
        # Check if greedy is stuck
        cur = path[-1]
        cands = _candidates_at(cur, used_mask)
        
        if not cands:
            # Greedy stuck: use constraint-based backtracking to extend
            ext_path, ext_used = _try_backtrack_from(
                path, used_mask, deadline, rng,
                max_depth=20, max_attempts=8
            )
            update_best(ext_path)
            
            # If still stuck, try local search (truncate and re-extend)
            if len(ext_path) == len(path):
                ls_path, ls_used = _local_search_extend(
                    ext_path, ext_used, deadline, rng
                )
                update_best(ls_path)
        else:
            # Greedy found candidates but may not be optimal: use MRV to pick better
            ext_path, ext_used = _try_backtrack_from(
                path, used_mask, deadline, rng,
                max_depth=25, max_attempts=5
            )
            update_best(ext_path)
    
    # Phase 3: Deep constraint-based search from best path
    if best_path and time.time() < deadline - 10:
        # Try to extend the best path found so far
        best_used = 0
        for v in best_path:
            best_used |= (1 << v)
        
        ext_path, ext_used = _try_backtrack_from(
            best_path, best_used, deadline, rng,
            max_depth=30, max_attempts=10
        )
        update_best(ext_path)
        
        # Local search on best
        if time.time() < deadline - 5:
            ls_path, ls_used = _local_search_extend(
                best_path, best_used, deadline, rng
            )
            update_best(ls_path)
    
    # Phase 4: Continue with constraint-based search until deadline
    while time.time() < deadline - 2:
        seed = rng.randint(0, SIZE - 1)
        bit_order = list(range(N))
        rng.shuffle(bit_order)
        path, used_mask = _greedy_construct(seed, bit_order, deadline)
        
        # Always apply constraint-based extension
        ext_path, ext_used = _try_backtrack_from(
            path, used_mask, deadline, rng,
            max_depth=15, max_attempts=4
        )
        update_best(ext_path)
        
        # Occasionally do local search
        if rng.random() < 0.3:
            ls_path, ls_used = _local_search_extend(
                ext_path, ext_used, deadline, rng
            )
            update_best(ls_path)
    
    if best_path is None:
        return [0]
    return best_path
