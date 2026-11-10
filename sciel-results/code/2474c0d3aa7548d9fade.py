# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique_mask(adj, mask):
    if mask == 0:
        return True
    for v in range(14):
        if mask & (1 << v):
            if (adj[v] & mask) != (mask ^ (1 << v)):
                return False
    return True

def _is_independent_mask(adj, mask):
    if mask == 0:
        return True
    for v in range(14):
        if mask & (1 << v):
            if adj[v] & mask:
                return False
    return True

def _greedy_partition(adj):
    """Greedy heuristic to get an initial upper bound."""
    n = P['n']
    parts = []
    part_masks = []
    for v in range(n):
        vm = 1 << v
        best_idx = -1
        for i, pm in enumerate(part_masks):
            trial = pm | vm
            if _is_clique_mask(adj, trial) or _is_independent_mask(adj, trial):
                best_idx = i
                break
        if best_idx >= 0:
            part_masks[best_idx] |= vm
        else:
            part_masks.append(vm)
    result = []
    for pm in part_masks:
        lst = [v for v in range(n) if pm & (1 << v)]
        result.append(lst)
    return result

def _exact_dp(adj, deadline):
    """Exact O(3^N) DP to find minimum cochromatic number."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute valid part masks
    valid = [False] * (1 << n)
    valid[0] = True
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            return None
        valid[mask] = _is_clique_mask(adj, mask) or _is_independent_mask(adj, mask)
    
    # DP: dp[mask] = min number of valid parts to cover mask
    INF = n + 1
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            return None
        if valid[mask]:
            dp[mask] = 1
            continue
        # Try all submasks of mask
        sub = mask
        while sub > 0:
            if valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
            sub = (sub - 1) & mask
    
    if dp[full_mask] >= INF:
        return None
    
    # Reconstruct partition
    parts_masks = []
    mask = full_mask
    while mask > 0:
        sub = mask
        best_sub = None
        while sub > 0:
            if valid[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 == dp[mask]:
                    best_sub = sub
                    break
            sub = (sub - 1) & mask
        if best_sub is None:
            return None
        parts_masks.append(best_sub)
        mask ^= best_sub
    
    result = []
    for pm in parts_masks:
        lst = [v for v in range(n) if pm & (1 << v)]
        result.append(lst)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    adj = _build_graph()
    greedy_parts = _greedy_partition(adj)
    
    result = None
    try:
        result = _exact_dp(adj, deadline)
    except Exception:
        result = None
    
    if result is None:
        result = greedy_parts
    
    # Validation and cleanup
    if not isinstance(result, list):
        result = greedy_parts
    
    final_result = []
    seen = set()
    valid = True
    for part in result:
        if not isinstance(part, (list, tuple)):
            valid = False
            break
        for v in part:
            if not isinstance(v, int) or isinstance(v, bool):
                valid = False
                break
            if v in seen:
                valid = False
                break
            seen.add(v)
        if not valid:
            break
            
    if not valid or len(seen) != P['n']:
        result = greedy_parts
        final_result = []
        for part in result:
            final_result.append([int(v) for v in part])
    else:
        for part in result:
            final_result.append([int(v) for v in part])
            
    return final_result
# EVOLVE-BLOCK-END
