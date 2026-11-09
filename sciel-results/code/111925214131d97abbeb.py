import time
import random
import numpy as np

def _find_conflict_fast(S, target=200):
    """
    Fast subset-sum check using numpy bitset.
    Returns True if any subset of S sums to target, False otherwise.
    """
    if not S:
        return target == 0
    
    # Use numpy bitset for fast subset sum
    # bit i is set if sum i is achievable
    bits = np.zeros(target + 1, dtype=np.uint8)
    bits[0] = 1
    
    for x in S:
        if x > target:
            continue
        # Shift and OR
        shifted = np.zeros(target + 1, dtype=np.uint8)
        shifted[x:] = bits[:target + 1 - x]
        bits = bits | shifted
    
    return bool(bits[target])

def _find_conflict_subset(S, target=200):
    """
    Find an actual subset that sums to target.
    Returns list or None.
    Uses DP with parent tracking.
    """
    if not S:
        return [0] if target == 0 else None
    
    n = len(S)
    # dp[s] = set of indices used to achieve sum s
    # For efficiency, store just the last element added
    parent = {}  # (sum, last_idx) -> previous sum
    reachable = {0: None}  # sum -> last index used
    
    for i, x in enumerate(S):
        if x > target:
            continue
        new_reachable = {}
        for s in list(reachable.keys()):
            ns = s + x
            if ns <= target and ns not in reachable:
                new_reachable[ns] = i
        for ns, idx in new_reachable.items():
            if ns not in reachable:
                reachable[ns] = idx
                parent[ns] = (ns - x, reachable.get(ns - x))
    
    if target not in reachable:
        return None
    
    # Reconstruct
    subset = []
    cur = target
    while cur is not None:
        if cur == 0:
            break
        idx = reachable[cur]
        subset.append(S[idx])
        prev_sum, _ = parent.get(cur, (cur - S[idx], None))
        cur = prev_sum if prev_sum != cur else None
        if cur is None or cur == 0:
            break
    
    return subset if subset else None

def _build_conflict_graph(S, target=200, sample_size=50):
    """
    Build a conflict graph where edge (u,v) means u and v frequently
    appear together in sum-target subsets.
    Returns adjacency list and degree dict.
    """
    n = len(S)
    S_list = list(S)
    idx_map = {x: i for i, x in enumerate(S_list)}
    
    # Count co-occurrences in conflicts
    co_count = [[0] * n for _ in range(n)]
    
    # Find multiple conflicts
    found = 0
    attempts = 0
    while found < sample_size and attempts < 200:
        attempts += 1
        # Use random subset to find diverse conflicts
        if len(S_list) > 30:
            sample = random.sample(S_list, min(len(S_list), 30))
        else:
            sample = S_list[:]
        
        # Quick check if conflict exists
        if not _find_conflict_fast(sample, target):
            continue
        
        # Find the conflict
        conflict = _find_conflict_subset(sample, target)
        if conflict is None or len(conflict) == 0:
            continue
        
        # Count co-occurrences
        for i in range(len(conflict)):
            for j in range(i + 1, len(conflict)):
                a = conflict[i]
                b = conflict[j]
                if a in idx_map and b in idx_map:
                    ia = idx_map[a]
                    ib = idx_map[b]
                    co_count[ia][ib] += 1
                    co_count[ib][ia] += 1
        found += 1
    
    # Build adjacency: connect if co_count > threshold
    threshold = max(1, found // 5)
    adj = {i: set() for i in range(n)}
    degree = {i: 0 for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            if co_count[i][j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)
                degree[i] += 1
                degree[j] += 1
    
    return adj, degree, S_list, idx_map

def _greedy_independent_set(adj, degree, S_list, idx_map, target=200):
    """
    Greedy minimum-degree independent set selection.
    Repeatedly pick the vertex with minimum degree, add it,
    and remove it and its neighbors.
    """
    n = len(S_list)
    remaining = set(range(n))
    selected = []
    
    # Copy degree for dynamic updates
    deg = dict(degree)
    adj_copy = {i: set(adj[i]) for i in adj}
    
    while remaining:
        # Find vertex with minimum degree among remaining
        min_deg = float('inf')
        min_vertex = None
        for v in remaining:
            # Count only neighbors that are still remaining
            active_deg = len(adj_copy[v] & remaining)
            if active_deg < min_deg:
                min_deg = active_deg
                min_vertex = v
        
        if min_vertex is None:
            break
        
        # Check if adding this vertex creates a conflict
        # We need to verify that the selected set plus this vertex has no sum-200 subset
        candidate = [S_list[v] for v in selected] + [S_list[min_vertex]]
        if _find_conflict_fast(candidate, target):
            # Can't add, remove from remaining
            remaining.discard(min_vertex)
            continue
        
        selected.append(min_vertex)
        remaining.discard(min_vertex)
        
        # Remove neighbors
        for neighbor in list(adj_copy[min_vertex]):
            if neighbor in remaining:
                remaining.discard(neighbor)
    
    return [S_list[v] for v in selected]

def _greedy_addition_improved(S, target=200):
    """
    Improved greedy addition: try to add elements in order of
    "least conflict potential" - elements that are less likely
    to form sum-200 subsets.
    """
    # Sort by value - smaller values have more combinations
    # but also: elements near target/2 are most dangerous
    # Let's try multiple orderings
    
    best = []
    
    orderings = [
        sorted(S),           # ascending
        sorted(S, reverse=True),  # descending
        sorted(S, key=lambda x: abs(x - target // 2)),  # by distance from target/2
        sorted(S, key=lambda x: x % 7),  # modular
    ]
    
    for ordering in orderings:
        selected = []
        for x in ordering:
            test = selected + [x]
            if not _find_conflict_fast(test, target):
                selected.append(x)
        
        if len(selected) > len(best):
            best = selected[:]
    
    return best

def _local_search_swap(S, target=200, max_iters=1000):
    """
    Local search: try swapping elements to increase set size.
    Start from a valid set, try to add one element by removing another.
    """
    current = list(S)
    if _find_conflict_fast(current, target):
        return current
    
    best = current[:]
    best_size = len(current)
    
    for _ in range(max_iters):
        improved = False
        
        # Try to add each element not in current
        for x in range(1, 101):
            if x in current:
                continue
            
            # Try adding x directly
            test = current + [x]
            if not _find_conflict_fast(test, target):
                current = test
                if len(current) > best_size:
                    best_size = len(current)
                    best = current[:]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove y, add x
        found_swap = False
        for y in current[:]:
            for x in range(1, 101):
                if x in current or x == y:
                    continue
                test = [v for v in current if v != y] + [x]
                if not _find_conflict_fast(test, target):
                    if len(test) >= len(current):  # only swap if size doesn't decrease
                        current = test
                        if len(current) > best_size:
                            best_size = len(current)
                            best = current[:]
                        found_swap = True
                        break
            if found_swap:
                break
        
        if not found_swap:
            break
    
    return best

def _greedy_low_degree():
    """
    Greedy approach using conflict graph and minimum-degree selection.
    """
    S = list(range(1, 101))
    
    # Build conflict graph
    adj, degree, S_list, idx_map = _build_conflict_graph(S, target=200, sample_size=30)
    
    # Use greedy independent set
    result = _greedy_independent_set(adj, degree, S_list, idx_map)
    
    # Verify and repair if needed
    while _find_conflict_fast(result, 200):
        # Find a conflict and remove one element
        conflict = _find_conflict_subset(result, 200)
        if conflict is None:
            break
        # Remove the element that appears in most conflicts
        remove_candidates = conflict
        elem_to_remove = max(remove_candidates)  # Remove largest to preserve small values
        result.remove(elem_to_remove)
    
    return result

def _greedy_greedy_addition():
    """
    Start with empty set, add elements one by one if they don't create a conflict.
    Try multiple orderings.
    """
    return _greedy_addition_improved(list(range(1, 101)))

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    best_set = None
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if S and len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Strategy 1: Deterministic descending greedy (fast baseline)
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    update_best(selected)
    
    # Strategy 2: Graph-based greedy independent set
    if time.time() - start_time < deadline - 15:
        result = _greedy_low_degree()
        if result:
            update_best(result)
    
    # Strategy 3: Improved greedy addition with multiple orderings
    if time.time() - start_time < deadline - 10:
        result = _greedy_greedy_addition()
        if result:
            update_best(result)
    
    # Strategy 4: Local search with swaps from best found
    if best_set and time.time() - start_time < deadline - 5:
        result = _local_search_swap(best_set, max_iters=500)
        if result:
            update_best(result)
    
    # Strategy 5: Randomized restarts with local search
    while time.time() - start_time < deadline - 2:
        # Start from a random valid subset
        S = list(range(1, 101))
        random.shuffle(S)
        
        # Greedily build a valid subset
        selected = []
        for x in S:
            test = selected + [x]
            if not _find_conflict_fast(test, 200):
                selected.append(x)
        
        if not selected:
            continue
        
        # Improve with local search
        result = _local_search_swap(selected, max_iters=200)
        if result:
            update_best(result)
        
        if time.time() - start_time > deadline - 2.0:
            break
    
    if best_set is None:
        # Fallback: return a simple valid set
        return [100]
    
    return best_set
