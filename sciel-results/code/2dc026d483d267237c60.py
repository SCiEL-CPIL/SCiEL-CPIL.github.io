import random
import time
import sys

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns True/False and the subset if found.
    """
    if target == 0:
        return True, []
    if not weights:
        return False, None
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return False, None
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    # Generate all subset sums for left and right
    # To keep it fast, we only store sums <= target
    left_sums = {0: []}
    for w in left:
        new_sums = {}
        for s, path in left_sums.items():
            if s + w <= target:
                new_sums[s + w] = path + [w]
        for s, path in new_sums.items():
            if s not in left_sums or len(path) < len(left_sums.get(s, [])):
                left_sums[s] = path
                
    right_sums = {0: []}
    for w in right:
        new_sums = {}
        for s, path in right_sums.items():
            if s + w <= target:
                new_sums[s + w] = path + [w]
        for s, path in new_sums.items():
            if s not in right_sums or len(path) < len(right_sums.get(s, [])):
                right_sums[s] = path
        
    for s, l_path in left_sums.items():
        if (target - s) in right_sums:
            r_path = right_sums[target - s]
            return True, l_path + r_path
    return False, None

def find_conflict(S):
    """
    Find a violation in S.
    Returns (x, Y) such that 1/x = sum(1/y for y in Y) and Y subset of S\{x}.
    Returns None if S is valid.
    """
    S_set = set(S)
    for x in S:
        target = L // x
        weights = [L // y for y in S_set if y != x]
        # We need to map weights back to elements for the conflict set
        # Since weights are unique (L/y is unique for distinct y), we can track indices
        # But simpler: just find if sum exists.
        exists, subset_weights = subset_sum_exists(weights, target)
        if exists:
            # Reconstruct Y from subset_weights
            # subset_weights contains values L/y.
            # Map value to y: y = L // value
            Y = [L // w for w in subset_weights]
            return x, Y
    return None

def is_valid(S):
    """
    Check if S is valid.
    """
    return find_conflict(S) is None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_set = list(range(13, 25))
    best_size = len(best_set)
    
    # Strategy 1: Conflict Resolution from Full Set
    def resolve_conflicts():
        S = list(range(1, 25))
        S_set = set(S)
        # Track removals to avoid cycles
        removals = []
        
        while time.time() < deadline:
            conflict = find_conflict(S)
            if conflict is None:
                return S
            
            x, Y = conflict
            candidates = [x] + Y
            
            # Heuristic: Remove the smallest element
            # Small elements are targets of many sums
            min_elem = min(candidates)
            
            # Check if removing min_elem leads to a better state
            # Simple: just remove it
            S.remove(min_elem)
            S_set.remove(min_elem)
            removals.append(min_elem)
            
            # If we removed too many, or cycle detected (simple check)
            if len(removals) > 10:
                break
                
        return S

    S1 = resolve_conflicts()
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1

    # Strategy 2: Greedy Construction with Local Search
    def greedy_construct(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                S_set.add(x)
        return S

    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
        [12, 13, 11, 14, 10, 15, 9, 16, 8, 17, 7, 18, 6, 19, 5, 20, 4, 21, 3, 22, 2, 23, 1, 24],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Strategy 3: Local Search on Best Set
    def local_search(S, max_iterations=500):
        S = list(S)
        S_set = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each missing element
            missing = [i for i in range(1, 25) if i not in S_set]
            random.shuffle(missing)
            for m in missing:
                if time.time() > deadline:
                    return S
                new_S = S + [m]
                if is_valid(new_S):
                    S = new_S
                    S_set.add(m)
                    improved = True
                    break # Restart loop with new S
            
            # Try swapping
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    for j in range(i + 1, len(S)):
                        if time.time() > deadline:
                            return S
                        # Remove S[i], S[j]. Add two missing?
                        # Simpler: Remove one, add one.
                        S_temp = [x for x in S if x != S[i]]
                        S_temp_set = set(S_temp)
                        candidates = [k for k in range(1, 25) if k not in S_temp_set]
                        random.shuffle(candidates)
                        for c in candidates:
                            new_S = S_temp + [c]
                            if is_valid(new_S):
                                # Check if this swap is "better"
                                # For now, just accept if valid
                                # To improve, we could check if it allows adding more later
                                S = new_S
                                S_set = set(S)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return S

    if best_set is not None and time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # Strategy 4: Random Restarts
    random.seed(42)
    while time.time() < deadline and best_size < 24:
        random_order = list(range(1, 25))
        random.shuffle(random_order)
        S = greedy_construct(random_order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
        
        if time.time() < deadline:
            improved_set = local_search(S, max_iterations=100)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set

    if best_set is None:
        return [1]
    
    if not is_valid(best_set):
        return list(range(13, 25))
        
    return best_set
