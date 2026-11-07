import time
import random

def _build_dp(arr):
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _check_dp(dp):
    acc = 0
    for k in range(len(dp)):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _try_add(dp, v, n):
    new_dp = [0] * (n + 2)
    new_dp[0] = 1
    for k in range(n + 1):
        new_dp[k] = dp[k]
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << v)
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
    return new_dp

def _grow(start, deadline):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _build_dp(cur)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for v in range(256, 0, -1):
            if v in cur_set:
                continue
            new_dp = _try_add(dp, v, n)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                improved = True
                break
    return cur

def _reverse_construct(target_n, deadline):
    """
    Reverse construction: start with target_n slots and assign largest values first.
    We maintain bounds: for each k, min_sum[k] and max_sum[k].
    Constraint: min_sum[k] > max_sum[k-1] for all k >= 1.
    """
    if time.monotonic() > deadline:
        return None
    
    # We'll build the set in reverse: assign values from 256 down to 1
    # Start with all slots empty, assign largest available values
    assigned = [0] * target_n  # Will hold the final values
    assigned_set = set()
    
    # We process positions from last to first (largest to smallest)
    # For position i (0-indexed from smallest), we assign a value
    # But we want largest values first, so we assign from position target_n-1 down to 0
    
    # Track current min and max sums for each cardinality
    # Initially all empty
    min_sums = [0] * (target_n + 1)
    max_sums = [0] * (target_n + 1)
    
    # We'll use a backtracking approach
    # But for speed, let's try a greedy reverse assignment
    
    # Start with the largest possible values
    available = list(range(256, 0, -1))
    
    # Assign values from largest to smallest positions
    for pos in range(target_n - 1, -1, -1):
        if time.monotonic() > deadline:
            return None
        
        found = False
        for val in available:
            # Try assigning val to position pos
            # Check if this maintains the constraint
            # New min_sums and max_sums would be updated
            
            # For simplicity, let's just try and verify at the end
            assigned[pos] = val
            assigned_set.add(val)
            available.remove(val)
            found = True
            break
        
        if not found:
            return None
    
    # Verify the full set
    if len(assigned_set) != target_n:
        return None
    
    arr = sorted(assigned)
    if len(set(arr)) != target_n:
        return None
    
    dp = _build_dp(arr)
    if _check_dp(dp):
        return arr
    
    return None

def _reverse_construct_smart(target_n, deadline):
    """
    Smarter reverse construction using constraint propagation.
    We assign values from largest to smallest, maintaining sum bounds.
    """
    if time.monotonic() > deadline:
        return None
    
    n = target_n
    assigned = [0] * n
    assigned_set = set()
    available = set(range(1, 257))
    
    # min_sum[k] = minimum sum of any k-subset of assigned values
    # max_sum[k] = maximum sum of any k-subset of assigned values
    # We process positions from n-1 down to 0 (largest to smallest)
    
    # For efficiency, we'll maintain:
    # sorted_assigned: sorted list of currently assigned values
    # And compute min/max sums on the fly
    
    def compute_sums(vals):
        m = len(vals)
        min_s = [0] * (m + 1)
        max_s = [0] * (m + 1)
        for k in range(1, m + 1):
            min_s[k] = sum(sorted(vals)[:k])
            max_s[k] = sum(sorted(vals, reverse=True)[:k])
        return min_s, max_s
    
    def check_constraint(vals):
        m = len(vals)
        if m == 0:
            return True
        min_s, max_s = compute_sums(vals)
        for k in range(1, m + 1):
            if k - 1 >= 0 and min_s[k] <= max_s[k - 1]:
                return False
        return True
    
    # Greedy: assign largest available values first
    for pos in range(n - 1, -1, -1):
        if time.monotonic() > deadline:
            return None
        
        found = False
        # Try largest available values first
        for val in sorted(available, reverse=True):
            test_vals = list(assigned)
            test_vals[pos] = val
            # Remove zeros (unassigned positions)
            test_vals = [v for v in test_vals if v != 0]
            
            if check_constraint(test_vals):
                assigned[pos] = val
                assigned_set.add(val)
                available.discard(val)
                found = True
                break
        
        if not found:
            return None
    
    arr = sorted(assigned)
    dp = _build_dp(arr)
    if _check_dp(dp):
        return arr
    return None

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = None
    best_len = 0
    
    def consider(arr):
        nonlocal best, best_len
        if arr is None:
            return
        if len(arr) > best_len:
            if _check_dp(_build_dp(arr)):
                best = list(arr)
                best_len = len(arr)
    
    # 1. Forward greedy from powers of 2
    start_powers = [1, 2, 4, 8, 16, 32, 64, 128]
    g1 = _grow(start_powers, deadline)
    consider(g1)
    
    # 2. Try reverse construction for various target sizes
    # Start from high targets and work down
    for target_n in range(15, 5, -1):
        if time.monotonic() > deadline - 10:
            break
        if target_n <= best_len:
            continue
        result = _reverse_construct_smart(target_n, deadline)
        consider(result)
    
    # 3. Forward greedy from other starts
    structured_starts = [
        [128, 129, 130, 131, 132, 133, 134, 135],
        [10, 20, 40, 80, 160],
        [50, 100, 150, 200, 250],
        [1, 10, 50, 100, 200],
    ]
    
    for start in structured_starts:
        if time.monotonic() > deadline - 15:
            break
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
    
    # 4. Random restarts
    rng = random.Random(42)
    attempts = 0
    while time.monotonic() < deadline - 5 and attempts < 500:
        attempts += 1
        n_start = rng.randint(5, 10)
        pool = list(range(1, 257))
        start = rng.sample(pool, n_start)
        
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
    
    if best is None:
        best = start_powers
    
    return best
