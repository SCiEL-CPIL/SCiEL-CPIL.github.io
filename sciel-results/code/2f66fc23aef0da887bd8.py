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
    """
    Try to add v to a set represented by dp (for n elements).
    Returns new_dp if valid, None otherwise.
    """
    new_dp = [0] * (n + 2)
    new_dp[0] = 1
    # Update from high k to low k to avoid using the new v multiple times
    # Actually, we are constructing new_dp from dp.
    # new_dp[k] = dp[k] (subsets not using v) | (dp[k-1] << v) (subsets using v)
    # Note: dp has indices 0..n. new_dp has indices 0..n+1.
    
    # Copy old parts
    for k in range(n + 1):
        new_dp[k] = dp[k]
        
    # Add contributions from v
    for k in range(n + 1, 0, -1):
        # new_dp[k] already has dp[k] (if k <= n) or 0 (if k == n+1)
        # We OR in (dp[k-1] << v)
        new_dp[k] |= (dp[k - 1] << v)
        
    # Check validity
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
    
    # Candidate values: 1..256 not in cur
    # Sort by descending value? Larger values push sums higher, reducing collision probability with lower cardinality sums?
    # Or just iterate 1..256?
    # Let's try iterating 1..256 but skipping existing.
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        # Try adding each candidate
        # To speed up, we can pre-filter candidates that are obviously bad?
        # For now, just try all.
        for v in range(1, 257):
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

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0
    best = None
    best_len = 0
    
    def consider(arr):
        nonlocal best, best_len
        if len(arr) > best_len:
            # Verify before accepting as best to be safe, though _grow ensures validity
            # But random starts might be invalid if we didn't use _grow?
            # We will only feed valid sets to _grow.
            best = list(arr)
            best_len = len(arr)

    # 1. Start with powers of 2 (size 8, max sum 255)
    start_powers = [1, 2, 4, 8, 16, 32, 64, 128]
    g1 = _grow(start_powers, deadline)
    consider(g1)
    
    # 2. Start with high consecutive numbers?
    # Try a few structured starts
    structured_starts = [
        [200, 201, 202, 203, 204, 205, 206, 207],
        [100, 150, 200, 250], # sparse
        [10, 20, 40, 80, 160], # powers of 2 scaled
    ]
    
    for start in structured_starts:
        if time.monotonic() > deadline - 5:
            break
        # Verify start is valid first
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
            
    # 3. Random restarts
    rng = random.Random(42)
    attempts = 0
    while time.monotonic() < deadline - 2 and attempts < 5000:
        attempts += 1
        # Generate a random valid start of size 5-10
        # Strategy: Pick random numbers, check validity, if valid, grow.
        # To ensure start is valid, we can just pick small numbers or powers of 2 variants.
        # Or just pick random and check.
        
        n_start = rng.randint(5, 12)
        # Bias towards larger numbers to avoid sum collisions?
        # Or mix?
        pool = list(range(1, 257))
        start = rng.sample(pool, n_start)
        
        # Check if start is valid
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
            
    if best is None:
        best = start_powers
        
    return best
