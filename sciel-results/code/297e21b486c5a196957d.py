import time
import multiprocessing as mp

def _can_match_fast(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using fast matching."""
    # Build adjacency lists
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        first = ((m // i) + 1) * i
        end = m + L
        v = first
        while v <= end:
            adj[i - 1].append(v)
            v += i
        
    # Quick check
    for i in range(n):
        if not adj[i]:
            return False

    # Use array for match_val to speed up lookup
    # Max value is m + L. m < 27720, L < 144. So max val < 28000.
    # We can allocate a small array or use a dict. Dict is safer for arbitrary m, 
    # but m is bounded in our scan. 
    # Let's stick to dict for safety but optimize the DFS.
    
    match_val = {}
    
    # Iterative Kuhn's algorithm to avoid recursion limit/overhead
    # Actually, for n=12, recursion is fine. The overhead is the dict lookups.
    
    def dfs(i, visited):
        for v in adj[i]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_val:
                match_val[v] = i
                return True
            # If v is matched to j, try to reassign j
            j = match_val[v]
            if dfs(j, visited):
                match_val[v] = i
                return True
        return False

    for i in range(n):
        visited = set()
        if not dfs(i, visited):
            return False
    return True

def _f(m, n=12):
    """Compute f(n, m) using binary search."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _worker(args):
    m_start, m_end, n = args
    best_m = m_start
    best_f = 0
    for m in range(m_start, m_end):
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    return best_m, best_f

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 12
    period = 27720  # lcm(1, 2, ..., 12)
    
    # Use multiprocessing to speed up the scan
    # Split the range into chunks
    num_procs = mp.cpu_count()
    if num_procs > 8:
        num_procs = 8
        
    chunk_size = period // num_procs
    chunks = []
    for i in range(num_procs):
        start = i * chunk_size
        end = period if i == num_procs - 1 else (i + 1) * chunk_size
        chunks.append((start, end, n))
        
    best_m = 0
    best_f = 0
    
    # Check if we have enough time to use multiprocessing
    # Setup overhead is small, but let's be safe.
    if time.time() - start_time < 10:
        with mp.Pool(processes=num_procs) as pool:
            results = pool.map(_worker, chunks)
            
        for m_val, f_val in results:
            if f_val > best_f:
                best_f = f_val
                best_m = m_val
    else:
        # Fallback to sequential if something went wrong with time
        for m in range(period):
            if time.time() > deadline:
                break
            f_val = _f(m, n)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                
    return best_m
