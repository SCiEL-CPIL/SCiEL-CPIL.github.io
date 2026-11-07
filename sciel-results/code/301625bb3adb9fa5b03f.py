"""Find m in [0, 1000000] maximizing f(12, m) - the least L such that (m, m+L] has distinct b_i with i|b_i."""
import time

N = 12
MAX_M = 1000000
MAX_L = 144  # f(12,m) is at most 144
PERIOD = 27720  # lcm(1..12)

# Precompute change points for each i and residue
# _change_cache[(m % i, i)] = list of L values where a new multiple of i enters (m, m+L]
_change_cache = {}

def _get_changes(m, i):
    """Get the list of L values (1..MAX_L) where a new multiple of i enters (m, m+L]."""
    key = (m % i, i)
    if key not in _change_cache:
        # First multiple of i greater than m (using m % i as representative)
        r = m % i
        k = (r // i) + 1  # This is 1 since r < i
        val = k * i  # = i
        # Actually, we need to be more careful.
        # For a given residue r = m % i, the first multiple of i > m is at distance:
        # if r == 0: distance is i (next multiple is m+i)
        # if r > 0: distance is i - r
        # Wait, let's re-derive.
        # m = q*i + r, where 0 <= r < i.
        # Multiples of i are ..., (q-1)*i, q*i, (q+1)*i, ...
        # m is between q*i and (q+1)*i (if r>0) or equals q*i (if r=0).
        # The interval is (m, m+L].
        # The first multiple of i strictly greater than m:
        # If r == 0: m is a multiple. Next is m+i. Distance = i.
        # If r > 0: m is not a multiple. Next is (q+1)*i. Distance = (q+1)*i - (q*i + r) = i - r.
        # So distance d = i if r==0 else i-r.
        # Subsequent multiples are at d, d+i, d+2i, ...
        
        if r == 0:
            d = i
        else:
            d = i - r
        
        changes = []
        Lc = d
        while Lc <= MAX_L:
            changes.append(Lc)
            Lc += i
        _change_cache[key] = changes
    return _change_cache[key]

def _can_match(m, L):
    """Return True if (m, m+L] contains 12 distinct b_i with i|b_i."""
    # Build adjacency using bitmasks for speed
    adj = [0] * N
    for i in range(1, N + 1):
        mask = 0
        for Lc in _get_changes(m, i):
            if Lc <= L:
                # The multiple is at value m+Lc. Its index in (m, m+L] is Lc-1.
                mask |= (1 << (Lc - 1))
            else:
                break
        if mask == 0:
            return False
        adj[i - 1] = mask
    
    # Kuhn's algorithm with bitmask iteration
    match_to_right = [-1] * L  # match_to_right[v] = u, where v is 0..L-1
    
    def dfs(u, seen):
        mask = adj[u]
        while mask:
            lsb = mask & (-mask)
            v = lsb.bit_length() - 1
            mask ^= lsb
            if seen & lsb:
                continue
            seen |= lsb
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen):
                match_to_right[v] = u
                return True
        return False
    
    # Process nodes with fewer options first for better pruning
    # Order by number of set bits in adj[u]
    order = sorted(range(N), key=lambda u: bin(adj[u]).count('1'))
    
    for u in order:
        if not dfs(u, 0):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m) = least positive L such that (m, m+L] has valid distinct b_i."""
    # Lower bound: max distance to next multiple
    lo = 1
    for i in range(1, N + 1):
        r = m % i
        if r == 0:
            d = i
        else:
            d = i - r
        if d > lo:
            lo = d
    
    hi = MAX_L + 1
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    best_f = 0
    best_residue = 0
    
    # Evaluate all residues 0..PERIOD-1
    for r in range(PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f(r)
        if f_val is not None and f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find the smallest m in [0, MAX_M] with m % PERIOD == best_residue
    best_m = best_residue
    if best_m > MAX_M:
        best_m = best_residue % (MAX_M + 1)
    
    # Local search around best_m for robustness
    if time.time() < deadline - 3:
        search_start = max(0, best_m - 10)
        search_end = min(MAX_M, best_m + 10)
        for mm in range(search_start, search_end + 1):
            if time.time() > deadline:
                break
            f_val = _compute_f(mm)
            if f_val is not None and f_val > best_f:
                best_f = f_val
                best_m = mm
    
    return int(best_m)
