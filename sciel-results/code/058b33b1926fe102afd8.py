import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s to be safe against 90s limit
    
    n = 12
    max_m = 1000000
    period = 27720  # LCM(1, 2, ..., 12)
    
    best_L = 0
    best_m = 0
    
    # CRT-based approach: try offset vectors r_i in [0, i-1]
    # We want to maximize the span max(r_i) - min(r_i)
    # Strategy: try extreme offset vectors first (high span), then fill in
    
    # Generate candidate offset vectors by trying combinations of extremes
    # For each i, r_i can be 0 or i-1 (extremes) or intermediate
    # Start with all-extreme combinations, then relax
    
    def crt_solve(residues, moduli):
        """Solve system x ≡ r_i (mod m_i) using CRT. Returns x mod lcm or None."""
        from math import gcd
        x = 0
        M = 1
        for r, m in zip(residues, moduli):
            # Solve x ≡ r (mod m) given current x ≡ x (mod M)
            # x + k*M ≡ r (mod m) => k*M ≡ r - x (mod m)
            g = gcd(M, m)
            if (r - x) % g != 0:
                return None
            # k ≡ (r-x)/g * M^{-1} (mod m/g)
            M_g = M // g
            m_g = m // g
            # Find inverse of M_g mod m_g
            try:
                inv = pow(M_g, -1, m_g)
            except ValueError:
                return None
            k = ((r - x) // g * inv) % m_g
            x = x + k * M
            M = M * m_g
            x %= M
        return x, M
    
    moduli = list(range(1, n + 1))
    
    # Phase 1: Try extreme offset vectors (r_i = 0 or i-1)
    # There are 2^12 = 4096 such vectors, very fast to enumerate
    # For each, compute the span and use CRT to get m, then verify f(12,m)
    
    def compute_f(m, max_L_check=144):
        """Compute f(12, m) = least positive L such that (m, m+L] has distinct b_i with i|b_i."""
        for L in range(n, max_L_check + 1):
            # Build candidate lists
            candidates = [[] for _ in range(n + 1)]
            for i in range(1, n + 1):
                k = m // i + 1
                x = k * i
                while x <= m + L:
                    candidates[i].append(x)
                    x += i
            
            if any(len(candidates[i]) == 0 for i in range(1, n + 1)):
                continue
            
            # Hall's condition quick check: for every subset S of {1..n},
            # |union of candidates[i] for i in S| >= |S|
            # We check this using bitmask union
            all_nums_set = set()
            for i in range(1, n + 1):
                for x in candidates[i]:
                    all_nums_set.add(x)
            
            if len(all_nums_set) < n:
                continue
            
            # Map each number to an index
            nums = sorted(all_nums_set)
            num_to_idx = {x: i for i, x in enumerate(nums)}
            
            # For each i, precompute bitmask of available numbers
            i_masks = [0] * (n + 1)
            for i in range(1, n + 1):
                mask = 0
                for x in candidates[i]:
                    mask |= (1 << num_to_idx[x])
                i_masks[i] = mask
            
            # Hall's condition check using bitmask unions
            # For efficiency, check subsets by size
            # If any subset of size k has union < k, no matching exists
            # We can use a greedy-like check: sort by mask popcount and verify
            # Quick necessary condition: for each i, popcount(i_masks[i]) >= 1
            # Stronger: for any two i,j, popcount(i_masks[i] | i_masks[j]) >= 2
            # etc. Let's do a partial Hall check for small subsets
            
            # Check all pairs
            for i in range(1, n + 1):
                for j in range(i + 1, n + 1):
                    union_mask = i_masks[i] | i_masks[j]
                    if bin(union_mask).count('1') < 2:
                        break
                else:
                    continue
                break
            else:
                # Check triples
                for i in range(1, n + 1):
                    for j in range(i + 1, n + 1):
                        for k in range(j + 1, n + 1):
                            union_mask = i_masks[i] | i_masks[j] | i_masks[k]
                            if bin(union_mask).count('1') < 3:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    # Passed small-subset Hall checks, do full matching
                    # Order by fewest candidates first
                    order = sorted(range(1, n + 1), key=lambda i: bin(i_masks[i]).count('1'))
                    
                    # DFS with bitmask
                    def dfs(pos, used_mask):
                        if pos == n:
                            return True
                        i = order[pos]
                        available = i_masks[i] & ~used_mask
                        while available:
                            bit = available & (-available)
                            available ^= bit
                            if dfs(pos + 1, used_mask | bit):
                                return True
                        return False
                    
                    if dfs(0, 0):
                        return L
            
            # If Hall's small-subset check failed, skip this L
            continue
        
        return None  # No valid L found up to max_L_check
    
    # Generate all 2^12 extreme offset vectors
    # Sort by span descending to find good candidates first
    extreme_vectors = []
    for mask in range(1 << n):
        r = []
        for i in range(1, n + 1):
            if (mask >> (i-1)) & 1:
                r.append(i - 1)  # max
            else:
                r.append(0)      # min
        span = max(r) - min(r)
        extreme_vectors.append((span, r))
    
    # Sort by span descending
    extreme_vectors.sort(key=lambda x: -x[0])
    
    checked_m = set()
    
    for span, r_vec in extreme_vectors:
        if time.time() > deadline:
            break
        
        if best_L == 144:
            break
        
        # Compute m using CRT: m ≡ -r_i (mod i)
        residues = [(-r_vec[i-1]) % i for i in range(1, n + 1)]
        result = crt_solve(residues, moduli)
        if result is None:
            continue
        
        m_base, M = result
        # m_base is in [0, period)
        # Check m_base and m_base + period (if within range)
        for m_candidate in [m_base, m_base + period]:
            if m_candidate > max_m:
                continue
            if m_candidate in checked_m:
                continue
            checked_m.add(m_candidate)
            
            # Verify by computing f(12, m_candidate)
            L_val = compute_f(m_candidate)
            if L_val is not None and L_val > best_L:
                best_L = L_val
                best_m = m_candidate
    
    # Phase 2: If we have time, do a local search around best_m
    # Try m values near best_m that might give better results
    if best_L < 144 and time.time() < deadline - 5:
        # Try a range of m values around best_m
        # Also try m values that are near multiples of period with offsets
        # that maximize certain r_i
        
        # Try offsets around the best residue
        best_r = best_m % period
        # Try nearby residues
        for dr in range(-100, 101):
            if time.time() > deadline:
                break
            r_try = (best_r + dr) % period
            m_try = r_try
            if m_try > max_m:
                continue
            if m_try in checked_m:
                continue
            checked_m.add(m_try)
            
            L_val = compute_f(m_try)
            if L_val is not None and L_val > best_L:
                best_L = L_val
                best_m = m_try
    
    # Phase 3: If still time, try more systematic search
    if best_L < 144 and time.time() < deadline - 10:
        # Try intermediate offset vectors
        # For each i, try r_i = 0, i//2, i-1
        # This gives 3^12 ≈ 531441 vectors, too many
        # Instead, do a hill-climbing from best vector
        
        # Try m values: for each i, try shifting m by ±1, ±2, etc.
        # to see if we can improve
        for delta in range(-50, 51):
            if time.time() > deadline:
                break
            m_try = best_m + delta
            if m_try < 0 or m_try > max_m:
                continue
            if m_try in checked_m:
                continue
            checked_m.add(m_try)
            
            L_val = compute_f(m_try)
            if L_val is not None and L_val > best_L:
                best_L = L_val
                best_m = m_try
    
    return int(best_m)
