import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 12
    # LCM(1..12) = 27720
    LCM = 27720
    MAX_L = 144  # maximum possible f value
    
    # Precompute: for each residue r in 0..LCM-1, and each L in 1..MAX_L,
    # determine if matching is feasible.
    # f(N, m) = least L such that matching exists in (m, m+L]
    # Since f depends on m mod LCM, we compute f for r = 0..LCM-1.
    
    # For a given m and L, we need to check if there exist distinct b_1,...,b_12
    # in (m, m+L] such that i | b_i for each i=1..12.
    
    # b_i in (m, m+L] means b_i = m + j for some j in 1..L
    # i | b_i means (m + j) % i == 0, i.e., j % i == (-m) % i
    
    # For each i, the valid j values in 1..L are those with j ≡ (-m) mod i.
    
    # We'll use a bitmask approach: for each i, compute a bitmask of valid positions j (0-indexed: j-1).
    # Then check if we can assign distinct positions to all 12 indices.
    
    def check_feasible(m_mod, L):
        """Check if (m, m+L] has a valid matching, where m ≡ m_mod (mod LCM)."""
        # For each i in 1..12, find which positions j (1-indexed, 1..L) satisfy i | (m + j)
        # (m + j) % i == 0 => j % i == (-m) % i
        # m ≡ m_mod (mod LCM), and since i | LCM, m ≡ m_mod (mod i)
        # So (-m) % i = (-m_mod) % i
        
        options = []
        for i in range(1, N + 1):
            target = (-m_mod) % i
            # j in 1..L, j % i == target
            # j = target + k*i, for j in 1..L
            # If target == 0, j = i, 2i, 3i, ...
            # If target > 0, j = target, target+i, target+2i, ...
            
            mask = 0
            if target == 0:
                first = i
            else:
                first = target
            
            j = first
            while j <= L:
                mask |= (1 << (j - 1))  # 0-indexed position
                j += i
            
            if mask == 0:
                return False
            options.append((mask.bit_count(), mask))
        
        # Sort by number of options (most constrained first)
        options.sort(key=lambda x: x[0])
        
        # DFS with bitmask
        def dfs(k, used_mask):
            if k == N:
                return True
            _, avail = options[k]
            # Try each available position not yet used
            available = avail & ~used_mask
            while available:
                lowbit = available & (-available)
                if dfs(k + 1, used_mask | lowbit):
                    return True
                available &= available - 1
            return False
        
        return dfs(0, 0)
    
    # Compute f for each residue r in 0..LCM-1
    # f(r) = least L in 1..MAX_L such that check_feasible(r, L) is True
    # If no L works up to MAX_L, f is MAX_L (or we handle it)
    
    best_f = 0
    best_r = 0
    r = 0
    while r < LCM:
        if time.time() > deadline:
            break
        
        # Binary search for least L
        lo, hi = 1, MAX_L
        # First check if even L=MAX_L works
        if not check_feasible(r, MAX_L):
            # This shouldn't happen for L=144, but handle it
            f_val = MAX_L
        else:
            # Find least L
            while lo < hi:
                mid = (lo + hi) // 2
                if check_feasible(r, mid):
                    hi = mid
                else:
                    lo = mid + 1
            f_val = lo
        
        if f_val > best_f:
            best_f = f_val
            best_r = r
        
        r += 1
    
    # Return m = best_r (which is in 0..27719, well within 0..1000000)
    return int(best_r)
