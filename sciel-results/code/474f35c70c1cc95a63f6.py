import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    PERIOD = 27720
    MAX_L = 144
    MIN_L = 12
    
    # Precompute for each residue r in [0, PERIOD), the set of valid offsets for each divisor i
    # An offset j (1-indexed) is valid for divisor i if (r + j) % i == 0
    # i.e., j ≡ (-r) mod i, so j = (-r mod i), (-r mod i) + i, (-r mod i) + 2i, ...
    
    # For each residue r, we want to find the maximum minimal window length
    # f(12, r) = least L such that there exist distinct b_1,...,b_12 in (r, r+L] with i|b_i
    
    # Strategy: For each residue r, precompute the sorted list of valid offsets for each i
    # Then use a sliding window / bipartite matching approach
    
    # Key insight: The problem is periodic with period LCM(1..12) = 27720
    # For a given m, only m mod 27720 matters
    
    # Precompute valid offset sets for each residue and divisor
    # valid_offsets[r][i] = sorted list of j in [1, MAX_L] such that (r + j) % i == 0
    
    # But we need to find the MINIMUM L such that we can assign distinct offsets to all 12 divisors
    # This is a bipartite matching problem
    
    # Alternative structural approach:
    # For each residue r, the valid offsets for divisor i are: j ≡ (-r) mod i, j >= 1
    # So j = ((-r) % i) if ((-r) % i) >= 1, else ((-r) % i) + i
    # Then j, j+i, j+2i, ...
    
    # For a window of length L starting at r+1, the valid offsets for i are:
    # j in [1, L] such that j ≡ (-r) mod i
    
    # Let's precompute for each residue r and each L from 12 to 144:
    # Can we find a perfect matching?
    
    # More efficient: For each residue r, compute the "earliest" valid offset for each i
    # earliest_offset[r][i] = smallest j >= 1 such that (r+j) % i == 0
    # This is either ((-r) % i) if that's >= 1, or ((-r)%i) + i
    
    # But we need DISTINCT offsets, so it's not just about earliest offsets
    
    # Let me think differently. For a fixed residue r, define:
    # S_i = {j in Z+ : (r+j) % i == 0} = {j : j ≡ (-r) mod i}
    # We need to pick distinct j_1, ..., j_12 with j_i in S_i, minimizing max(j_i)
    
    # This is equivalent to: find the minimum L such that there exist distinct j_1,...,j_12 
    # with j_i ≡ (-r) mod i and j_i <= L
    
    # For each residue r, let's compute this minimum L using a greedy + matching approach
    
    # Precomputation: For each r in [0, PERIOD), compute min_L[r]
    
    # Use the fact that for a fixed r, the constraint j_i ≡ a_i (mod i) where a_i = (-r) % i
    # (with the convention that if a_i == 0, then a_i = i)
    
    # Let me compute this for all residues using a clever approach
    
    # For each residue r, the residues a_i = (-r) mod i for i=1..12
    # Note: (-r) mod i = (i - (r mod i)) mod i, and if r mod i == 0, then (-r) mod i = 0, 
    # meaning j must be a multiple of i, so smallest positive j is i
    
    # Let me just brute force over residues but with an efficient matching check
    
    # For a given r and candidate L, check if matching exists:
    # For each i, the valid offsets in [1,L] are: 
    #   start_i = ((-r) % i) if ((-r) % i) != 0 else i
    #   offsets: start_i, start_i + i, start_i + 2i, ... up to L
    
    # Use Hall's theorem or direct bipartite matching
    
    # Since MAX_L is only 144, and we have 12 divisors, we can use a simple DFS matching
    
    # Precompute for each residue r the minimum L
    
    # Optimization: Group residues by their pattern of (-r) mod i for i=1..12
    # But there are at most 27720 residues, so let's just compute directly
    
    # Let's precompute min_L for all 27720 residues
    
    min_L_table = [0] * PERIOD
    
    for r in range(PERIOD):
        if time.time() > deadline - 5:
            # Not enough time, use best found so far
            break
        
        # For this residue, find the minimum L
        # Binary search on L
        lo = MIN_L
        hi = MAX_L
        
        while lo < hi:
            mid = (lo + hi) // 2
            if _check_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        min_L_table[r] = lo
    
    # Find the residue with maximum min_L
    best_L = 0
    best_r = 0
    for r in range(PERIOD):
        if min_L_table[r] > best_L:
            best_L = min_L_table[r]
            best_r = r
    
    # Return m = best_r (which is in [0, 27720) and thus in [0, 1000000])
    return int(best_r)


def _check_matching(r, L):
    """Check if there exists a perfect matching for residue r and window length L."""
    # For each divisor i (1..12), compute valid offsets in [1, L]
    # j ≡ (-r) mod i, j >= 1
    
    # Represent as bitmasks for efficiency
    # Bit j-1 is set if offset j is valid for divisor i
    
    masks = [0] * 13
    for i in range(1, 13):
        rem = (-r) % i
        if rem == 0:
            start = i
        else:
            start = rem
        # j ranges from start to L step i
        for j in range(start, L + 1, i):
            masks[i] |= (1 << (j - 1))
    
    # Order by most constrained first (fewest options)
    indices = list(range(1, 13))
    indices.sort(key=lambda i: masks[i].bit_count())
    
    # DFS with memoization
    memo = {}
    
    def dfs(k, used_mask):
        if k == 12:
            return True
        key = (k, used_mask)
        if key in memo:
            return memo[key]
        
        i = indices[k]
        avail = masks[i] & ~used_mask
        while avail:
            lsb = avail & -avail
            if dfs(k + 1, used_mask | lsb):
                memo[key] = True
                return True
            avail ^= lsb
        
        memo[key] = False
        return False
    
    return dfs(0, 0)
