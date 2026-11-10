"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    # For each i in 1..12, find multiples of i in (m, m+L]
    # Build bipartite graph: left = indices 1..12, right = candidate numbers
    # Use DFS-based bipartite matching
    
    candidates = {}
    for i in range(1, 13):
        # Smallest multiple of i > m
        start = (m // i + 1) * i
        # All multiples of i in (m, m+L]
        mults = []
        x = start
        while x <= m + L:
            mults.append(x)
            x += i
        candidates[i] = mults
    
    # Check if each index has at least one candidate
    for i in range(1, 13):
        if not candidates[i]:
            return False
    
    # Bipartite matching: assign each index i to a distinct number from candidates[i]
    # Right side: all unique candidate numbers
    # Use standard DFS augmenting path
    
    # Map numbers to index list for reverse lookup
    num_to_indices = {}
    for i in range(1, 13):
        for num in candidates[i]:
            if num not in num_to_indices:
                num_to_indices[num] = []
            num_to_indices[num].append(i)
    
    # Match: match[i] = number assigned to index i, or -1
    match = {}
    for i in range(1, 13):
        match[i] = -1
    
    def dfs(i, visited):
        for num in candidates[i]:
            if num in visited:
                continue
            visited.add(num)
            if match.get(num, -1) == -1 or dfs(match[num], visited):
                match[i] = num
                match[num] = i
                return True
        return False
    
    for i in range(1, 13):
        if not dfs(i, set()):
            return False
    
    return True

def _compute_f(m, max_L=144):
    """Compute f(12, m): least positive L such that (m, m+L] has valid matching."""
    for L in range(1, max_L + 1):
        if _has_perfect_matching(m, L):
            return L
    return max_L  # Should not happen per problem statement

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    best_m = 12
    best_f = _compute_f(best_m)
    
    # Search strategy: focus on m values where m ≡ 0 (mod 12) to force b_12 far
    # But also try other residues to find if any give larger f
    # First, scan m ≡ 0 mod 12 in [0, 1000000]
    
    # Phase 1: Scan all m ≡ 0 mod 12 (these give b_12 at m+12, so L≥12)
    # Sample strategically due to time constraints
    step = 12
    candidates_m = list(range(0, 1000001, step))
    
    # Random sample + systematic scan
    import random
    random.seed(42)
    
    # Test a subset first
    sample_size = min(len(candidates_m), 500)
    sampled = random.sample(candidates_m, sample_size)
    sampled.sort()
    
    for m in sampled:
        if time.time() - start_time > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Phase 2: If we have time, do a denser scan around promising regions
    if time.time() - start_time < deadline - 5:
        # Try m ≡ 0 mod 12 more densely
        for m in range(0, 1000001, step):
            if time.time() - start_time > deadline:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    # Phase 3: Try other residues that might also work
    if time.time() - start_time < deadline - 3:
        for residue in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            if time.time() - start_time > deadline:
                break
            for m in range(residue, 1000001, 12):
                if time.time() - start_time > deadline:
                    break
                # Only check if this could beat best_f
                # Quick check: what's the minimum possible L for this m?
                min_L = 0
                for i in range(1, 13):
                    first_mult = (m // i + 1) * i
                    dist = first_mult - m
                    min_L = max(min_L, dist)
                if min_L >= best_f:
                    f_val = _compute_f(m)
                    if f_val > best_f:
                        best_f = f_val
                        best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
