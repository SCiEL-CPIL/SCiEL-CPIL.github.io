"""Rewritten program: search for m that maximizes f(12, m)."""
import time
import sys

N = 12
MAX_M = 1000000
# Maximum possible L is 144 per contract
MAX_L = 144

# Deadline guard
_START_TIME = time.time()
DEADLINE = 90.0  # seconds, evaluator kills at 90s
SAFETY_MARGIN = 5.0  # seconds to leave before deadline

def _can_match(m, L):
    """
    Check if there exist distinct b_1,...,b_12 in (m, m+L] such that i divides b_i.
    Uses bipartite matching (DFS with backtracking) on a small graph.
    """
    # For each i in 1..12, the valid b values are multiples of i in (m, m+L]
    # Build adjacency: for each i, list of candidate b values (as offsets or actual values)
    # We need to assign distinct b's.
    
    # Candidate b values for each i
    # b must be in (m, m+L], so b in [m+1, m+L]
    # b must be divisible by i
    
    # To make matching efficient, use the actual b values
    # But b values can be up to m+L which is ~1e6+144, that's fine for a set
    
    # For each i, compute candidate b values
    # b = k*i where k*i > m and k*i <= m+L
    # k > m/i and k <= (m+L)/i
    # k_min = floor(m/i) + 1
    # k_max = floor((m+L)/i)
    
    candidates = []
    for i in range(1, N + 1):
        k_min = m // i + 1
        k_max = (m + L) // i
        if k_min > k_max:
            return False
        # Store the actual b values
        vals = []
        for k in range(k_min, k_max + 1):
            vals.append(k * i)
        candidates.append(vals)
    
    # Now we need to find a matching: assign to each i a distinct b from candidates[i]
    # Use DFS with backtracking. Order i's by number of candidates (most constrained first)
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    # Map original index to position in sorted order for the matching
    # Actually, let's just use the sorted order for DFS
    # matched[b] = i that is assigned to b
    
    # Since b values can be large, use a dict or set for matched
    # But the total number of distinct b values across all candidates is at most sum of len(candidates[i])
    # which is at most 12 * 144 = 1728, but actually much less since b values overlap
    
    # Let's collect all unique b values that appear in any candidate list
    all_b = set()
    for i in range(N):
        for b in candidates[i]:
            all_b.add(b)
    
    # If total unique b values < N, no matching possible
    if len(all_b) < N:
        return False
    
    # For DFS, we'll assign in the order of 'order'
    # matched is a dict: b -> i
    matched = {}
    
    def dfs(idx):
        if idx == N:
            return True
        i = order[idx]
        for b in candidates[i]:
            if b not in matched:
                matched[b] = i
                if dfs(idx + 1):
                    return True
                del matched[b]
        return False
    
    return dfs(0)

def _f_val(m):
    """
    Compute f(12, m): the least positive L such that (m, m+L] contains
    distinct b_1,...,b_12 with i dividing b_i.
    """
    # Binary search on L? Or linear scan since L is at most 144?
    # Linear scan is fine since L <= 144
    for L in range(1, MAX_L + 1):
        if _can_match(m, L):
            return L
    return MAX_L  # Should not reach here

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_m = 12  # default
    best_score = 0  # f(12, m) value
    
    # First, evaluate the seed value
    seed_val = _f_val(12)
    best_score = seed_val
    best_m = 12
    
    # Now search for better m values
    # Strategy: random sampling + local search
    import random
    rng = random.Random(42)
    
    # Precompute: for each m, f(12, m) depends on the structure of multiples
    # in (m, m+L]. The function is periodic in some sense modulo lcm(1..12) = 27720
    
    # Let's sample m values strategically
    # Try m values that are offsets from multiples of 27720
    # Also try random values
    
    evaluated = set()
    evaluated.add(12)
    
    # Phase 1: Evaluate m = 0 and some small values
    for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 20, 50, 100]:
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        if m in evaluated:
            continue
        val = _f_val(m)
        evaluated.add(m)
        if val > best_score:
            best_score = val
            best_m = m
    
    # Phase 2: Random sampling
    # Try to find m that gives high L
    # High L means it's hard to find distinct multiples, which happens when
    # the interval (m, m+L] is "sparse" in terms of covering all 12 divisibility constraints
    
    num_samples = 2000
    for _ in range(num_samples):
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        m = rng.randint(0, MAX_M)
        if m in evaluated:
            continue
        val = _f_val(m)
        evaluated.add(m)
        if val > best_score:
            best_score = val
            best_m = m
    
    # Phase 3: Local search around best_m
    # Try neighbors of best_m
    for delta in range(-50, 51):
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        m = best_m + delta
        if m < 0 or m > MAX_M or m in evaluated:
            continue
        val = _f_val(m)
        evaluated.add(m)
        if val > best_score:
            best_score = val
            best_m = m
    
    # Phase 4: More targeted search
    # Try m values near the end of intervals where multiples are sparse
    # The lcm of 1..12 is 27720
    # Try m = k*27720 + offset for various offsets
    for k in range(0, MAX_M // 27720 + 1):
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        base = k * 27720
        for offset in [0, 1, 2, 3, 10, 50, 100, 500, 1000, 27719]:
            m = base + offset
            if m > MAX_M or m in evaluated:
                continue
            val = _f_val(m)
            evaluated.add(m)
            if val > best_score:
                best_score = val
                best_m = m
    
    # Phase 5: Broader random search with more samples
    num_samples2 = 5000
    for _ in range(num_samples2):
        if time.time() - _START_TIME > DEADLINE - SAFETY_MARGIN:
            break
        m = rng.randint(0, MAX_M)
        if m in evaluated:
            continue
        val = _f_val(m)
        evaluated.add(m)
        if val > best_score:
            best_score = val
            best_m = m
    
    # Return as Python int
    return int(best_m)

# Make sure construct returns a proper int
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}, type: {type(result)}")
