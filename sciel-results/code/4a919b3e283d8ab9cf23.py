"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256
_DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START
def _is_sidon(S):
    """Check if S is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(S):
    """Return set of all pairwise sums a+b for a<=b in S."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    return sums

def _is_maximal(S, N, sums=None):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    S_set = set(S)
    if sums is None:
        sums = _get_sums(S)
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if adding x preserves Sidon
        # New sums: x+x and x+a for a in S
        # 1. x+x must not be in sums
        if 2 * x in sums:
            continue
        # 2. x+a must not be in sums for any a in S
        # Note: if x+a is in sums, it's formed by some b+c in S.
        # Since x not in S, x+a != x+x (unless a=x, impossible).
        # Also x+a != x+b unless a=b.
        # So we just need to check intersection of {x+a for a in S} with sums.
        blocked = False
        for a in S:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_maximal_extend(S, N, randomize=True):
    """Extend S to a maximal Sidon set greedily."""
    S = list(S)
    sums = _get_sums(S)
    S_set = set(S)
    
    candidates = list(range(1, N + 1))
    if randomize:
        random.shuffle(candidates)
    else:
        candidates.sort()
        
    for x in candidates:
        if x in S_set:
            continue
        # Check if x can be added
        if 2 * x in sums:
            continue
        ok = True
        for a in S:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            S.append(x)
            S_set.add(x)
            # Update sums
            sums.add(2 * x)
            for a in S:
                if a != x:
                    sums.add(x + a)
    
    S.sort()
    return S

def _count_blocked(S, N):
    """Count how many missing elements are blocked by each element in S."""
    S_set = set(S)
    sums = _get_sums(S)
    blocked_count = {x: 0 for x in S}
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x is blocked
        if 2 * x in sums:
            # 2x is a sum from some pair in S
            # Find which elements in S contribute to this
            for i, a in enumerate(S):
                for j in range(i, len(S)):
                    if S[i] + S[j] == 2 * x:
                        blocked_count[a] += 1
                        blocked_count[S[j]] += 1
                        break
        else:
            for a in S:
                if (x + a) in sums:
                    blocked_count[a] += 1
                    break
    return blocked_count

def _try_shrink(S, N):
    """Try to shrink a maximal Sidon set by removing elements that keep it maximal."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        # Try removing each element, prioritizing those that block fewer elements
        if len(S) <= 1:
            break
        blocked = _count_blocked(S, N)
        # Sort by blocked count ascending (remove least useful first)
        indices = sorted(range(len(S)), key=lambda i: blocked[S[i]])
        for i in indices:
            candidate = S[:i] + S[i+1:]
            if _is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def _local_search_swap(S, N, max_iters=100):
    """Try local search: swap pairs of elements to maintain maximality and potentially enable further shrinkage."""
    S = list(S)
    S_set = set(S)
    for _ in range(max_iters):
        improved = False
        # Try swapping two elements with a missing element
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                # Try removing S[j] and adding some x
                for x in range(1, N + 1):
                    if x in S_set or x == S[j]:
                        continue
                    candidate = S[:i] + [x] + S[i+1:j] + S[j+1:]
                    # Remove S[j], add x at position i
                    # Actually let's just try: remove S[j], add x
                    candidate = [y for y in S if y != S[j]] + [x]
                    candidate.sort()
                    if _is_maximal(candidate, N):
                        # Check if this enables further shrinkage
                        shrunk = _try_shrink(candidate, N)
                        if len(shrunk) < len(S):
                            S = shrunk
                            S_set = set(S)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return S

def _random_small_sidon(min_size=5, max_size=10):
    """Generate a random small Sidon set."""
    S = []
    sums = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    target_size = random.randint(min_size, max_size)
    for x in candidates:
        if len(S) >= target_size:
            break
        if 2 * x in sums:
            continue
        ok = True
        for a in S:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            S.append(x)
            sums.add(2 * x)
            for a in S:
                if a != x:
                    sums.add(x + a)
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    # Strategy:
    # 1. Generate many random small Sidon sets.
    # 2. Extend them to maximal Sidon sets.
    # 3. Try to shrink the maximal sets.
    # 4. Keep the smallest.
    
    # We can also try deterministic greedy with different orderings
    # but random small starts might find "holes" more effectively.
    
    # Number of iterations depends on time.
    # Each iteration:
    # - Random small set gen: O(N) approx
    # - Extend: O(N * K)
    # - Shrink: O(K^2 * N) approx? 
    #   Shrink loop: tries to remove K elements.
    #   Each removal check: O(N * K).
    #   So one shrink pass is O(K^2 * N).
    #   With K=15, N=256, 225 * 256 ~ 57k ops.
    #   Very fast.
    
    # We can do thousands of iterations.
    
    rng = random.Random(42)
    
    # Phase 1: Random restarts
    while time.time() - start_time < _DEADLINE - 2:
        # Generate a random small Sidon set
        # Vary the size
        S_init = _random_small_sidon(3, 15)
        
        # Extend to maximal
        # Use random ordering for extension to get different maximal sets
        S_max = _greedy_maximal_extend(S_init, N, randomize=True)
        
        # Shrink
        S_shrunk = _try_shrink(S_max, N)
        
        # Local search to try to improve
        S_local = _local_search_swap(S_shrunk, N, max_iters=20)
        
        if len(S_local) < best_size:
            best_size = len(S_local)
            best_set = S_local
            # If we find a very small set, we might break early?
            if best_size <= 15:
                break
                
    # Phase 2: If best is still large, try specific heuristics
    if best_set is None or best_size > 18:
        # Try a few structured starts
        for start in [1, 2, 3]:
            if time.time() - start_time > _DEADLINE - 2:
                break
            S_struct = list(range(start, N + 1, 4))  # Every 4th number
            S_struct = [x for x in S_struct if x <= N]
            S_max = _greedy_maximal_extend(S_struct, N, randomize=False)
            S_shrunk = _try_shrink(S_max, N)
            S_local = _local_search_swap(S_shrunk, N, max_iters=10)
            if len(S_local) < best_size:
                best_size = len(S_local)
                best_set = S_local

    # Fallback
    if best_set is None:
        S = _greedy_maximal_extend([], N, randomize=False)
        best_set = _try_shrink(S, N)
        
    return best_set
# EVOLVE-BLOCK-END
