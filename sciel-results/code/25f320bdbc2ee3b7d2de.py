"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflict_bitsets():
    """Build conflict bitsets for the problem.
    Two vertices i, j conflict if i+j is a perfect square.
    Also, vertex i conflicts with itself if 2*i is a perfect square.
    """
    n = 256
    conflicts = [0] * n
    for i in range(n):
        val = i + 1
        # Check if 2*val is a perfect square (self-conflict)
        if _is_square(2 * val):
            conflicts[i] |= (1 << i)
        for j in range(i + 1, n):
            if _is_square(val + (j + 1)):
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    return conflicts

def _greedy_min_degree(conflicts, n, deadline):
    """Greedy construction: repeatedly pick vertex with fewest conflicts among remaining."""
    remaining = list(range(n))
    result = []
    while remaining:
        if time.time() > deadline:
            break
        # Find vertex with minimum degree within remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = 0
            for u in remaining:
                if u != v and (conflicts[v] >> u) & 1:
                    deg += 1
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
        result.append(best_v)
        # Remove best_v and all its neighbors
        mask_to_remove = conflicts[best_v] | (1 << best_v)
        remaining = [v for v in remaining if not (mask_to_remove >> v) & 1]
    return sorted([v + 1 for v in result])

def _randomized_greedy(conflicts, n, deadline, rng):
    """Randomized greedy: pick from top-k lowest-degree vertices."""
    remaining = list(range(n))
    result = []
    k = 5
    while remaining:
        if time.time() > deadline:
            break
        # Compute degrees
        degrees = {}
        for v in remaining:
            deg = 0
            for u in remaining:
                if u != v and (conflicts[v] >> u) & 1:
                    deg += 1
            degrees[v] = deg
        # Sort by degree
        sorted_rem = sorted(remaining, key=lambda v: degrees[v])
        top_k = sorted_rem[:min(k, len(sorted_rem))]
        # Pick randomly from top_k
        v = rng.choice(top_k)
        result.append(v)
        mask_to_remove = conflicts[v] | (1 << v)
        remaining = [u for u in remaining if not (mask_to_remove >> u) & 1]
    return sorted([v + 1 for v in result])

def _local_search_polish(conflicts, n, initial_set, deadline):
    """Add-only local search: try to add vertices that don't conflict with current set."""
    current = set(initial_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if v in current:
                continue
            # Check if v conflicts with any in current
            ok = True
            for u in current:
                if (conflicts[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                current.add(v)
                improved = True
    return sorted([v + 1 for v in current])

def _verify(result):
    """Verify that result is a valid subset (no square pair sums including a=b)."""
    s = set(result)
    for a in result:
        if _is_square(2 * a):
            return False
        for b in result:
            if a <= b and _is_square(a + b):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin for 90s limit
    
    conflicts = _build_conflict_bitsets()
    n = 256
    
    # 1. Deterministic greedy (min-degree)
    best_set = _greedy_min_degree(conflicts, n, deadline)
    best_size = len(best_set)
    
    # 2. Multiple randomized greedy restarts
    rng = random.Random(42)
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        s = _randomized_greedy(conflicts, n, deadline, rng)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # 3. Local search polish on best set
    if time.time() < deadline:
        polished = _local_search_polish(conflicts, n, best_set, deadline)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
    
    # 4. Try a few more randomized restarts with polish if time allows
    for i in range(20):
        if time.time() > deadline - 3.0:
            break
        s = _randomized_greedy(conflicts, n, deadline, rng)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
        if time.time() < deadline - 2.0:
            polished = _local_search_polish(conflicts, n, best_set, deadline)
            if len(polished) > best_size:
                best_set = polished
                best_size = len(polished)
    
    # Final verification
    result = best_set
    if not _verify(result):
        # Fallback: simple valid set
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
