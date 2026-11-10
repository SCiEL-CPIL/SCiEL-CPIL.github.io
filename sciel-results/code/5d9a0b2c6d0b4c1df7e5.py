"""Rewritten program: search for 10 distinct multiples minimizing max-min."""
import time
import random
from math import gcd
from functools import lru_cache

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
DEADLINE = 90.0
SAFETY_MARGIN = 2.0  # seconds before hard kill

def _lcm(a, b):
    return a // gcd(a, b) * b

def _window_feasible(lo, hi):
    """Check if there exists a distinct multiple of each divisor in [lo, hi].
    Returns True if feasible, False otherwise."""
    n = hi - lo + 1
    if n < N:
        return False
    # For each divisor, the set of multiples in [lo, hi]
    # We need to pick one per divisor, all distinct
    # This is a bipartite matching problem: divisors -> values
    # But values can be up to n which can be large.
    # Use Hall's theorem check: for any subset S of divisors, 
    # the union of their multiples in [lo,hi] must have size >= |S|
    
    # For small windows, do exact matching via bipartite matching
    # For larger windows, use Hall's condition efficiently
    
    # Get the count of multiples for each divisor
    # And check Hall's condition for all subsets (only feasible for small N)
    # N=10, so 2^10 = 1024 subsets, feasible
    
    # For each divisor i, multiples in [lo, hi]:
    # first multiple >= lo: ceil(lo/d_i) * d_i
    # count: floor(hi/d_i) - ceil(lo/d_i) + 1 if ceil(lo/d_i)*d_i <= hi
    
    counts = []
    for d in DIVISORS:
        first = ((lo + d - 1) // d) * d
        if first > hi:
            return False
        last = (hi // d) * d
        counts.append(last // d - first // d + 1)
    
    # Hall's condition: for every subset S of {0..N-1}, 
    # |union of multiples of d_i for i in S| >= |S|
    # The union size is the number of integers in [lo,hi] divisible by at least one d_i in S
    # We can compute this via inclusion-exclusion or by iterating values (if window small)
    
    # For window size n, if n is small enough, just iterate
    if n <= 2000:
        # For each value in [lo, hi], mark which divisors it's divisible by
        # Then check Hall's condition
        # Actually, let's just do bipartite matching directly
        # Build adjacency: divisor i is connected to values that are multiples of d_i
        # But values are up to n=2000, divisors=10, so this is fine
        
        # For each divisor, list of offsets (0-indexed) where it has a multiple
        adj = []
        for i, d in enumerate(DIVISORS):
            first = ((lo + d - 1) // d) * d
            adj.append([])
            v = first
            while v <= hi:
                adj[i].append(v - lo)
                v += d
        
        # Bipartite matching: divisors (left, size N) to values (right, size n)
        # Use DFS-based matching
        match_val = [-1] * n  # which divisor is matched to this value offset
        def try_kun(i, visited):
            for v_off in adj[i]:
                if visited[v_off]:
                    continue
                visited[v_off] = True
                if match_val[v_off] == -1 or try_kun(match_val[v_off], visited):
                    match_val[v_off] = i
                    return True
            return False
        
        found = 0
        for i in range(N):
            visited = [False] * n
            if try_kun(i, visited):
                found += 1
        return found == N
    else:
        # For large windows, use Hall's condition with inclusion-exclusion
        # For each subset S, compute |union of multiples of d_i in S within [lo,hi]|
        # Use the fact that |multiples of d in [lo,hi]| = floor(hi/d) - ceil((lo-1)/d)
        # For union, use inclusion-exclusion over the subset
        # But 2^10 = 1024 subsets, and for each we need inclusion-exclusion over sub-subsets
        # That's 3^10 ~ 59049 operations, still feasible
        
        # Actually, simpler: for each subset S of divisors, 
        # the union size = number of x in [lo,hi] such that x is divisible by some d in S
        # = n - number of x in [lo,hi] not divisible by any d in S
        # The latter can be computed via the Chinese Remainder Theorem / inclusion-exclusion
        # over the LCMs of subsets of S
        
        # For a subset S, let D_S = {d_i : i in S}
        # Count of x in [lo,hi] divisible by at least one d in D_S:
        # By inclusion-exclusion: sum over non-empty T subset of D_S of (-1)^(|T|+1) * count(multiples of lcm(T))
        
        # Precompute all LCMs of subsets
        from itertools import combinations
        
        all_subsets = []
        for r in range(1, N+1):
            for comb in combinations(range(N), r):
                all_subsets.append(comb)
        
        # For each subset S of divisors (the outer Hall subset), 
        # we need |union| for that S
        # Hall's condition: for all S, |union_S| >= |S|
        
        # This is expensive: 2^N subsets S, and for each, 2^|S| inner subsets
        # Total: sum_{k=0}^{N} C(N,k) * 2^k = 3^N = 59049 for N=10
        # Each requires LCM computation and counting, which is fast
        
        # Precompute LCM for all subsets of divisors
        lcm_cache = {}
        def subset_lcm(sub):
            if not sub:
                return 1
            key = tuple(sorted(sub))
            if key in lcm_cache:
                return lcm_cache[key]
            # Compute LCM incrementally
            l = 1
            for idx in key:
                l = _lcm(l, DIVISORS[idx])
                if l > MAX_VAL:
                    break
            lcm_cache[key] = l
            return l
        
        # For each Hall subset S, compute union size
        for r in range(1, N+1):
            for S in combinations(range(N), r):
                # Compute |union of multiples of d_i for i in S| in [lo, hi]
                # Inclusion-exclusion
                union_count = 0
                for rr in range(1, len(S)+1):
                    for T in combinations(S, rr):
                        l = subset_lcm(T)
                        if l > hi:
                            # No multiples in [lo, hi] if l > hi and lo >= 1
                            # Actually need: count of multiples of l in [lo,hi]
                            cnt = hi // l - (lo - 1) // l
                        else:
                            cnt = hi // l - (lo - 1) // l
                        sign = 1 if rr % 2 == 1 else -1
                        union_count += sign * cnt
                if union_count < r:
                    return False
        return True

def _find_best_window(lo, hi, max_iterations=50000):
    """Given a window [lo, hi], find an actual assignment of distinct multiples.
    Returns list of values or None."""
    n = hi - lo + 1
    if n < N:
        return None
    
    # Build adjacency
    adj = []
    for i, d in enumerate(DIVISORS):
        first = ((lo + d - 1) // d) * d
        adj.append([])
        v = first
        while v <= hi:
            adj[i].append(v)
            v += d
    
    # Bipartite matching with values as right side
    # Map values to indices
    val_to_idx = {}
    all_vals = []
    for i in range(N):
        for v in adj[i]:
            if v not in val_to_idx:
                val_to_idx[v] = len(all_vals)
                all_vals.append(v)
    
    nvals = len(all_vals)
    if nvals < N:
        return None
    
    # Rebuild adj with indices
    adj_idx = []
    for i in range(N):
        adj_idx.append([val_to_idx[v] for v in adj[i]])
    
    match_val = [-1] * nvals
    def try_kun(i, visited):
        for v_idx in adj_idx[i]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_val[v_idx] == -1 or try_kun(match_val[v_idx], visited):
                match_val[v_idx] = i
                return True
        return False
    
    found = 0
    for i in range(N):
        visited = [False] * nvals
        if try_kun(i, visited):
            found += 1
    if found < N:
        return None
    
    # Extract assignment
    result = [0] * N
    for v_idx in range(nvals):
        if match_val[v_idx] != -1:
            result[match_val[v_idx]] = all_vals[v_idx]
    return result

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best = None
    best_range = float('inf')
    
    divisors = DIVISORS
    n = N
    
    # Strategy: try different window sizes, search for feasible windows
    # Start with small windows and grow
    
    # The minimum possible range is at least n-1 = 9 (since we need n distinct ints)
    # But also constrained by the divisors
    
    # Try window sizes from small to large
    # For each window size, try multiple starting positions
    
    rng = random.Random(42)
    
    # First, try a systematic search with small windows
    # Then randomized search for larger windows
    
    # Estimate: the answer is likely in the range of a few hundred to a few thousand
    # Let's try window sizes: 9, 10, 11, ..., up to some limit
    
    max_window_size = 50000  # upper bound on window size to try systematically
    
    # For each window size, try several starting points
    # The starting point should be a multiple of the LCM of some subset, or near such
    
    # LCM of all divisors:
    lcm_all = 1
    for d in divisors:
        lcm_all = _lcm(lcm_all, d)
    # lcm_all = lcm(6,10,14,15,21,22,26,33,35,39)
    # Let's compute: 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2*3*5*7*11*13 = 30030
    
    # So multiples of 30030 are good candidates for alignment
    
    # Try window sizes from 9 upward
    for wsize in range(9, max_window_size + 1):
        if time.time() > deadline:
            break
        
        # Number of starting positions to try for this window size
        # More tries for smaller windows (harder), fewer for larger
        if wsize <= 50:
            num_starts = 200
        elif wsize <= 200:
            num_starts = 100
        elif wsize <= 1000:
            num_starts = 50
        else:
            num_starts = 20
        
        # Choose starting points: multiples of lcm_all and nearby
        # Also random starts
        starts = set()
        
        # Multiples of lcm_all
        base = lcm_all
        for k in range(0, 100):
            s = k * base
            if s >= 1 and s + wsize - 1 <= MAX_VAL:
                starts.add(s)
            if s - wsize >= 1:
                starts.add(s - wsize + 1)
        
        # Random starts
        for _ in range(num_starts):
            s = rng.randint(1, MAX_VAL - wsize + 1)
            starts.add(s)
        
        # Also try starts aligned to individual divisor multiples
        for d in divisors:
            for _ in range(5):
                s = rng.randint(1, MAX_VAL - wsize + 1)
                s = ((s + d - 1) // d) * d  # round up to multiple of d
                if s >= 1 and s + wsize - 1 <= MAX_VAL:
                    starts.add(s)
        
        for lo in starts:
            if time.time() > deadline:
                break
            hi = lo + wsize - 1
            if hi > MAX_VAL:
                continue
            if _window_feasible(lo, hi):
                # Found a feasible window, extract assignment
                result = _find_best_window(lo, hi)
                if result is not None:
                    # Verify distinctness and divisibility
                    if len(set(result)) == N:
                        valid = True
                        for i, v in enumerate(result):
                            if v <= 0 or v > MAX_VAL or v % divisors[i] != 0:
                                valid = False
                                break
                        if valid:
                            cur_range = max(result) - min(result)
                            if cur_range < best_range:
                                best_range = cur_range
                                best = result[:]
                                # If we found range = wsize - 1, this is optimal for this wsize
                                # Can we do better? Only with smaller wsize
                                if best_range == wsize - 1:
                                    # Optimal for this window size, continue to smaller... 
                                    # Actually we're going up in wsize, so this is the best for this wsize
                                    # But maybe a smaller wsize was missed? We go in order so no.
                                    # Break inner loop, continue to next wsize
                                    break
    
    # If no result found, do a randomized search with larger windows
    if best is None or time.time() < deadline:
        for _ in range(10000):
            if time.time() > deadline:
                break
            # Random window
            wsize = rng.randint(50, 10000)
            lo = rng.randint(1, MAX_VAL - wsize + 1)
            hi = lo + wsize - 1
            if _window_feasible(lo, hi):
                result = _find_best_window(lo, hi)
                if result is not None:
                    if len(set(result)) == N:
                        valid = True
                        for i, v in enumerate(result):
                            if v <= 0 or v > MAX_VAL or v % divisors[i] != 0:
                                valid = False
                                break
                        if valid:
                            cur_range = max(result) - min(result)
                            if cur_range < best_range:
                                best_range = cur_range
                                best = result[:]
    
    # Final fallback: return the divisors themselves (valid but not optimal)
    if best is None:
        best = divisors[:]
    
    # Ensure types are Python int
    return [int(v) for v in best]
