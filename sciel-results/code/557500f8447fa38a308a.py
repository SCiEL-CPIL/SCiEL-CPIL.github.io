"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_7_smooth(n):
    """Check if n is divisible only by primes 2,3,5,7."""
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def _build_compatibility_graph(max_val, target_primes):
    """
    Build a compatibility graph where edge (a,b) exists iff a+b has only
    prime factors from target_primes. Returns list of sets: compat[i] = set of j
    such that (i+1)+(j+1) is target_primes-smooth.
    """
    n = max_val
    compat = [set() for _ in range(n)]
    # Precompute smoothness for all sums 2..2*max_val
    smooth = [False] * (2 * max_val + 1)
    for s in range(2, 2 * max_val + 1):
        # Check if s is target_primes-smooth
        x = s
        for p in target_primes:
            while x % p == 0:
                x //= p
        smooth[s] = (x == 1)
    
    for i in range(1, max_val + 1):
        for j in range(i + 1, max_val + 1):
            if smooth[i + j]:
                compat[i - 1].add(j - 1)
                compat[j - 1].add(i - 1)
    return compat

def _find_clique_backtrack(compat, target_size, deadline, start_time, max_nodes=None):
    """
    Backtracking search for a clique of target_size in the compatibility graph.
    Uses forward checking: maintain candidate set that must be adjacent to all chosen.
    Returns list of node indices (0-based) or None.
    """
    n = len(compat)
    if max_nodes is not None and n > max_nodes:
        # Restrict to first max_nodes nodes for tractability
        pass
    
    best = [None]
    
    def backtrack(chosen, candidates):
        if time.time() > deadline:
            return False
        if len(chosen) == target_size:
            best[0] = chosen[:]
            return True
        if not candidates:
            return False
        # Need enough candidates to complete
        if len(candidates) < target_size - len(chosen):
            return False
        
        # Try each candidate in candidates
        cand_list = sorted(candidates)
        for node in cand_list:
            if time.time() > deadline:
                return False
            new_chosen = chosen + [node]
            # Forward check: new candidates must be adjacent to node AND all previously chosen
            new_candidates = set()
            for c in candidates:
                if c <= node:  # maintain ordering to avoid duplicates
                    continue
                if c in compat[node]:
                    # Also must be adjacent to all in chosen
                    ok = True
                    for ch in chosen:
                        if c not in compat[ch]:
                            ok = False
                            break
                    if ok:
                        new_candidates.add(c)
            if len(new_candidates) >= target_size - len(new_chosen) - 1 or len(new_chosen) == target_size - 1:
                if backtrack(new_chosen, new_candidates):
                    return True
        return False
    
    # Initial candidates: all nodes
    initial_candidates = set(range(n))
    backtrack([], initial_candidates)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    best_arr = None
    best_count = float('inf')
    
    # Phase 1: Try to find a 12-clique in the 7-smooth compatibility graph
    # This guarantees at most 4 distinct primes {2,3,5,7}
    target_primes = [2, 3, 5, 7]
    
    # Start with a restricted universe for tractability
    # Try increasing sizes
    for max_val in [500, 1000, 2000, 5000, 10000]:
        if time.time() > deadline - 10:
            break
        
        if max_val <= 500:
            # Small enough for full graph
            compat = _build_compatibility_graph(max_val, target_primes)
            clique = _find_clique_backtrack(compat, 12, deadline - 5, start_time)
            if clique is not None:
                arr = [c + 1 for c in clique]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr[:]
                    if best_count <= 4:
                        return best_arr
        elif max_val <= 2000:
            # Use a heuristic: try to build clique greedily with backtracking
            # First, find high-degree nodes
            smooth = [False] * (2 * max_val + 1)
            for s in range(2, 2 * max_val + 1):
                x = s
                for p in target_primes:
                    while x % p == 0:
                        x //= p
                smooth[s] = (x == 1)
            
            # Degree of each node
            degrees = [0] * max_val
            for i in range(1, max_val + 1):
                for j in range(i + 1, max_val + 1):
                    if smooth[i + j]:
                        degrees[i - 1] += 1
                        degrees[j - 1] += 1
            
            # Sort by degree descending, take top candidates
            sorted_nodes = sorted(range(max_val), key=lambda x: -degrees[x])
            top_candidates = sorted_nodes[:500]
            
            # Build subgraph on top candidates
            sub_compat = [set() for _ in range(len(top_candidates))]
            node_to_idx = {node: idx for idx, node in enumerate(top_candidates)}
            for i_idx in range(len(top_candidates)):
                for j_idx in range(i_idx + 1, len(top_candidates)):
                    a = top_candidates[i_idx] + 1
                    b = top_candidates[j_idx] + 1
                    if smooth[a + b]:
                        sub_compat[i_idx].add(j_idx)
                        sub_compat[j_idx].add(i_idx)
            
            clique = _find_clique_backtrack(sub_compat, 12, deadline - 5, start_time)
            if clique is not None:
                arr = [top_candidates[c] + 1 for c in clique]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr[:]
                    if best_count <= 4:
                        return best_arr
        else:
            # For larger max_val, use AP-based approach
            # Numbers ≡ 0 mod M where M = 2*3*5*7 = 210
            # Sums ≡ 0 mod 210, so factors of sum/210 matter
            M = 210
            arr = []
            val = M  # start at 210 (≡0 mod 210)
            while val <= max_val and len(arr) < 12:
                arr.append(val)
                val += M
            if len(arr) >= 12:
                arr = arr[:12]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr[:]
                    if best_count <= 4:
                        return best_arr
    
    # Phase 2: Fallback - systematic AP search
    if best_arr is None or best_count > 4:
        for d in range(2, 300):
            if time.time() > deadline - 5:
                break
            for r in range(d):
                first = r
                if first < 1:
                    first += d
                arr = []
                val = first
                while val <= 10000 and len(arr) < 12:
                    arr.append(val)
                    val += d
                if len(arr) < 12:
                    continue
                arr = arr[:12]
                count = _count_distinct_primes(arr, pf)
                if count < best_count:
                    best_count = count
                    best_arr = arr[:]
                    if best_count <= 4:
                        return best_arr
    
    # Phase 3: Local search to improve
    import random
    random.seed(42)
    
    if best_arr is None:
        best_arr = list(range(1, 13))
        best_count = _count_distinct_primes(best_arr, pf)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            arr = best_arr[:]
        else:
            d = random.choice([2, 3, 4, 6, 12, 24, 30, 42, 60, 120, 210])
            r = random.randint(0, d - 1)
            first = r
            if first < 1:
                first += d
            arr = []
            val = first
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += d
            if len(arr) < 12:
                while len(arr) < 12:
                    x = random.randint(1, 10000)
                    if x not in arr:
                        arr.append(x)
            arr = arr[:12]
        
        count = _count_distinct_primes(arr, pf)
        if count < best_count:
            best_count = count
            best_arr = arr[:]
            if best_count <= 4:
                return best_arr
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for pos in range(12):
                old = arr[pos]
                candidates = set()
                for delta in range(-50, 51, 10):
                    c = old + delta
                    if 1 <= c <= 10000 and c not in arr:
                        candidates.add(c)
                for _ in range(30):
                    c = random.randint(1, 10000)
                    if c not in arr:
                        candidates.add(c)
                
                for c in list(candidates)[:50]:
                    arr[pos] = c
                    new_count = _count_distinct_primes(arr, pf)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 4:
                                return best_arr
                    else:
                        arr[pos] = old
                if time.time() > deadline - 5:
                    break
        
        if time.time() > deadline - 5:
            break
    
    return best_arr

# EVOLVE-BLOCK-END
