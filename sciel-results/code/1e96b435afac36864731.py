"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 5000
TIME_LIMIT = 280.0  # safety margin before 300s

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_residues(primes_list, removed, N):
    """Choose residue for each prime to minimize marginal removals."""
    residues = {}
    for p in primes_list:
        # Count unremoved in each residue class
        counts = [0] * p
        for x in range(N + 1):
            if not removed[x]:
                counts[x % p] += 1
        # Choose residue with minimum count
        min_count = min(counts)
        best_r = counts.index(min_count)
        residues[p] = best_r
        # Remove all unremoved x with x % p == best_r
        for x in range(best_r, N + 1, p):
            if not removed[x]:
                removed[x] = True
    return residues

def _compute_S(residues, N):
    """Compute S given residues dict."""
    S = []
    for x in range(N + 1):
        ok = True
        for p, r in residues.items():
            if x % p == r:
                ok = False
                break
        if ok:
            S.append(x)
    return S

def _is_admissible(S, N):
    """Check if S is admissible."""
    K = len(S)
    if K == 0:
        return True
    primes = _primes(K)
    S_set = set(S)
    for p in primes:
        classes = set(x % p for x in S_set)
        if len(classes) == p:
            return False
    return True

def construct():
    start_time = time.time()
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    best_S = None
    best_size = 0
    
    # Try different K values (upper bound on primes to constrain)
    # K must be >= |S|, and |S| decreases with K
    # Try K from 500 to 2000
    for K in [500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        primes_to_use = [p for p in all_primes if p <= K]
        
        # Multiple restarts with different random seeds for tie-breaking
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            random.seed(trial * 12345 + K)
            
            removed = [False] * (N + 1)
            # Shuffle primes order with some randomness
            primes_order = primes_to_use[:]
            random.shuffle(primes_order)
            
            # But process smaller primes first with some probability
            # Actually, let's try both: sorted and shuffled
            if trial % 2 == 0:
                primes_order = sorted(primes_order)
            else:
                random.shuffle(primes_order)
            
            residues = _greedy_residues(primes_order, removed, N)
            
            # Compute S
            S = [x for x in range(N + 1) if not removed[x]]
            size = len(S)
            
            # Check admissibility
            if size <= K and _is_admissible(S, N):
                if size > best_size:
                    best_size = size
                    best_S = S[:]
            
            # Local search: try to re-add removed elements
            # For each removed x, check if adding it back keeps admissibility
            if size > best_size and time.time() - start_time < TIME_LIMIT - 10:
                improved = True
                while improved and time.time() - start_time < TIME_LIMIT - 5:
                    improved = False
                    S_set = set(S)
                    # Try to add each removed element
                    for x in range(N + 1):
                        if removed[x]:
                            new_S = S_set | {x}
                            new_size = len(new_S)
                            if new_size <= K and _is_admissible(list(new_S), N):
                                S = list(new_S)
                                removed[x] = False
                                best_size = new_size
                                best_S = S[:]
                                improved = True
                                break
    
    # Fallback: use seed algorithm if nothing found
    if best_S is None:
        points = [1] + [p for p in all_primes if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best_S = points
    
    return best_S
