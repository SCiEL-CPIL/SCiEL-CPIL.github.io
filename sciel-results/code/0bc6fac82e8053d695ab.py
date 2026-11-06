"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [i for i in range(n + 1) if sieve[i]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has count >= 2."""
    L = min(cap, 50000)  # reasonable upper bound for search
    counts = np.zeros(L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # positions: r, r+p, r+2p, ... up to L
        # but r could be 0, meaning p, 2p, 3p, ...
        start = r if r > 0 else p
        if start <= L:
            counts[start::p] += 1
    # find first position where count < 2
    # check from 1 onwards
    if L > 0 and counts[1] < 2:
        return 0
    # find the longest prefix where all counts >= 2
    # use cumsum trick: find first index where count < 2
    mask = counts[1:] < 2
    if not mask.any():
        return L
    first_bad = np.argmax(mask) + 1
    return first_bad

def _score(residues, primes, cap=100000):
    return _compute_L(residues, primes, cap)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)  # 168
    
    # Initial construction: try to make each integer covered by at least 2 primes
    # Strategy: assign residues so that small numbers get hit by multiple primes
    # For prime p, residue r means numbers r, r+p, r+2p, ... are covered
    # We want every number 1..L to be in at least 2 such sequences
    
    # Initial: set r_p = 1 for all p (so 1 is covered by all, 1+p by all, etc.)
    # But that's wasteful. Better: distribute residues.
    # Let's try: for the i-th prime, set r_p = (i % p) or similar
    # Actually, a better initial: try to cover 1..200 well
    
    # Simple initial: r_p = 1 for small primes, vary for larger
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = 1  # all cover 1, 1+p, 1+2p, ...
    
    current_score = _score(residues.tolist(), primes)
    best_residues = residues.copy()
    best_score = current_score
    
    # Local search: hill climbing
    # For each position, try all possible residues for that prime
    # Accept if it improves the score
    rng = np.random.default_rng(42)
    
    # Precompute for speed: for each prime p, the step pattern
    # We'll use a numpy-based approach for fast re-scoring
    
    # To speed up, we'll maintain a counts array and update incrementally
    # But for simplicity, let's do full re-score but limit the search space
    
    # Key optimization: only try residues that are "useful" - 
    # for prime p, residue r is useful if r <= some bound or r % p is in a useful range
    # Actually, for covering 1..L, residue r mod p covers r, r+p, ... 
    # If r > L, then the first covered number is r (if r <= L) or none in 1..L
    # Wait: if r=0, covers p, 2p, 3p, ...
    # if r>0, covers r, r+p, r+2p, ...
    
    # For covering 1..L, we need r in [1, min(p-1, L)] or r=0 (covers p, 2p, ...)
    # But r=0 means the covered numbers are p, 2p, 3p, ... which start at p
    # So for small L < p, r=0 covers nothing in 1..L
    # For r in 1..L, it covers r, r+p, ...
    
    # So useful residues for covering 1..L are r in [1, min(p-1, L)] plus r=0 if L >= p
    
    max_L_search = 5000  # how far we try to push
    
    # Incremental scoring: maintain counts array
    def compute_counts(res):
        counts = np.zeros(max_L_search + 1, dtype=np.int32)
        for r, p in zip(res, primes):
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L_search:
                counts[start::p] += 1
        return counts
    
    def find_L_from_counts(counts):
        # Find largest L such that counts[1..L] all >= 2
        if max_L_search > 0 and counts[1] < 2:
            return 0
        # Find first index where count < 2
        mask = counts[1:] < 2
        if not mask.any():
            return max_L_search
        first_bad = int(np.argmax(mask)) + 1
        return first_bad
    
    current_counts = compute_counts(residues)
    current_L = find_L_from_counts(current_counts)
    best_L = current_L
    best_res = residues.copy()
    
    # Local search with incremental updates
    # When we change residue of prime i from old_r to new_r:
    # decrement counts for old_r pattern, increment for new_r pattern
    
    def apply_change(counts, i, new_r, sign):
        """Apply a change to counts for prime i. sign=+1 to add, -1 to remove."""
        p = primes[i]
        if new_r == 0:
            start = p
        else:
            start = new_r
        if start <= max_L_search:
            counts[start::p] += sign
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of primes to try
        order = rng.permutation(n_primes)
        for i in order:
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = int(residues[i])
            
            # Try candidate residues
            # Focus on residues that could help: 1..min(p-1, max_L_search) and 0
            # But that's too many for large p. Let's be smarter.
            # Try a subset: residues 1..min(p-1, 200) and 0
            max_try = min(p - 1, 200)
            candidates = list(range(1, max_try + 1)) + [0]
            rng.shuffle(candidates)
            
            for new_r in candidates:
                if time.time() > deadline:
                    break
                if new_r == old_r:
                    continue
                
                # Try this change
                apply_change(current_counts, i, old_r, -1)
                apply_change(current_counts, i, new_r, +1)
                new_L = find_L_from_counts(current_counts)
                
                if new_L > current_L:
                    # Accept
                    current_L = new_L
                    residues[i] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_res = residues.copy()
                    improved = True
                    # Don't break - try more changes in this iteration
                else:
                    # Revert
                    apply_change(current_counts, i, new_r, -1)
                    apply_change(current_counts, i, old_r, +1)
    
    # If we have time, do a second pass with larger search range
    if time.time() < deadline - 20:
        # Extend max_L_search and try to push further
        # For now, return best found
        pass
    
    return best_res.tolist()
# EVOLVE-BLOCK-END
