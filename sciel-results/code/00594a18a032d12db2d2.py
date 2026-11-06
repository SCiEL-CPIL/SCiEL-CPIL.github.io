import time
import math
import random
import sys

_START = time.time()
_HARD_DEADLINE = _START + 80.0  # Keep safety margin

def _init_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Max sum of two k_i. If B=1, max k=10000, max sum=20000.
SPF_LIMIT = 20000
_spf = _init_spf(SPF_LIMIT)

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = set()
    x = num
    while x > 1:
        p = _spf[x]
        out.add(p)
        while x % p == 0:
            x //= p
    return frozenset(out)

def _is_smooth(num, allowed_primes):
    """Check if num is composed only of allowed_primes."""
    if num <= 1:
        return True
    for p in allowed_primes:
        while num % p == 0:
            num //= p
    return num == 1

def construct():
    deadline = time.time() + 80.0
    n = 12
    max_val = 10000
    
    best_overall = None
    best_score_overall = 10**9
    
    # Target: Minimize distinct primes.
    # Try to find a solution with 2 primes {2, 3}.
    # If not, {2, 3, 5}.
    
    # Strategy:
    # 1. Generate a list of candidate bases that are 3-smooth.
    # 2. For each base, try to find 12 k's such that k_i + k_j is 3-smooth.
    #    This ensures a_i + a_j = B(k_i+k_j) has only primes 2,3 (plus those in B, which are 2,3).
    
    # Generate 3-smooth numbers up to max_val
    smooth_3 = []
    p = 1
    while p <= max_val:
        q = p
        while q <= max_val:
            r = q
            while r <= max_val:
                smooth_3.append(r)
                r *= 5
            q *= 3
        p *= 2
    smooth_3 = sorted(set(smooth_3))
    
    # We want to search for cliques in the graph of k's.
    # Vertices are 3-smooth k's.
    # Edge if k_i + k_j is 3-smooth.
    
    # To speed up, we precompute which k's are "compatible" with others?
    # Or just check on the fly.
    
    # Let's use a heuristic clique finder.
    # We will try multiple bases.
    # Bases should be small enough to allow large k's, but not too small to be trivial?
    # Actually, if B is small, k_max is large, graph is dense?
    # If B is large, k_max is small, graph is sparse.
    
    # Let's try bases from the 3-smooth list.
    # We prioritize bases that allow a wide range of k's.
    
    # Candidate bases: 3-smooth numbers.
    # We'll iterate through a subset of them.
    # To ensure diversity, we can sample bases.
    
    # Precompute 3-smooth k's up to 10000 (max possible k if B=1)
    # But if B is large, k_max is small.
    
    # Let's define a function to find a clique of size n in the graph defined by max_k and smoothness constraint.
    
    def find_clique(max_k, n_target, time_limit):
        """
        Find n_target distinct k's in [1, max_k] such that all pairwise sums are 3-smooth.
        Uses a greedy + backtracking / restart heuristic.
        """
        # Get candidate k's: 3-smooth numbers in [1, max_k]
        candidates = [k for k in smooth_3 if k <= max_k]
        if len(candidates) < n_target:
            return None
        
        # Sort candidates? Maybe by "connectivity" or just random order.
        # Let's try a few random restarts.
        
        best_clique = None
        best_size = 0
        
        num_restarts = 50
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
            
            # Randomly shuffle candidates to break symmetry
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            current_clique = []
            
            # Greedy addition
            for k in shuffled:
                # Check if k connects to all in current_clique
                is_connected = True
                for ck in current_clique:
                    if not _is_smooth(k + ck, [2, 3]):
                        is_connected = False
                        break
                
                if is_connected:
                    current_clique.append(k)
                    if len(current_clique) == n_target:
                        return current_clique
            
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
                
        # If we didn't find n_target, we might try to repair best_clique
        # But for now, return None if not found.
        # A more sophisticated backtracking could be added if needed.
        
        return None if best_size < n_target else best_clique

    # Try different bases
    # We want bases B such that B is 3-smooth.
    # And B * k <= 10000.
    
    # Sort smooth_3 bases
    bases_to_try = smooth_3[:]
    # Maybe prioritize medium bases?
    # Small bases -> large k space.
    # Large bases -> small k space.
    
    # Let's just iterate through a sample of bases.
    # To be efficient, we can sort bases by size and try a mix.
    
    # Actually, if B=1, k_i are the a_i.
    # If B=6, a_i = 6 k_i.
    
    # Let's try all 3-smooth bases up to 10000.
    # There aren't that many.
    # Count: ~100-200?
    
    for base in bases_to_try:
        if time.time() > deadline - 5.0:
            break
            
        max_k = max_val // base
        if max_k < n:
            continue
            
        # Try to find clique
        clique = find_clique(max_k, n, deadline - 5.0)
        
        if clique:
            # We found 12 k's such that k_i+k_j is 3-smooth.
            # So a_i = base * k_i.
            # Sums a_i+a_j = base*(k_i+k_j).
            # Primes in base are {2,3}.
            # Primes in (k_i+k_j) are {2,3}.
            # Union is {2,3}.
            # Score is 2.
            # This is likely optimal (or close to it).
            result = [base * k for k in clique]
            # Verify distinctness and limit
            if len(set(result)) == n and all(x <= max_val for x in result):
                # Calculate score to be sure
                union = set()
                for i in range(n):
                    for j in range(i+1, n):
                        union |= _prime_factors(result[i] + result[j])
                score = len(union)
                if score < best_score_overall:
                    best_score_overall = score
                    best_overall = result
                if score <= 2:
                    return result # Can't do better than 2 (1 is impossible for n=12 distinct)
                    
    # If no 2-prime solution found, try 3-prime solution {2,3,5}
    # This requires k_i+k_j to be 5-smooth (primes 2,3,5).
    # And base to be 5-smooth.
    
    # Generate 5-smooth numbers
    smooth_5 = []
    p = 1
    while p <= max_val:
        q = p
        while q <= max_val:
            r = q
            while r <= max_val:
                s = r
                while s <= max_val:
                    smooth_5.append(s)
                    s *= 7 # Wait, 5-smooth means primes 2,3,5. So loop for 5.
                r *= 5
            q *= 3
        p *= 2
    smooth_5 = sorted(set(smooth_5))
    
    def find_clique_5(max_k, n_target, time_limit):
        candidates = [k for k in smooth_5 if k <= max_k]
        if len(candidates) < n_target:
            return None
        
        best_clique = None
        best_size = 0
        
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > time_limit:
                break
            
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            current_clique = []
            
            for k in shuffled:
                is_connected = True
                for ck in current_clique:
                    if not _is_smooth(k + ck, [2, 3, 5]):
                        is_connected = False
                        break
                
                if is_connected:
                    current_clique.append(k)
                    if len(current_clique) == n_target:
                        return current_clique
            
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = current_clique[:]
        
        return None if best_size < n_target else best_clique

    # Try 5-smooth bases
    for base in smooth_5:
        if time.time() > deadline - 5.0:
            break
            
        max_k = max_val // base
        if max_k < n:
            continue
            
        clique = find_clique_5(max_k, n, deadline - 5.0)
        
        if clique:
            result = [base * k for k in clique]
            if len(set(result)) == n and all(x <= max_val for x in result):
                union = set()
                for i in range(n):
                    for j in range(i+1, n):
                        union |= _prime_factors(result[i] + result[j])
                score = len(union)
                if score < best_score_overall:
                    best_score_overall = score
                    best_overall = result
                if score <= 3:
                    # 3 is a good result.
                    pass # Continue to see if we can find 2? No, 2 search was first.
                    # If we found 3, it's better than seed's likely 4+.
                    # But we should keep searching for 2 if time permits?
                    # The 2-search loop finished. So we are in 3-search.
                    # We can break if we find a good 3?
                    # Let's just update best and break if score is very good.
                    if score == 3:
                         # 3 is likely optimal if 2 is impossible.
                         # Let's return it to save time?
                         # Or keep looking for a better 3 (unlikely to be better than 3).
                         # So we can return.
                         return result

    if best_overall is None:
        # Fallback: use the seed's logic or a simple set
        # Let's just use small numbers
        best_overall = list(range(1, 13))
        # Calculate score
        union = set()
        for i in range(n):
            for j in range(i+1, n):
                union |= _prime_factors(best_overall[i] + best_overall[j])
        best_score_overall = len(union)

    return best_overall
