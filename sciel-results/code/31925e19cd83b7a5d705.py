import time
from fractions import Fraction
import random
import itertools

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    primes = _sieve(1000)
    
    # Precompute reciprocals as Fractions for exact budget check
    # But for speed in loops, we can use floats for initial filtering and Fractions for final check?
    # No, Fraction is slow. Use float for search, Fraction for validation/return.
    # The budget constraint is strict: sum <= 1.
    # We will use float sum and check if sum <= 1.0 + 1e-9.
    # Then verify with Fraction at the end.
    
    best_L = 0
    best_classes = []
    
    # Candidate primes for the main set. Small primes are most important.
    # We will search subsets of these.
    search_primes = primes[:20] # First 20 primes up to 71
    
    # Generate all subsets of search_primes that fit the budget
    # To avoid 2^20 = 1M subsets, we can be smarter.
    # But 1M is okay if the filter is fast.
    # Let's just iterate.
    
    valid_sets = []
    # Use itertools.combinations for sizes 1 to 10
    # Most valid sets will be size 3-7.
    
    for k in range(1, 11):
        for combo in itertools.combinations(search_primes, k):
            # Fast float check
            s = sum(1.0 / p for p in combo)
            if s <= 1.0 + 1e-9:
                # Exact check
                exact_s = sum(Fraction(1, p) for p in combo)
                if exact_s <= 1:
                    valid_sets.append(list(combo))
                    
        if time.time() > deadline - 20:
            break

    # Sort valid sets by "potential"
    # Heuristic: Smaller primes are better.
    # Sort by sum of 1/p descending? Or by min prime?
    # Let's just process them.
    
    def compute_L(primes_set, residues, current_best):
        """
        Computes the largest L such that 1..L is covered by (p, r) in zip(primes_set, residues).
        Stops early if L cannot exceed current_best.
        """
        # We check n from 1 upwards.
        # If we find a gap at n, L = n - 1.
        # If n > current_best, we can return n - 1 (which is > current_best).
        # We don't need to go beyond current_best + 1000 unless we are improving.
        # Actually, if we reach current_best + 1 and it's covered, we have improved.
        # We can return a sentinel value or just keep checking a bit more to find the actual L.
        # For optimization, we just need to know if it's better than current_best.
        # But we also want to record the actual L for the best solution.
        
        # Optimization: Precompute nothing, just loop.
        # k = len(primes_set)
        # For each n, check if any (n % p == r)
        
        # To speed up, we can check up to limit = current_best + 50.
        # If all covered up to limit, return limit (indicating improvement).
        # The caller can then do a full check if needed.
        
        limit = current_best + 100
        if limit > 10000:
            limit = 10000
            
        for n in range(1, limit + 1):
            covered = False
            for p, r in zip(primes_set, residues):
                if n % p == r:
                    covered = True
                    break
            if not covered:
                return n - 1
        
        # If we made it to limit, L is at least limit.
        # We return limit to signal improvement.
        # The caller will verify if this is the global best.
        return limit

    def optimize_residues(primes_set, current_best):
        """
        Finds best residues for a fixed set of primes.
        Returns (best_L_local, best_residues)
        """
        k = len(primes_set)
        product = 1
        for p in primes_set:
            product *= p
            
        local_best_L = 0
        local_best_res = [0] * k
        
        if product < 5000:
            # Exhaustive search
            # Generate all residue tuples
            ranges = [range(p) for p in primes_set]
            for res_tuple in itertools.product(*ranges):
                L = compute_L(primes_set, list(res_tuple), local_best_L)
                if L > local_best_L:
                    local_best_L = L
                    local_best_res = list(res_tuple)
                    # Early exit if we reached the cap?
                    if local_best_L >= 10000:
                        break
        else:
            # Local Search
            # Multiple restarts
            for restart in range(50):
                # Random init
                res = [random.randint(0, p - 1) for p in primes_set]
                cur_L = compute_L(primes_set, res, local_best_L)
                
                if cur_L > local_best_L:
                    local_best_L = cur_L
                    local_best_res = res[:]
                
                # Hill climbing
                for step in range(200):
                    if time.time() > deadline - 5:
                        return local_best_L, local_best_res
                    
                    # Pick random prime index
                    idx = random.randint(0, k - 1)
                    old_r = res[idx]
                    new_r = random.randint(0, primes_set[idx] - 1)
                    if new_r == old_r:
                        continue
                    
                    res[idx] = new_r
                    new_L = compute_L(primes_set, res, local_best_L)
                    
                    if new_L > local_best_L:
                        local_best_L = new_L
                        local_best_res = res[:]
                    else:
                        # Revert
                        res[idx] = old_r
                        
                    if local_best_L >= 10000:
                        return local_best_L, local_best_res
                        
        return local_best_L, local_best_res

    # Process valid sets
    # To improve efficiency, process sets with smaller primes first?
    # Or just iterate.
    
    for p_set in valid_sets:
        if time.time() > deadline - 10:
            break
            
        L, res = optimize_residues(p_set, best_L)
        
        if L > best_L:
            # Verify exact L with a full scan up to 10000
            # Since compute_L might have stopped early at limit
            # We need the exact L for the score.
            # But for the score, we just need the classes.
            # The evaluator will compute L.
            # We just need to ensure our candidate is valid and likely good.
            
            # Let's do a quick exact L calculation for this candidate to update best_L accurately
            # to help pruning in subsequent compute_L calls.
            
            exact_L = 0
            # Max L is 10000
            for n in range(1, 10001):
                covered = False
                for p, r in zip(p_set, res):
                    if n % p == r:
                        covered = True
                        break
                if not covered:
                    break
                exact_L = n
                
            if exact_L > best_L:
                best_L = exact_L
                best_classes = [[p, r] for p, r in zip(p_set, res)]
                
                # If we hit the cap, we can stop?
                # No, maybe we can find a better set of primes?
                # But L is capped at 10^5. If we hit 10000, we are doing well.
                # The target is 999.
                # So if best_L >= 1000, we are safe.
                # We can break early if best_L is very high?
                # Let's keep searching for a bit to see if we can get higher.
                
    # Fallback
    if not best_classes:
        # Just return a valid set
        # e.g., [2, 1]
        best_classes = [[2, 1]]
        best_L = 1 # Actually covers all odds, so L is at least 1, 3, 5...
        # Wait, if best_classes is [[2,1]], it covers 1,3,5...
        # 2 is not covered. So L=1.
        # We should return something better if possible.
        # The loop above should have found something better.

    return best_classes

