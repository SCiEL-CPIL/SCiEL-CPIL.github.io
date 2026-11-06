import math
from itertools import combinations
import time
import random

def construct():
    """
    Find 12 distinct positive integers <= 10000 that minimize the number of
    distinct prime factors across all pairwise sums of distinct elements.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    n = 12
    max_val = 10000
    
    # Generate primes up to 2*max_val (max possible sum is 19999)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(2 * max_val)
    prime_set = set(primes)
    
    # Precompute prime factor sets for all sums
    def factorize(x):
        factors = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        return factors
    
    sum_factors = {}
    for s in range(2, 2 * max_val + 1):
        sum_factors[s] = factorize(s)
        
    def objective(arr):
        all_primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                all_primes.update(sum_factors[s])
        return len(all_primes)

    best_set = None
    best_score = float('inf')
    
    # Initial best: random or simple structured
    # Use a simple heuristic start
    random.seed(42)
    initial_set = random.sample(range(1, max_val + 1), n)
    best_set = initial_set[:]
    best_score = objective(best_set)
    
    # Strategy: Graph Clique Search
    # We look for sets where all pairwise sums are P-smooth for small prime sets P.
    
    # Candidate prime sets
    # We focus on small primes as they are more likely to form dense graphs
    candidate_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    def find_clique_for_prime_set(P, time_limit):
        """
        Finds a clique of size n in the graph where edge (u,v) exists 
        if u+v is P-smooth.
        Returns the clique if found, else None.
        """
        P_set = set(P)
        
        # Precompute valid sums for this P
        # valid_sum[s] is True if all prime factors of s are in P_set
        # Note: s can be 1? No, min sum is 1+2=3.
        valid_sum = [False] * (2 * max_val + 1)
        for s in range(2, 2 * max_val + 1):
            facs = sum_factors[s]
            if facs.issubset(P_set):
                valid_sum[s] = True
        
        # Greedy Clique Search
        # We try multiple random starts
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > time_limit:
                return None
            
            # Start with a random vertex
            # To increase success rate, start with vertices that have high "potential"
            # But for now, just random
            start_val = random.randint(1, max_val)
            clique = [start_val]
            
            # Candidate pool: all other numbers
            # We filter candidates iteratively
            # Candidates must be distinct from clique members
            # And sum with each clique member must be valid
            
            # Optimization: Instead of checking all 10000 numbers at each step,
            # we can maintain a list of candidates.
            # However, 10000 * 12 * 66 checks is ~8M ops, which is fine in Python if optimized?
            # Actually, checking `valid_sum[a+b]` is O(1).
            # 12 steps * 10000 candidates * 12 checks = 1.4M checks per attempt.
            # 100 attempts = 140M checks. Might be slow in Python.
            
            # Better: Precompute adjacency lists?
            # Memory: 10000 * 10000 bits = 100MB. Too much?
            # Bitsets: 10000 * 10000 / 8 = 12.5 MB. Acceptable.
            # But building it for each P takes time.
            
            # Let's stick to the on-the-fly check but optimize the candidate list.
            
            candidates = list(range(1, max_val + 1))
            candidates.remove(start_val)
            
            # Sort candidates? No.
            
            found = True
            for k in range(1, n):
                if not candidates:
                    found = False
                    break
                
                # Filter candidates
                new_candidates = []
                # To make it faster, we can pick a random candidate from the current list
                # and check it? No, we need to find *one* that works.
                
                # Optimization: 
                # The condition is: for all v in clique, valid_sum[c + v] is True.
                # We can iterate through candidates and check.
                
                # To speed up, we can sample a subset of candidates if the list is huge?
                # No, we need to be exhaustive for the current step to ensure we don't miss a clique.
                # But we can break early if we find one? 
                # The greedy algorithm picks ONE candidate to add.
                # Which one? One that maximizes future potential.
                # For simplicity, pick the first one that works? Or random one that works?
                # Random one that works is better for diversity.
                
                # Let's collect all valid candidates for this step.
                valid_cands = []
                for c in candidates:
                    is_valid = True
                    for v in clique:
                        if not valid_sum[c + v]:
                            is_valid = False
                            break
                    if is_valid:
                        valid_cands.append(c)
                        
                if not valid_cands:
                    found = False
                    break
                
                # Pick one
                # Heuristic: Pick one that has the most connections to other valid_cands?
                # That requires O(|valid_cands|^2) checks.
                # For now, pick random to keep it fast.
                next_val = random.choice(valid_cands)
                clique.append(next_val)
                
                # Update candidates: remove next_val, and remove any that are invalid with next_val
                # We can just filter the remaining candidates against the new clique.
                # But we already have `valid_cands` which are valid against the *old* clique.
                # We need to filter `valid_cands` (minus next_val) against the *new* clique (which includes next_val).
                
                candidates = []
                for c in valid_cands:
                    if c == next_val:
                        continue
                    # Check against next_val only? 
                    # Yes, because c was already valid against all previous clique members.
                    if valid_sum[c + next_val]:
                        candidates.append(c)
                        
            if found and len(clique) == n:
                return clique
        
        return None

    # Iterate over prime set sizes
    # Start with size 3, then 4, then 5
    for size in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
            
        # Generate combinations
        # Prioritize combinations with smaller primes
        # We can just iterate all combinations of the first 10 primes
        prime_combos = list(combinations(candidate_primes, size))
        # Shuffle to avoid bias? Or sort by sum of primes?
        # Smaller primes generally allow more smooth numbers.
        # Let's sort by the product or sum.
        prime_combos.sort(key=lambda x: sum(x))
        
        for P in prime_combos:
            if time.time() > deadline - 2:
                break
                
            # Allocate time for this P set
            # If we have 100 combos, and 50s left, 0.5s each.
            # Let's give each 0.5s initially, adjust if needed.
            time_limit = time.time() + 0.5
            
            clique = find_clique_for_prime_set(P, time_limit)
            
            if clique:
                # Verify distinctness and bounds
                if len(set(clique)) == n and max(clique) <= max_val:
                    score = objective(clique)
                    if score < best_score:
                        best_score = score
                        best_set = clique[:]
                        
                        # If we found a very good score (e.g., 3 or 4), we can stop early
                        # since the theoretical minimum is likely 3 (sums must be even -> 2, and if odd sums exist, need another prime).
                        # Actually, if all numbers are odd, sums are even. Factors are just factors of even numbers.
                        # If all numbers are even, sums are even.
                        # Can we get score 2? P={2,3}. Sums must be 2^a 3^b.
                        # If we find score 3, it's likely optimal or near optimal.
                        if score <= 4:
                            # We can break out of loops if we are confident
                            # But let's run a bit longer to see if we can get 3.
                            pass

    # If no clique found or time running out, do a final local search polish on best_set
    if time.time() < deadline - 2:
        # Simple local search to polish
        current = best_set[:]
        current_score = best_score
        temp = 5.0
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            new_val = random.randint(1, max_val)
            if new_val in current and current.index(new_val) != idx:
                continue
            current[idx] = new_val
            new_score = objective(current)
            delta = new_score - current_score
            if delta < 0 or (temp > 0.01 and random.random() < math.exp(-delta / temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
            else:
                current[idx] = old_val
            temp *= 0.99

    return best_set

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Distinct: {len(set(result)) == 12}")
    print(f"Max: {max(result)}")
    
    # Count prime factors
    def factorize(x):
        factors = set()
        temp = x
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
            p += 1
        if temp > 1:
            factors.add(temp)
        return factors
    
    all_primes = set()
    for i in range(12):
        for j in range(i + 1, 12):
            s = result[i] + result[j]
            all_primes.update(factorize(s))
    print(f"Distinct primes: {len(all_primes)}")
    print(f"Primes: {sorted(all_primes)}")
