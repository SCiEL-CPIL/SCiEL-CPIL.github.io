import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    def is_smooth(n, primes):
        """Check if n is smooth with respect to the given prime set."""
        if n <= 0:
            return False
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1
    
    def generate_smooth_numbers(limit, primes):
        """Generate all smooth numbers up to limit using the given primes."""
        smooth = set()
        def recurse(idx, current):
            if current > limit:
                return
            if current > 0:
                smooth.add(current)
            if idx < len(primes):
                val = current
                while val <= limit:
                    recurse(idx + 1, val)
                    val *= primes[idx]
        recurse(0, 1)
        return sorted(smooth)
    
    def factorize(n):
        """Return set of prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    def count_distinct_primes(nums):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                primes |= factorize(s)
        return len(primes), primes
    
    # Strategy 1: Try to find 12 distinct positive integers whose pairwise sums
    # are all smooth with respect to a small prime set.
    
    # Try prime sets in order of size
    prime_sets_to_try = [
        [2],           # 1 prime - powers of 2
        [2, 3],        # 2 primes - 3-smooth
        [2, 3, 5],     # 3 primes - 5-smooth
        [2, 3, 5, 7],  # 4 primes - 7-smooth (current best)
        [2, 3, 5, 7, 11],  # 5 primes
    ]
    
    best_answer = None
    best_prime_count = float('inf')
    
    for primes in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        
        # The multiplier is the product of the primes
        multiplier = 1
        for p in primes:
            multiplier *= p
        
        # We need 12 distinct positive integers k_1..k_12 such that:
        # 1. multiplier * k_i <= 10000 for all i
        # 2. k_i + k_j is smooth (with respect to primes) for all i < j
        max_k = 10000 // multiplier
        
        if max_k < 12:
            continue
        
        # Generate all smooth numbers up to 2*max_k
        smooth_limit = 2 * max_k
        smooth_nums = set(generate_smooth_numbers(smooth_limit, primes))
        
        # Build adjacency: for each k in 1..max_k, which other values v satisfy
        # k + v is smooth?
        # adj[k] = set of v != k such that k + v is in smooth_nums
        adj = {}
        for k in range(1, max_k + 1):
            adj[k] = set()
            for v in range(1, max_k + 1):
                if v != k and (k + v) in smooth_nums:
                    adj[k].add(v)
        
        # Find a clique of size 12 using backtracking
        def find_clique(size, candidates, clique, adj, deadline):
            if len(clique) == size:
                return clique
            if not candidates:
                return None
            if time.time() > deadline:
                return None
            
            # Try each candidate
            for v in sorted(candidates):
                new_clique = clique + [v]
                # All remaining candidates must be adjacent to v
                new_candidates = [u for u in candidates if u > v and u in adj[v]]
                result = find_clique(size, new_candidates, new_clique, adj, deadline)
                if result is not None:
                    return result
            return None
        
        # Use a time-limited search
        search_deadline = min(deadline - 3, start_time + 15)
        clique = find_clique(12, list(range(1, max_k + 1)), [], adj, search_deadline)
        
        if clique is not None:
            answer = [k * multiplier for k in clique]
            if len(set(answer)) == 12 and all(1 <= x <= 10000 for x in answer):
                pc, _ = count_distinct_primes(answer)
                if pc < best_prime_count:
                    best_prime_count = pc
                    best_answer = answer
                # We found a valid solution; can stop early if prime count is minimal
                if len(primes) <= best_prime_count:
                    return answer
    
    # Strategy 2: If no smooth construction worked, use local search
    if best_answer is None:
        # Start with a reasonable initial set
        best_set = list(range(1, 13))
        best_score, _ = count_distinct_primes(best_set)
        
        random.seed(42)
        no_improve = 0
        
        while time.time() < deadline - 5 and no_improve < 5000:
            current = best_set[:]
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try several random replacements
            found_improvement = False
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_val = random.randint(1, 10000)
                if new_val in current:
                    continue
                current[idx] = new_val
                score, _ = count_distinct_primes(current)
                if score < best_score:
                    best_set = current[:]
                    best_score = score
                    found_improvement = True
                    no_improve = 0
                    break
                else:
                    current[idx] = old_val
            
            if not found_improvement:
                no_improve += 1
            else:
                no_improve = 0
        
        best_answer = best_set
    
    return best_answer
