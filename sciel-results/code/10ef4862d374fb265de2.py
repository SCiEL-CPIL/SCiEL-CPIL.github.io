"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 80  # 10s safety margin before 90s kill
    
    # Precompute gcd table for all pairs in [1, 128]
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def has_bad_triple(selected, x):
        """Check if adding x creates any triple with all-equal pairwise gcds."""
        n = len(selected)
        for i in range(n):
            a = selected[i]
            ga = gcd_table[a][x]
            for j in range(i + 1, n):
                b = selected[j]
                gb = gcd_table[b][x]
                gab = gcd_table[a][b]
                if ga == gb == gab:
                    return True
        return False
    
    def prime_sig(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    def greedy(order):
        selected = []
        for x in order:
            if not has_bad_triple(selected, x):
                selected.append(x)
        return selected
    
    def local_search(selected, max_swaps=200):
        """Try to improve the set by swapping elements."""
        selected_set = set(selected)
        nums = list(range(1, 129))
        best = list(selected)
        best_size = len(best)
        
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i, a in enumerate(best):
                # Try removing a and adding some b not in best
                for b in nums:
                    if b in selected_set:
                        continue
                    # Try swap: remove a, add b
                    new_set = [x for x in best if x != a]
                    if not has_bad_triple(new_set, b):
                        new_set.append(b)
                        if len(new_set) > best_size:
                            best = new_set
                            best_size = len(best)
                            selected_set = set(best)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return best
    
    nums = list(range(1, 129))
    
    # Try multiple orderings, keep the best result
    best = []
    
    orderings = []
    
    # Ordering 1: descending by number of distinct prime factors
    orderings.append(sorted(nums, key=lambda n: (-len(set(prime_sig(n))), n)))
    
    # Ordering 2: ascending by value (simple)
    orderings.append(sorted(nums))
    
    # Ordering 3: descending by value
    orderings.append(sorted(nums, reverse=True))
    
    # Ordering 4: by largest prime factor descending
    orderings.append(sorted(nums, key=lambda n: (-max(prime_sig(n), default=1), n)))
    
    # Ordering 5: powers of 2 first, then others
    powers2 = [2**i for i in range(7) if 2**i <= 128]
    others = [n for n in nums if n not in set(powers2)]
    orderings.append(powers2 + others)
    
    # Ordering 6: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    orderings.append(primes + composites)
    
    # Ordering 7: by number of divisors ascending (fewer divisors first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    orderings.append(sorted(nums, key=lambda n: (num_divisors(n), n)))
    
    # Ordering 8: random-ish deterministic shuffle based on prime signature
    import random
    rng = random.Random(42)
    shuffled = nums[:]
    rng.shuffle(shuffled)
    orderings.append(shuffled)
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = greedy(order)
        # Apply local search to improve
        result = local_search(result, max_swaps=50)
        if len(result) > len(best):
            best = result
            if time.time() > deadline - 10:
                break
    
    # Ensure nonempty
    if not best:
        best = [1, 2]
    
    return best
# EVOLVE-BLOCK-END
