import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 80.0  # 80 seconds (10 second safety margin)
    
    n = 12
    maximum = 10000
    
    # Build SPF (smallest prime factor) sieve for factorization
    def build_spf(limit):
        spf = list(range(limit + 1))
        if limit >= 0:
            spf[0] = 0
        if limit >= 1:
            spf[1] = 1
        for i in range(2, int(limit**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf
    
    spf = build_spf(maximum * 2 + 1)
    
    def factorize(num):
        """Return set of prime factors of num using SPF."""
        if num <= 1:
            return set()
        factors = set()
        while num > 1:
            p = spf[num]
            factors.add(p)
            while num % p == 0:
                num //= p
        return factors
    
    def evaluate(arr):
        """Return number of distinct prime factors across all pairwise sums."""
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    # Initial solution
    best = list(range(1, n + 1))
    best_score = evaluate(best)
    
    # Local search with swap and perturb
    def local_search(arr, max_iters):
        arr = list(arr)
        score = evaluate(arr)
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                return arr, score
            improved = False
            # Try all swap moves
            random.shuffle(range(n))
            indices = list(range(n))
            for i in range(n):
                for j in range(i + 1, n):
                    if arr[i] == arr[j]:
                        continue
                    # Swap i and j
                    arr[i], arr[j] = arr[j], arr[i]
                    new_score = evaluate(arr)
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        arr[i], arr[j] = arr[j], arr[i]
                    if time.time() > deadline - 2.0:
                        return arr, score
            if not improved:
                # Perturb: replace a random element
                idx = random.randint(0, n - 1)
                old_val = arr[idx]
                arr[idx] = random.randint(1, maximum)
                # Ensure distinctness
                attempts = 0
                while arr.count(arr[idx]) > 1 and attempts < 100:
                    arr[idx] = random.randint(1, maximum)
                    attempts += 1
                new_score = evaluate(arr)
                if new_score > score:
                    arr[idx] = old_val  # revert
                    # Try more perturbations
                    for _ in range(10):
                        idx2 = random.randint(0, n - 1)
                        old_val2 = arr[idx2]
                        arr[idx2] = random.randint(1, maximum)
                        attempts = 0
                        while arr.count(arr[idx2]) > 1 and attempts < 100:
                            arr[idx2] = random.randint(1, maximum)
                            attempts += 1
                        new_score2 = evaluate(arr)
                        if new_score2 < score:
                            score = new_score2
                            break
                        elif new_score2 == score:
                            break
                        else:
                            arr[idx2] = old_val2
                else:
                    score = new_score
        return arr, score
    
    # Multi-start local search
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate random starting point
        arr = random.sample(range(1, maximum + 1), n)
        arr, score = local_search(arr, 200)
        
        if score < best_score:
            best_score = score
            best = arr
            # Continue local search from improved point
            arr, score = local_search(arr, 500)
            if score < best_score:
                best_score = score
                best = arr
    
    # Final check
    if time.time() > deadline - 1.0:
        return best
    
    # Try a structured approach: powers of 2 (all sums have few prime factors)
    # Actually, let's try multiples of a highly composite number
    # Try: all elements are multiples of 2 -> all sums are even -> prime factor 2 at minimum
    # But we want to minimize total distinct primes
    
    # Try elements that are all odd: sums are even
    # Try elements that are all congruent mod some number
    
    # Structured: use small numbers with specific properties
    # If all elements are powers of 2: sums are powers of 2 minus 1 or similar
    # Actually sums of two powers of 2: 2^a + 2^b = 2^min(a,b) * (1 + 2^|a-b|)
    # The factor (1 + 2^k) can have various prime factors
    
    # Better structured approach: all elements even
    even_arr = [2 * i for i in range(1, n + 1)]
    even_score = evaluate(even_arr)
    if even_score < best_score:
        best_score = even_score
        best = even_arr
        best, best_score = local_search(best, 500)
    
    # All elements multiples of 3
    mult3 = [3 * i for i in range(1, n + 1)]
    m3_score = evaluate(mult3)
    if m3_score < best_score:
        best_score = m3_score
        best = mult3
        best, best_score = local_search(best, 500)
    
    # Mixed structured: elements from {2^a * 3^b}
    candidates = []
    for a in range(0, 14):
        for b in range(0, 10):
            v = (2 ** a) * (3 ** b)
            if 1 <= v <= maximum:
                candidates.append(v)
    candidates = list(set(candidates))
    candidates.sort()
    
    if len(candidates) >= n:
        # Pick n from candidates
        for start in range(0, min(len(candidates) - n, 50)):
            if time.time() > deadline - 3.0:
                break
            arr = candidates[start:start + n]
            if len(set(arr)) < n:
                continue
            score = evaluate(arr)
            if score < best_score:
                best_score = score
                best = arr
                best, best_score = local_search(best, 300)
    
    return best
