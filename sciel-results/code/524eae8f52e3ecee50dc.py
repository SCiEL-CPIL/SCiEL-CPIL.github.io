import time
import math

def construct():
    start_time = time.time()
    deadline = 58.0

    limit = 1000000
    min_n = 10000

    # Primes to consider for smooth numbers
    # First try 11-smooth (primes 2,3,5,7,11)
    # If we find consecutive 11-smooth numbers, we return that n.
    # Otherwise, try 13-smooth (primes 2,3,5,7,11,13)
    
    def generate_smooth_numbers(primes, limit):
        # Returns a set of smooth numbers <= limit
        smooths = {1}
        # This simple DFS might be slow if not careful, but limit is 1e6.
        # Number of 11-smooth numbers is small.
        # Let's use a min-heap or just recursive expansion.
        
        # Better: iterate and filter? No, generation is better.
        # Use a list to store numbers, sort, and deduplicate.
        # Start with [1]
        nums = [1]
        # We can multiply by primes repeatedly.
        # Simple approach:
        # For each number in the set, multiply by each prime and add if <= limit.
        # Repeat until no new numbers added?
        # Or just generate all combinations.
        
        # Since the count is small, we can just generate all $2^a 3^b 5^c 7^d 11^e \le 10^6$.
        generated = set([1])
        
        # Max powers
        # 2^19 = 524288, 2^20 > 1e6
        # 3^12 = 531441, 3^13 > 1e6
        # 5^8 = 390625, 5^9 > 1e6
        # 7^7 = 823543, 7^8 > 1e6
        # 11^5 = 161051, 11^6 > 1e6
        
        # Nested loops are easy to write and fast enough for this count.
        p2 = 1
        while p2 <= limit:
            p3 = 1
            while p3 <= limit // p2:
                p5 = 1
                while p5 <= limit // (p2 * p3):
                    p7 = 1
                    while p7 <= limit // (p2 * p3 * p5):
                        p11 = 1
                        while p11 <= limit // (p2 * p3 * p5 * p7):
                            val = p2 * p3 * p5 * p7 * p11
                            if val >= min_n:
                                generated.add(val)
                            p11 *= 11
                        p7 *= 7
                    p5 *= 5
                p3 *= 3
            p2 *= 2
            
        return generated

    # 1. Check for 11-smooth consecutive pairs
    primes_11 = [2, 3, 5, 7, 11]
    # The generation function above is hardcoded for 5 primes.
    # Let's make it generic or just write two versions.
    # For 11-smooth, primes are 2,3,5,7,11.
    # For 13-smooth, primes are 2,3,5,7,11,13.
    
    # Let's refactor to use a helper that takes a list of primes.
    # But nested loops depth varies.
    # Given the small number of primes (5 or 6), hardcoding is safer and faster.
    
    # Generate 11-smooth numbers
    smooth_11 = set([1])
    p2 = 1
    while p2 <= limit:
        p3 = 1
        while p3 <= limit // p2:
            p5 = 1
            while p5 <= limit // (p2 * p3):
                p7 = 1
                while p7 <= limit // (p2 * p3 * p5):
                    p11 = 1
                    while p11 <= limit // (p2 * p3 * p5 * p7):
                        val = p2 * p3 * p5 * p7 * p11
                        if val >= min_n:
                            smooth_11.add(val)
                        p11 *= 11
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
        
    # Sort and check for consecutive
    sorted_11 = sorted(smooth_11)
    best_n_11 = -1
    for i in range(len(sorted_11) - 1):
        if sorted_11[i] + 1 == sorted_11[i+1]:
            # Found consecutive 11-smooth numbers
            # We want to minimize the LPF, which is 11.
            # Any such pair is better than LPF 13.
            # We can return the first one found, or the one with smallest n?
            # The score depends only on the LPF value (11), not n.
            # So any valid n works.
            best_n_11 = sorted_11[i]
            break

    if best_n_11 != -1:
        return best_n_11

    # 2. Check for 13-smooth consecutive pairs
    # Primes: 2, 3, 5, 7, 11, 13
    smooth_13 = set([1])
    p2 = 1
    while p2 <= limit:
        p3 = 1
        while p3 <= limit // p2:
            p5 = 1
            while p5 <= limit // (p2 * p3):
                p7 = 1
                while p7 <= limit // (p2 * p3 * p5):
                    p11 = 1
                    while p11 <= limit // (p2 * p3 * p5 * p7):
                        p13 = 1
                        while p13 <= limit // (p2 * p3 * p5 * p7 * p11):
                            val = p2 * p3 * p5 * p7 * p11 * p13
                            if val >= min_n:
                                smooth_13.add(val)
                            p13 *= 13
                        p11 *= 11
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2

    sorted_13 = sorted(smooth_13)
    best_n_13 = -1
    for i in range(len(sorted_13) - 1):
        if sorted_13[i] + 1 == sorted_13[i+1]:
            best_n_13 = sorted_13[i]
            break

    if best_n_13 != -1:
        return best_n_13

    # 3. Fallback: Sieve (if no 13-smooth pair found, which shouldn't happen if current program is correct)
    # But to be safe, run the sieve if time permits.
    if time.time() - start_time < deadline:
        max_val = limit + 1
        lpf = [0] * (max_val + 1)
        for p in range(2, max_val + 1):
            if lpf[p] == 0:
                for j in range(p, max_val + 1, p):
                    lpf[j] = p
        
        best_n = min_n
        min_score = lpf[min_n] if lpf[min_n] > lpf[min_n+1] else lpf[min_n+1]
        
        for n in range(min_n + 1, limit + 1):
            a = lpf[n]
            b = lpf[n + 1]
            score = a if a > b else b
            if score < min_score:
                min_score = score
                best_n = n
            if (n - min_n) % 50000 == 0:
                if time.time() - start_time > deadline:
                    break
        return best_n

    return min_n
