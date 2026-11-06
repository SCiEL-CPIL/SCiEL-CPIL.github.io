import math
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 140

    max_val = 10**18

    def tau_perp(n):
        if n < 2:
            return 0
        divisors = [1]
        temp = n
        i = 2
        while i * i <= temp:
            if temp % i == 0:
                exp = 0
                while temp % i == 0:
                    temp //= i
                    exp += 1
                new_divs = []
                for d in divisors:
                    power = 1
                    for e in range(exp + 1):
                        new_divs.append(d * power)
                        power *= i
                divisors = new_divs
            i += 1
        if temp > 1:
            new_divs = []
            for d in divisors:
                new_divs.append(d)
                new_divs.append(d * temp)
            divisors = new_divs

        divisors.sort()

        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    best_n = 2
    best_score = -1

    def try_candidate(n):
        nonlocal best_n, best_score
        if n < 2 or n > max_val:
            return
        if time.time() - start_time > TIME_LIMIT - 3:
            return
        s = tau_perp(n)
        if s > best_score:
            best_score = s
            best_n = n

    # Squarefree products of first k primes
    prod = 1
    for k in range(1, len(primes)):
        prod *= primes[k-1]
        if prod > max_val:
            break
        try_candidate(prod)

    # DFS with non-increasing exponents
    def dfs(idx, max_exp, current_n):
        nonlocal best_n, best_score
        if time.time() - start_time > TIME_LIMIT - 3:
            return
        if current_n >= 2:
            s = tau_perp(current_n)
            if s > best_score:
                best_score = s
                best_n = current_n

        if idx >= len(primes):
            return

        p = primes[idx]
        for exp in range(max_exp, -1, -1):
            if exp > 0:
                new_n = current_n * (p ** exp)
            else:
                new_n = current_n
            if new_n > max_val:
                continue
            dfs(idx + 1, exp, new_n)

    dfs(0, 8, 1)

    return best_n
