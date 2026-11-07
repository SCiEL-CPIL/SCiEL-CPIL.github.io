import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0

    def get_products(subset):
        prods = {1}
        for x in subset:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            prods.update(new_prods)
        return prods

    def check_valid(subset):
        prods = {1}
        for x in subset:
            new_prods = set()
            for p in prods:
                np = p * x
                if np in prods:
                    return False
                new_prods.add(np)
            prods.update(new_prods)
        return True

    def can_add(current_set, candidate):
        if candidate in current_set:
            return False
        prods = get_products(current_set)
        for p in prods:
            if p * candidate in prods:
                return False
        return True

    # Start with primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    current = list(primes)

    # Try adding composites sorted by simplicity
    remaining = [x for x in range(2, 41) if x not in current]

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    remaining.sort(key=num_prime_factors)

    for c in remaining:
        if time.time() - start_time > DEADLINE:
            break
        if can_add(current, c):
            current.append(c)

    # Try swaps to potentially improve
    improved = True
    while improved and (time.time() - start_time < DEADLINE - 5):
        improved = False
        for i, elem in enumerate(current):
            for c in range(2, 41):
                if c in current:
                    continue
                new_set = current[:i] + current[i+1:] + [c]
                if can_add(new_set, c):
                    current = new_set
                    improved = True
                    break
            if improved:
                break

    if not check_valid(current):
        current = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return [int(x) for x in current]
