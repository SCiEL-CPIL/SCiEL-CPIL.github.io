import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 40
    numbers = list(range(2, N + 1))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    d = len(primes)

    vectors = {}
    for n in numbers:
        v = [0] * d
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        vectors[n] = tuple(v)

    def subset_sums_distinct(vecs):
        k = len(vecs)
        cur = {tuple([0] * d)}
        for v in vecs:
            new = set()
            for s in cur:
                new.add(tuple(s[j] + v[j] for j in range(d)))
            cur = cur | new
            if len(cur) != (1 << (k if False else 0)):  # placeholder
                pass
        return len(cur) == (1 << k)

    # The 16-set: superincreasing prime powers
    candidate16 = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Verify candidate16
    if subset_sums_distinct([vectors[n] for n in candidate16]):
        best = candidate16
    else:
        best = list(primes)

    # Light search to try to exceed
    import random
    random.seed(7)
    base = list(best)
    remaining = [n for n in numbers if n not in set(base)]

    def try_build(order):
        chosen = []
        cur_sums = {tuple([0] * d)}
        for n in order:
            v = vectors[n]
            new_sums = set()
            for s in cur_sums:
                new_sums.add(tuple(s[j] + v[j] for j in range(d)))
            if len(new_sums) == len(cur_sums):
                cur_sums = cur_sums | new_sums
                chosen.append(n)
        return chosen

    for _ in range(3000):
        if time.time() > deadline - 5.0:
            break
        shuffled = list(remaining)
        random.shuffle(shuffled)
        built = try_build(base + shuffled)
        if len(built) > len(best):
            best = built
            base = list(best)
            remaining = [n for n in numbers if n not in set(base)]

    if not subset_sums_distinct([vectors[n] for n in best]):
        best = list(primes)

    return [int(n) for n in best]
