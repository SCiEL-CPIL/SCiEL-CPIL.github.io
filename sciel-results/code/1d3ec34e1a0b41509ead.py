"""Few prime factors across pairwise sums - local search with precomputed factors."""
import time
import random

# Precompute smallest prime factor for fast factorization
_SPF = [0] * 20001
for i in range(2, 20001):
    if _SPF[i] == 0:
        for j in range(i, 20001, i):
            if _SPF[j] == 0:
                _SPF[j] = i

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    while n > 1:
        p = _SPF[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors for all possible sums (2..20000)
_PF = [set() for _ in range(20001)]
for s in range(2, 20001):
    _PF[s] = _prime_factors(s)

def _eval(arr):
    """Count distinct primes dividing sums of all distinct pairs."""
    primes = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            primes |= _PF[ai + arr[j]]
    return len(primes)

def _try_move(arr, i, new_val, current_primes, n=12):
    """Try replacing arr[i] with new_val; return (new_count, new_primes) or (None, None)."""
    if new_val in arr and arr.index(new_val) != i:
        return None, None
    old = arr[i]
    if new_val == old:
        return current_primes, current_primes

    # Compute primes contributed by pairs involving index i
    # Remove old contributions, add new contributions
    # We need to be careful: a prime might be contributed by multiple pairs
    # Use a counter approach
    from collections import Counter
    counts = Counter()
    for j in range(n):
        if j == i:
            continue
        for p in _PF[old + arr[j]]:
            counts[p] += 1

    new_counts = Counter()
    for j in range(n):
        if j == i:
            continue
        for p in _PF[new_val + arr[j]]:
            new_counts[p] += 1

    # Primes that were only in old pairs (count 1) and not in new pairs
    # Primes that are in new pairs but not old
    # Net effect on the global set:
    # A prime is removed if it was in old (count>=1) and not in new (count==0)
    # A prime is added if it's in new (count>=1) and not in old (count==0)
    # But we also need to consider primes from pairs NOT involving i

    # Simpler: just compute the full set difference
    # Primes from pairs not involving i stay the same
    # Primes from pairs involving i change

    # Old primes from pairs involving i:
    old_pair_primes = set()
    for j in range(n):
        if j == i:
            continue
        old_pair_primes |= _PF[old + arr[j]]

    new_pair_primes = set()
    for j in range(n):
        if j == i:
            continue
        new_pair_primes |= _PF[new_val + arr[j]]

    # Primes not involving i:
    other_primes = set()
    for j in range(n):
        for k in range(j + 1, n):
            if j == i or k == i:
                continue
            other_primes |= _PF[arr[j] + arr[k]]

    new_total = other_primes | new_pair_primes
    return len(new_total), new_total

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    random.seed(42)

    n = 12
    max_val = 10000

    # Start with a reasonable initial solution
    # Use multiples of a small number to reduce prime diversity
    # e.g., all multiples of 30 = 2*3*5
    # Sums of two multiples of 30 are multiples of 30, so prime factors subset of {2,3,5}
    # But we need 12 distinct values <= 10000
    # 30, 60, 90, ..., 360 -> 12 values, all sums are multiples of 30
    # Prime factors of sums: only 2, 3, 5
    best_arr = list(range(30, 390, 30))  # 30,60,...,360
    best_score = _eval(best_arr)

    # Try other bases
    for base in [6, 10, 12, 14, 15, 20, 21, 22, 24, 26, 28, 30, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 66, 70, 72, 75, 78, 80, 84, 90, 96, 100, 105, 108, 110, 112, 120, 126, 130, 132, 135, 140, 144, 150, 156, 160, 168, 175, 180, 182, 189, 190, 192, 195, 196, 198, 200, 210, 216, 220, 224, 225, 231, 234, 240, 245, 250, 252, 255, 256, 258, 260, 264, 270, 273, 280, 285, 288, 290, 294, 297, 300, 306, 308, 312, 315, 320, 322, 324, 325, 330, 336, 340, 342, 345, 348, 350, 351, 357, 360, 364, 369, 370, 372, 375, 378, 380, 384, 385, 390, 392, 396, 399, 400]:
        if base * n > max_val:
            continue
        arr = list(range(base, base * n + 1, base))
        if len(arr) == n and all(1 <= x <= max_val for x in arr):
            s = _eval(arr)
            if s < best_score:
                best_score = s
                best_arr = arr[:]

    # Local search: try replacing elements with nearby values
    arr = best_arr[:]
    current_score = best_score

    # Precompute for speed
    def fast_eval(a):
        primes = set()
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                primes |= _PF[ai + a[j]]
        return len(primes)

    # Local search with random restarts
    steps = 0
    while time.time() < deadline:
        # Try random moves
        improved = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            i = random.randint(0, n - 1)
            old_val = arr[i]

            # Try several candidate replacements
            candidates = []
            # Nearby values
            for delta in range(-20, 21):
                nv = old_val + delta
                if 1 <= nv <= max_val and nv != old_val:
                    candidates.append(nv)
            # Multiples of small numbers
            for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 48, 50, 54, 56, 60, 63, 66, 70, 72, 80, 84, 90, 96, 100, 105, 108, 112, 120, 126, 140, 144, 150, 160, 168, 180, 192, 200, 210, 220, 224, 240, 250, 252, 280, 288, 300, 336, 360, 420, 480, 500, 540, 560, 600, 630, 700, 720, 800, 840, 900, 960, 1000, 1050, 1080, 1120, 1200, 1260, 1400, 1440, 1500, 1600, 1680, 1800, 1920, 2000, 2100, 2200, 2400, 2500, 2520, 2800, 3000, 3360, 3600, 4200, 4800, 5000, 5400, 5600, 6000, 6300, 7000, 7200, 8000, 8400, 9000, 9600, 10000]:
                for k in range(1, max(1, max_val // m + 1)):
                    nv = m * k
                    if 1 <= nv <= max_val and nv != old_val:
                        candidates.append(nv)
                        if len(candidates) > 500:
                            break
                if len(candidates) > 500:
                    break

            random.shuffle(candidates)
            for nv in candidates[:100]:
                if nv in arr and arr.index(nv) != i:
                    continue
                arr[i] = nv
                new_score = fast_eval(arr)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    arr[i] = old_val

            steps += 1

        # If no improvement, try a random restart
        if not improved:
            # Generate a new random solution biased toward multiples
            base = random.choice([6, 10, 12, 14, 15, 20, 21, 22, 24, 26, 28, 30, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60, 63, 66, 70, 72, 75, 78, 80, 84, 90, 96, 100, 105, 108, 110, 112, 120, 126, 130, 132, 135, 140, 144, 150, 156, 160, 168, 175, 180, 182, 189, 190, 192, 195, 196, 198, 200, 210, 216, 220, 224, 225, 231, 234, 240, 245, 250, 252, 255, 256, 258, 260, 264, 270, 273, 280, 285, 288, 290, 294, 297, 300, 306, 308, 312, 315, 320, 322, 324, 325, 330, 336, 340, 342, 345, 348, 350, 351, 357, 360, 364, 369, 370, 372, 375, 378, 380, 384, 385, 390, 392, 396, 399, 400])
            if base * n <= max_val:
                arr = list(range(base, base * n + 1, base))
                current_score = fast_eval(arr)
                if current_score < best_score:
                    best_score = current_score
                    best_arr = arr[:]
            else:
                # Random distinct values
                arr = random.sample(range(1, max_val + 1), n)
                current_score = fast_eval(arr)
                if current_score < best_score:
                    best_score = current_score
                    best_arr = arr[:]

    # Final check
    final_score = _eval(best_arr)
    if final_score != best_score:
        best_score = final_score

    return best_arr
