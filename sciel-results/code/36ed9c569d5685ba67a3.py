import math
import time
from itertools import combinations

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Precompute: for each number, its prime factors
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Precompute prime factors for all numbers 1-256 (max sum is 128+128=256)
    pf_cache = {i: prime_factors(i) for i in range(1, 257)}

    # Fast evaluation: check pairwise coprimality via prime-factor overlap
    def fast_evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) < 2:
            return len(sums)
        slist = list(sums)
        # Check pairwise coprimality using precomputed prime factors
        for i in range(len(slist)):
            fi = pf_cache[slist[i]]
            for j in range(i + 1, len(slist)):
                if fi & pf_cache[slist[j]]:
                    return 0
        return len(sums)

    # Precompute a "quality score" for each number 1..128:
    # prefer numbers whose sums with small numbers tend to be coprime-friendly
    # Simple heuristic: fewer distinct prime factors is better
    num_quality = {}
    for v in range(1, 129):
        num_quality[v] = -len(pf_cache[v]) if v > 1 else 0

    import random
    random.seed(42)

    def evaluate(A, B):
        return fast_evaluate(A, B)

    # Local search with neutral-move acceptance to escape plateaus
    def local_search(A, B, max_iter=8000):
        A = list(A)
        B = list(B)
        curr_score = evaluate(A, B)
        best_A = list(A)
        best_B = list(B)
        best_score = curr_score
        stall_count = 0
        max_stall = 30

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            # Try all single-element replacements in A and B
            for pos in range(3):
                if time.time() > deadline:
                    break
                # Try replacing A[pos]
                for new_val in range(1, 129):
                    if new_val in A:
                        continue
                    new_A = list(A)
                    new_A[pos] = new_val
                    new_score = evaluate(new_A, B)
                    if new_score > curr_score:
                        A = new_A
                        curr_score = new_score
                        improved = True
                        stall_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break
                # Try replacing B[pos]
                for new_val in range(1, 129):
                    if new_val in B:
                        continue
                    new_B = list(B)
                    new_B[pos] = new_val
                    new_score = evaluate(A, new_B)
                    if new_score > curr_score:
                        B = new_B
                        curr_score = new_score
                        improved = True
                        stall_count = 0
                        if curr_score > best_score:
                            best_score = curr_score
                            best_A = list(A)
                            best_B = list(B)
                        if curr_score == 9:
                            return best_A, best_B, best_score
                        break
                if improved:
                    break

            if not improved:
                # Stuck: try a neutral move (same score) to escape plateau
                stall_count += 1
                if stall_count > max_stall:
                    break
                # Random neutral replacement
                if random.random() < 0.5:
                    pos = random.randint(0, 2)
                    candidates = [v for v in range(1, 129) if v not in A]
                    new_val = random.choice(candidates)
                    new_A = list(A)
                    new_A[pos] = new_val
                    if evaluate(new_A, B) == curr_score:
                        A = new_A
                else:
                    pos = random.randint(0, 2)
                    candidates = [v for v in range(1, 129) if v not in B]
                    new_val = random.choice(candidates)
                    new_B = list(B)
                    new_B[pos] = new_val
                    if evaluate(A, new_B) == curr_score:
                        B = new_B

        return best_A, best_B, best_score

    # Initial best from a known small construction
    best_A = [1, 2, 3]
    best_B = [1, 2, 4]
    best_cardinality = evaluate(best_A, best_B)

    # Multiple random starts with quality-biased sampling
    for start in range(2000):
        if time.time() > deadline:
            break

        # Bias toward numbers with fewer prime factors and smaller magnitude
        # Weight: inverse of (value * num_prime_factors)
        weights = []
        for v in range(1, 129):
            w = 1.0 / (v * max(1, len(pf_cache[v])))
            weights.append(w)
        total_w = sum(weights)
        # Sample 3 distinct from A
        A = []
        pool = list(range(1, 129))
        wts = list(weights)
        for _ in range(3):
            r = random.random() * sum(wts)
            cum = 0
            idx = 0
            for i, w in enumerate(wts):
                cum += w
                if cum >= r:
                    idx = i
                    break
            A.append(pool[idx])
            pool.pop(idx)
            wts.pop(idx)

        B = []
        pool2 = list(range(1, 129))
        wts2 = list(weights)
        for _ in range(3):
            r = random.random() * sum(wts2)
            cum = 0
            idx = 0
            for i, w in enumerate(wts2):
                cum += w
                if cum >= r:
                    idx = i
                    break
            B.append(pool2[idx])
            pool2.pop(idx)
            wts2.pop(idx)

        if len(set(A)) < 3 or len(set(B)) < 3:
            continue

        res_A, res_B, res_score = local_search(A, B, max_iter=3000)

        if res_score > best_cardinality:
            best_A = res_A
            best_B = res_B
            best_cardinality = res_score
            if res_score == 9:
                return [best_A, best_B]

    return [best_A, best_B]
