import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = 1.0
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        size_s = len(sums)
        size_d = len(diffs)
        
        if size_s == 0:
            return 0.0
            
        ratio = size_d / size_s
        if ratio <= 1.0:
            return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def greedy_prime_init(max_prime_limit=500):
        """Build a set greedily by adding primes that maximize incremental score."""
        U = [0, 1]
        current_score = calc_score(U)
        
        primes = [p for p in range(2, max_prime_limit + 1) if is_prime(p)]
        
        for p in primes:
            if time.time() > deadline:
                break
            candidate = sorted(U + [p])
            new_score = calc_score(candidate)
            if new_score > current_score:
                U = candidate
                current_score = new_score
        
        return U, current_score

    def local_search(U, max_iterations=5000, temperature=1.0):
        """Local search: try moving elements to nearby positions."""
        U = sorted(set(U))
        current_score = calc_score(U)
        max_u = max(U)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try moving each element to a nearby position
            for i in range(1, len(U)):  # Keep 0 fixed
                original = U[i]
                # Try a few nearby positions
                for delta in [-3, -2, -1, 1, 2, 3, 5, 7]:
                    new_val = original + delta
                    if new_val < 1 or new_val > 2 * max_u + 10:
                        continue
                    if new_val in U:
                        continue
                    
                    new_U = sorted(U[:i] + [new_val] + U[i+1:])
                    new_score = calc_score(new_U)
                    
                    if new_score > current_score + 1e-12:
                        U = new_U
                        current_score = new_score
                        max_u = max(U)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return U, current_score

    # Phase 1: Greedy prime initialization
    greedy_set, greedy_score = greedy_prime_init(500)
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = greedy_set
    
    # Phase 2: Local search refinement on greedy set
    refined_set, refined_score = local_search(greedy_set, max_iterations=5000)
    if refined_score > best_score:
        best_score = refined_score
        best_set = refined_set

    # Phase 3: Heuristic 1: Geometric Progressions
    for r in range(2, 10):
        for n in range(3, 8):
            U = [r**i for i in range(n)]
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline:
                return best_set

    # Phase 4: Heuristic 2: Random sets with small max
    import random
    random.seed(42)
    
    for n in range(3, 12):
        for max_val in range(n, 50):
            for _ in range(1000):
                if time.time() > deadline:
                    return best_set
                candidates = list(range(1, max_val + 1))
                if len(candidates) < n - 1:
                    continue
                sample = random.sample(candidates, n - 1)
                U = sorted([0] + sample)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Phase 5: Exhaustive search for very small sets
    for n in range(3, 6):
        for max_val in range(n, 20):
            range_vals = list(range(1, max_val + 1))
            if len(range_vals) < n - 1:
                continue
            for comb in itertools.combinations(range_vals, n - 1):
                if time.time() > deadline:
                    return best_set
                U = sorted([0] + list(comb))
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Phase 6: Final local search polish on the best set found
    if time.time() < deadline - 5:
        polished_set, polished_score = local_search(best_set, max_iterations=3000)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set

    return best_set
