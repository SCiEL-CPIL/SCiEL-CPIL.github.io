"""Sum-free subset construction with greedy + local search."""
import time

def construct():
    from random import Random

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if (a + b) in s:
                    return False
        return True

    def greedy_sum_free(vals, rng):
        """Greedy: sort by absolute value descending, add if doesn't create violation."""
        s = set()
        for x in sorted(vals, key=abs, reverse=True):
            # Check if adding x creates a violation
            ok = True
            for a in s:
                if (x + a) in s:
                    ok = False
                    break
                if (2 * x) in s:
                    ok = False
                    break
            if ok:
                s.add(x)
        return list(s)

    def local_search_improve(subset, rng, time_limit):
        """Improve sum-free subset via 1-swap moves."""
        best = list(subset)
        deadline = time.time() + time_limit
        
        def can_swap(S, x, y):
            """Check if (S - {x}) ∪ {y} is sum-free, given S is sum-free."""
            # (A) 2y not in S or 2y == x
            if 2 * y in S and 2 * y != x:
                return False
            # (C) For all a in S\{x}: y+a not in S or y+a == x
            for a in S:
                if a == x:
                    continue
                if (y + a) in S and (y + a) != x:
                    return False
            # (B) No a,b in S\{x} with a+b == y
            for a in S:
                if a == x:
                    continue
                b = y - a
                if b in S and b != x:
                    return False
            return True

        while time.time() < deadline:
            improved = False
            S = set(best)
            # Try swaps: remove one, add one
            for x in best:
                # Try a few random y candidates
                for _ in range(30):
                    y = rng.randint(-10000, 10000)
                    if y == 0 or y in S:
                        continue
                    if can_swap(S, x, y):
                        best.remove(x)
                        best.append(y)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    results = []
    start = time.time()
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        # Start with parity-based
        base = [x for x in vals if x % 2 != 0]
        # Try greedy too, take the larger
        greedy = greedy_sum_free(vals, Random(seed))
        if len(greedy) > len(base):
            base = greedy
        # Local search with time budget
        remaining = max(0, 100 - (time.time() - start))
        time_per = remaining / (3 - len(results)) if len(results) < 3 else 0
        improved = local_search_improve(base, Random(seed + 1), min(time_per, 30))
        if not is_sum_free(improved):
            improved = base
        results.append(improved)
    
    return results
