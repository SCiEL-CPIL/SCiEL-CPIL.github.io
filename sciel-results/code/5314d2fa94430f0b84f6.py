"""Rewritten program: Greedy + Simulated Annealing for Sum-Free Subset."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def solve_one(vals, seed_for_rng):
        n = len(vals)
        if n == 0:
            return []
        rng = random.Random(seed_for_rng * 1000003 + 7)
        val_set = set(vals)

        # Helper: Check if adding x to S (set) maintains sum-free property
        def can_add(S, x):
            # Check for violations involving x
            # 1. x + a = c (c in S) -> x = c - a
            # 2. a + b = x (a, b in S) -> b = x - a
            # 3. x + x = c (c in S) -> 2x = c
            for a in S:
                if (x + a) in S:
                    return False
                if (x - a) in S:
                    return False
            if (2 * x) in S:
                return False
            return True

        # Helper: Check if S is sum-free
        def is_sum_free(S):
            Sset = set(S)
            for a in S:
                for b in S:
                    if (a + b) in Sset:
                        return False
            return True

        # 1. Greedy Initialization
        # Calculate "conflict degree" for each element
        degrees = {}
        for x in vals:
            deg = 0
            for u in vals:
                if (x + u) in val_set:
                    deg += 1
                if (x - u) in val_set:
                    deg += 1
            degrees[x] = deg
        
        # Sort by degree, then randomly shuffle ties
        sorted_vals = sorted(vals, key=lambda x: (degrees[x], rng.random()))
        
        cur = set()
        for x in sorted_vals:
            if can_add(cur, x):
                cur.add(x)
        
        best = set(cur)
        best_size = len(cur)
        
        # 2. Simulated Annealing
        T = 5.0
        cooling = 0.9995
        steps = 50000
        
        for step in range(steps):
            if T < 0.01:
                break
                
            r = rng.random()
            
            if r < 0.4:
                # Try to Add
                candidates = [v for v in vals if v not in cur]
                if candidates:
                    x = rng.choice(candidates)
                    if can_add(cur, x):
                        cur.add(x)
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                        
            elif r < 0.8:
                # Try to Swap
                if cur:
                    x = rng.choice(list(cur))
                    candidates = [v for v in vals if v not in cur]
                    if candidates:
                        y = rng.choice(candidates)
                        # Check if (cur - {x}) + {y} is sum-free
                        new_cur = cur - {x}
                        if can_add(new_cur, y):
                            new_cur.add(y)
                            # Accept swap to explore structure
                            cur = new_cur
                            # Update best only if size increases (it doesn't here)
                            # But better structure might allow future adds.
                        
            else:
                # Try to Remove
                if cur:
                    x = rng.choice(list(cur))
                    # Accept removal with probability exp(-1/T)
                    if rng.random() < math.exp(-1.0 / T):
                        cur.remove(x)
                        
            T *= cooling
            
        # Final Verification
        if not is_sum_free(best):
             best = set(cur)
             
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
