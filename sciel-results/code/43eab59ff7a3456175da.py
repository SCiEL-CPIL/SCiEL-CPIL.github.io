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

        # Helper: Check if adding x to S maintains sum-free property
        def can_add(S, x):
            Sset = set(S)
            for a in S:
                if (a + x) in Sset:
                    return False
                if (x - a) in Sset:
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
        # deg(x) = number of pairs (u,v) in vals such that u+v=x or x+u=v or x+v=u
        # We use this to order elements: low degree first (safer to include)
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
        
        S = []
        Sset = set()
        for x in sorted_vals:
            if can_add(S, x):
                S.append(x)
                Sset.add(x)
        
        cur = set(S)
        best = set(cur)
        best_size = len(cur)
        
        # 2. Simulated Annealing
        # We maintain a feasible state (sum-free).
        # Moves:
        # - Add: If valid, always accept (increases size).
        # - Remove: Accept with probability P_remove to escape local optima.
        # - Swap: Remove x, add y. If valid, accept (size same, but structure changes).
        
        T = 1.0
        cooling = 0.999
        steps = 50000
        p_remove = 0.1
        
        for step in range(steps):
            if T < 1e-6:
                break
                
            r = rng.random()
            
            if r < 0.5:
                # Try to Add
                # Pick random element not in cur
                candidates = [v for v in vals if v not in cur]
                if candidates:
                    x = rng.choice(candidates)
                    if can_add(cur, x):
                        cur.add(x)
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                            # Reset T or boost? No, just keep going.
                        
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
                            # Accept swap (size is same, but might enable better adds)
                            cur = new_cur
                            # Update best? Size is same, so no improvement in size, 
                            # but might be a better "shape". We only track size for best.
                            # However, if we find a better shape, we might add more later.
                            # We don't update 'best' here unless size increases.
                        
            else:
                # Try to Remove
                if cur:
                    x = rng.choice(list(cur))
                    # Accept removal with probability p_remove
                    if rng.random() < p_remove:
                        cur.remove(x)
                        
            T *= cooling
            # Adjust p_remove? Maybe increase as T decreases to encourage exploration?
            # Or keep constant.
            
        # Final Verification
        if not is_sum_free(best):
             # Fallback to greedy S if best is somehow invalid (shouldn't happen)
             best = set(S)
             
        return sorted(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
