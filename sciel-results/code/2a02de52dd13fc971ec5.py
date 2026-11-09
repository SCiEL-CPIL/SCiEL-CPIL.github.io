"""Rewritten program: Multi-start Greedy + Simulated Annealing for Sum-Free Subset."""
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
        
        # Precompute indices for faster lookup? 
        # Map value to index
        val_to_idx = {v: i for i, v in enumerate(vals)}

        # Helper: Check if adding index idx to S (set of indices) maintains sum-free property
        def can_add_idx(S_indices, idx):
            x = vals[idx]
            for a_idx in S_indices:
                a = vals[a_idx]
                # Check a + x = c
                c = a + x
                if c in val_set:
                    c_idx = val_to_idx[c]
                    if c_idx in S_indices:
                        return False
                # Check x - a = b (i.e., a + b = x)
                b = x - a
                if b in val_set:
                    b_idx = val_to_idx[b]
                    if b_idx in S_indices:
                        return False
            return True

        # Helper: Check if S is sum-free
        def is_sum_free_indices(S_indices):
            S_set = set(S_indices)
            for a_idx in S_indices:
                a = vals[a_idx]
                for b_idx in S_indices:
                    b = vals[b_idx]
                    c = a + b
                    if c in val_set:
                        c_idx = val_to_idx[c]
                        if c_idx in S_set:
                            return False
            return True

        def greedy_init(seed_offset):
            rng_g = random.Random(seed_for_rng + seed_offset)
            # Calculate degree
            degrees = [0] * n
            for i in range(n):
                x = vals[i]
                deg = 0
                for u in vals:
                    if (x + u) in val_set:
                        deg += 1
                    if (x - u) in val_set:
                        deg += 1
                degrees[i] = deg
            
            # Sort indices by degree, then random
            indices = list(range(n))
            indices.sort(key=lambda i: (degrees[i], rng_g.random()))
            
            S = set()
            for idx in indices:
                if can_add_idx(S, idx):
                    S.add(idx)
            return S

        # Run multiple greedy inits and pick the best
        best_init = set()
        best_init_size = 0
        for k in range(10):
            S = greedy_init(k)
            if len(S) > best_init_size:
                best_init_size = len(S)
                best_init = S

        cur = set(best_init)
        best = set(cur)
        best_size = len(cur)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        steps = 20000
        
        # Precompute lists for faster candidate selection?
        # We will just use random selection from list(set)
        
        for step in range(steps):
            if T < 1e-6:
                break
                
            r = rng.random()
            
            if r < 0.4:
                # Try to Add
                candidates = [i for i in range(n) if i not in cur]
                if candidates:
                    x_idx = rng.choice(candidates)
                    if can_add_idx(cur, x_idx):
                        cur.add(x_idx)
                        if len(cur) > best_size:
                            best = set(cur)
                            best_size = len(cur)
                        
            elif r < 0.8:
                # Try to Swap
                if cur:
                    x_idx = rng.choice(list(cur))
                    candidates = [i for i in range(n) if i not in cur]
                    if candidates:
                        y_idx = rng.choice(candidates)
                        # Check if (cur - {x}) + {y} is sum-free
                        new_cur = cur - {x_idx}
                        if can_add_idx(new_cur, y_idx):
                            new_cur.add(y_idx)
                            cur = new_cur
                            # Size is same, no best update needed for size, but state changes
                
            else:
                # Try to Remove
                if cur:
                    x_idx = rng.choice(list(cur))
                    # Accept removal with probability exp(1/T)
                    # Delta E = 1 (size decreases by 1)
                    p = math.exp(1.0 / T) if T > 0 else 0
                    if rng.random() < p:
                        cur.remove(x_idx)
                        
            T *= cooling
            
        # Final Verification
        if not is_sum_free_indices(best):
             # Fallback
             best = best_init
             
        # Convert indices back to values
        return sorted([vals[i] for i in best])

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        sub = solve_one(vals, seed)
        results.append(sub)
    return results
# EVOLVE-BLOCK-END
