import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # L is 30030
    
    best_span = float('inf')
    best_values = None
    
    # Deadline guard: 90s limit, we expect this to take < 1s
    start_time = time.time()
    
    for W in range(L):
        # Calculate the unique multiple of d_i in [W, W+L)
        # x_i = W + ((-W) % d_i)
        vals = [W + ((-W) % d) for d in divisors]
        
        # Check distinctness
        if len(set(vals)) == n:
            span = max(vals) - min(vals)
            if span < best_span:
                best_span = span
                best_values = vals[:]
        else:
            # Handle collisions
            # We try to resolve by moving elements to neighbors
            # Since n=10 is small, we can try a few adjustments
            
            # Simple strategy: Try moving the element with the "most freedom" or just iterate through elements
            # and try shifting them to -d, +d if it resolves collision without increasing span too much.
            # Actually, we can just try all combinations of shifts {-1, 0, 1} for the colliding elements?
            # That might be expensive if many collide.
            
            # Heuristic: The natural values are the best. Any shift increases span by at least min(d_i).
            # So we can just try to find *any* distinct configuration close to vals.
            
            # Let's try a simple local improvement:
            # Sort by value? No, indices matter for divisibility.
            
            # Try shifting each element i by -d_i or +d_i if it helps
            # We'll do a small exhaustive search over "shifts" for the elements involved in collisions.
            
            # Identify collision groups
            from collections import defaultdict
            val_to_indices = defaultdict(list)
            for i, v in enumerate(vals):
                val_to_indices[v].append(i)
                
            colliding_indices = []
            for v, indices in val_to_indices.items():
                if len(indices) > 1:
                    colliding_indices.extend(indices)
            
            if not colliding_indices:
                continue # Should not happen if distinct check failed
            
            # Try all combinations of shifts {-1, 0, 1} for colliding indices?
            # If 2 collide, 3^2 = 9. If 3 collide, 3^3 = 27.
            # Max collisions is small.
            
            # Generate candidate sets
            # We only change the colliding ones. The non-colliding ones stay put.
            # But changing a colliding one might create a new collision with a non-colliding one.
            
            # Let's just try shifting each colliding index to -1, 0, 1 relative to its "natural" slot in the sequence of multiples?
            # No, "natural" is the one in [W, W+L).
            # Neighbors are vals[i] - d_i and vals[i] + d_i.
            
            # We'll try to find the min span among distinct sets formed by choosing for each colliding index i
            # a value from {vals[i] - d_i, vals[i], vals[i] + d_i}
            
            # This is a small CSP.
            # Given the rarity of this branch, we can be slightly more expensive.
            
            # However, to keep it fast, we'll just try a few specific moves:
            # 1. Move the last one in the collision group to next multiple.
            # 2. Move the first one to prev multiple.
            
            # Let's do a proper small search for this W
            # k = len(colliding_indices)
            # If k > 5, we might have a problem, but unlikely.
            
            # We'll use itertools.product for shifts
            # But we need to ensure the new values are distinct from ALL other values (including non-colliding).
            
            # Fixed values
            fixed_vals = {}
            for i in range(n):
                if i not in colliding_indices:
                    fixed_vals[i] = vals[i]
            
            # Variable values
            var_indices = colliding_indices
            var_options = {i: [vals[i] - divisors[i], vals[i], vals[i] + divisors[i]] for i in var_indices}
            
            # Iterate over products
            # To avoid explosion, we limit the depth or just hope k is small.
            # If k is large, we can just skip or use a greedy approach.
            # Given L=30030, this branch is rare.
            
            if len(var_indices) > 6:
                # Fallback: just shift the last one
                last_idx = var_indices[-1]
                new_val = vals[last_idx] + divisors[last_idx]
                test_vals = vals[:]
                test_vals[last_idx] = new_val
                if len(set(test_vals)) == n:
                    span = max(test_vals) - min(test_vals)
                    if span < best_span:
                        best_span = span
                        best_values = test_vals[:]
                continue
            
            for shifts in __import__('itertools').product([-1, 0, 1], repeat=len(var_indices)):
                test_vals = vals[:]
                valid = True
                for idx, s in zip(var_indices, shifts):
                    if s == -1:
                        test_vals[idx] = vals[idx] - divisors[idx]
                    elif s == 1:
                        test_vals[idx] = vals[idx] + divisors[idx]
                    # else 0: no change
                
                if len(set(test_vals)) == n:
                    span = max(test_vals) - min(test_vals)
                    if span < best_span:
                        best_span = span
                        best_values = test_vals[:]
        
        # Time check
        if time.time() - start_time > 80:
            break

    if best_values is None:
        # Fallback to seed behavior (should not happen)
        return divisors[:]
        
    # Ensure types are int
    return [int(v) for v in best_values]

if __name__ == '__main__':
    res = construct()
    print(res)
