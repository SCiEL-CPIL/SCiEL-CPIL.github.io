"""Rewritten program with beam search for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute prime factorizations for numbers 2..128
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    factor_cache = {}
    for x in range(2, 129):
        factor_cache[x] = factorize(x)
    
    # A product is represented by its prime factorization tuple
    def prod_key(factors):
        return tuple(sorted(factors.items()))
    
    # Beam search
    beam_width = 500
    best_seq = []
    best_products = set()
    best_len = 0
    
    # Initialize beam with single elements
    beam = []
    for x in range(2, 129):
        seq = [x]
        prods = {x}
        # Store: (sequence, set_of_products, sum_of_values)
        beam.append((seq, prods, x))
    
    # Sort initial beam by sum (descending) to prefer larger values
    beam.sort(key=lambda t: -t[2])
    beam = beam[:beam_width]
    
    while time.time() < deadline:
        next_beam = []
        for seq, prods, seq_sum in beam:
            last = seq[-1]
            # Try adding each number > last that's in [2,128]
            for x in range(last + 1, 129):
                # Compute new products: x, x*last, x*last*prev, ...
                new_prods = []
                prod = x
                new_prods.append(prod)
                for a in reversed(seq):
                    prod *= a
                    new_prods.append(prod)
                
                # Check for collisions
                if any(p in prods for p in new_prods):
                    continue
                if len(set(new_prods)) != len(new_prods):
                    continue
                
                new_seq = seq + [x]
                new_prods_set = prods | set(new_prods)
                new_sum = seq_sum + x
                next_beam.append((new_seq, new_prods_set, new_sum))
                
                # Track best
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    best_products = new_prods_set
        
        if not next_beam:
            break
        
        # Deduplicate by sequence (keep the one with... they're all unique by construction)
        # Sort by length desc, then sum desc
        next_beam.sort(key=lambda t: (-len(t[0]), -t[2]))
        beam = next_beam[:beam_width]
        
        if not beam:
            break
    
    # Return the best sequence found
    if best_seq:
        return best_seq
    
    # Fallback: greedy
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values
# EVOLVE-BLOCK-END
