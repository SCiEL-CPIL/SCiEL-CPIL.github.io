import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    best_seq = []
    best_len = 0
    
    BEAM_WIDTH = 400
    
    def get_new_products(seq, c):
        """Get the list of new products when appending c to seq."""
        prods = []
        prod = c
        prods.append(prod)
        for a in reversed(seq):
            prod *= a
            prods.append(prod)
        return prods
    
    def has_collision(prods, product_set):
        """Check if any product in prods is in product_set."""
        for p in prods:
            if p in product_set:
                return True
        return False
    
    # Generate good candidate values: primes, prime powers, and small composites
    # Focus on values that are "independent" in terms of prime factorization
    def generate_candidates(last_val, count):
        """Generate candidate values greater than last_val, prioritizing primes."""
        candidates = set()
        # Add primes
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
        
        # Add some primes
        v = last_val + 1
        prime_count = 0
        while v <= 128 and prime_count < count // 2:
            if is_prime(v):
                candidates.add(v)
                prime_count += 1
            v += 1
        
        # Add some small composites (products of 2 primes)
        v = last_val + 1
        comp_count = 0
        while v <= 128 and comp_count < count // 4:
            # Check if v is a product of exactly 2 primes (or prime power)
            if v > 1:
                # Simple check: if v is not prime and not 1, consider it
                if not is_prime(v) and v > 1:
                    candidates.add(v)
                    comp_count += 1
            v += 1
        
        # Fill in with consecutive values if needed
        if len(candidates) < count:
            v = last_val + 1
            while v <= 128 and len(candidates) < count:
                candidates.add(v)
                v += 1
        
        return sorted(candidates)
    
    # Beam search
    beams = [([], set())]
    
    while time.time() < deadline:
        new_beams = []
        
        for seq, prods in beams:
            if not seq:
                last_val = 1
            else:
                last_val = seq[-1]
            
            # Generate diverse candidates
            candidates = generate_candidates(last_val, 20)
            
            for c in candidates:
                if c > 128:
                    break
                new_prods = get_new_products(seq, c)
                
                if not has_collision(new_prods, prods):
                    new_seq = seq + [c]
                    new_prod_set = prods.union(new_prods)
                    new_beams.append((new_seq, new_prod_set))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not new_beams:
            break
        
        # Sort beams: prefer longer, then smaller last value (more room to extend),
        # then fewer total products (less collision risk)
        def beam_sort_key(b):
            seq, prods = b
            return (-len(seq), seq[-1] if seq else 0, len(prods))
        
        new_beams.sort(key=beam_sort_key)
        beams = new_beams[:BEAM_WIDTH]
        
        # Adapt beam width
        remaining_time = deadline - time.time()
        if best_len >= 35:
            BEAM_WIDTH = max(100, int(400 * remaining_time / (deadline - start_time)))
        elif best_len >= 25:
            BEAM_WIDTH = max(200, int(400 * remaining_time / (deadline - start_time)))
    
    return best_seq
