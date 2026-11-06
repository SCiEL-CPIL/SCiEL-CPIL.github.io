import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Identify primes and composites in [2, 128]
    numbers = list(range(2, 129))
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [n for n in numbers if is_prime(n)]
    composites = [n for n in numbers if not is_prime(n)]
    
    best_seq = primes
    best_len = len(primes)
    
    # Helper to get all block products
    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods

    # Helper to check if inserting val into seq at position pos is valid
    def can_insert(seq, val):
        n = len(seq)
        # Find insertion position
        pos = 0
        while pos < n and seq[pos] < val:
            pos += 1
        
        # Get existing products
        existing_prods = get_products(seq)
        
        # Check new products involving val
        # Block [i, j] in new seq includes val if i <= pos <= j
        # In old seq indices:
        # Left part is seq[i, pos-1]
        # Right part is seq[pos, j]
        
        new_prods = set()
        
        # Precompute suffix products from pos to end? 
        # Or just iterate.
        
        for i in range(pos, -1, -1):
            left_prod = 1
            for k in range(i, pos):
                left_prod *= seq[k]
            
            for j in range(pos, n):
                right_prod = 1
                for k in range(pos, j):
                    right_prod *= seq[k]
                
                block_prod = left_prod * val * right_prod
                
                if block_prod in existing_prods or block_prod in new_prods:
                    return False
                new_prods.add(block_prod)
        
        return True

    # Main search loop
    # We try many random orders of composites
    # Since primes are always safe, we start with them.
    
    # Optimization: Sort composites by "difficulty" or just randomize.
    # Randomization helps escape local optima.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        current_set = set(primes)
        current_seq = list(primes)
        
        candidates = composites.copy()
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            if can_insert(current_seq, c):
                # Insert c into current_seq maintaining sort order
                # Python's bisect is good, but we can just insert and sort
                # Or find pos and insert
                pos = 0
                while pos < len(current_seq) and current_seq[pos] < c:
                    pos += 1
                current_seq.insert(pos, c)
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq
            
            # If we found a very good one, we might want to try to improve it further
            # by removing one element and adding two?
            # For now, just keep updating.
            
    return best_seq
