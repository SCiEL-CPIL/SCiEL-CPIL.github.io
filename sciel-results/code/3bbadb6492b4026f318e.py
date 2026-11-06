import time
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_valid(seq):
    if not seq:
        return True
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    primes = [p for p in range(2, 129) if is_prime(p)]
    composites = [c for c in range(2, 129) if not is_prime(c)]
    
    best_set = set(primes)
    best_len = len(primes)
    
    # Use a fixed seed for reproducibility in debugging, but random is fine for evaluation
    # random.seed(42) 
    
    # Perform multiple random passes to try different insertion orders
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        current_set = set(primes)
        candidates = composites.copy()
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            # Check if adding c is valid
            # Optimization: if len(current_set) + 1 <= best_len, we might still want to add
            # to potentially enable further additions.
            new_set = current_set | {c}
            seq = sorted(new_set)
            if is_valid(seq):
                current_set = new_set
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set
            
    return sorted(best_set)
