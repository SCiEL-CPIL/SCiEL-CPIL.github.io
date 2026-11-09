import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s margin for safety
    
    candidates = list(range(2, 129))
    best_seq = []
    best_len = 0
    
    def greedy_search(order):
        seq = []
        prods = set()
        suffix_prods = [1]
        
        for x in order:
            new_prods = [x * s for s in suffix_prods]
            
            # Check for conflicts
            # Using a loop is faster than set intersection for small lists
            conflict = False
            for p in new_prods:
                if p in prods:
                    conflict = True
                    break
            
            if not conflict:
                seq.append(x)
                prods.update(new_prods)
                suffix_prods = [1] + [x * s for s in suffix_prods]
        
        return seq

    # Try a few deterministic orders first
    # Order 1: Natural
    seq = greedy_search(candidates)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq
        
    # Order 2: Primes first (approximate)
    # Simple prime check
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    order2 = primes + composites
    seq = greedy_search(order2)
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq

    # Randomized search
    while time.time() < deadline:
        order = random.sample(candidates, len(candidates))
        seq = greedy_search(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
    return best_seq
