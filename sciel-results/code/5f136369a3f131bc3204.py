import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin for safety
    
    best_seq = []
    best_len = 0
    
    def greedy_build(order):
        values = []
        products = set()
        for x in order:
            if x < 2 or x > 128:
                continue
            # Find insertion point
            lo, hi = 0, len(values)
            while lo < hi:
                mid = (lo + hi) // 2
                if values[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            
            # Generate new products
            new_prods = set()
            
            # x itself
            new_prods.add(x)
            
            # Left side: x * suffix ending at pos-1
            prod = x
            for i in range(pos - 1, -1, -1):
                prod *= values[i]
                new_prods.add(prod)
                
            # Right side: x * prefix starting at pos
            prod = x
            for i in range(pos, len(values)):
                prod *= values[i]
                new_prods.add(prod)
                
            # Check conflict
            if not new_prods.intersection(products):
                values.insert(pos, x)
                products.update(new_prods)
                
        return values

    # 1. Standard Ascending
    seq = greedy_build(range(2, 129))
    best_seq = seq
    best_len = len(seq)
    
    # 2. Descending
    if time.time() < deadline:
        seq = greedy_build(range(128, 1, -1))
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    # 3. Primes First
    if time.time() < deadline:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        non_primes = [x for x in range(2, 129) if x not in primes]
        order = primes + non_primes
        seq = greedy_build(order)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            
    # 4. Randomized Restarts
    if time.time() < deadline:
        random.seed(42)
        # Run as many as possible
        while time.time() < deadline - 2:
            order = list(range(2, 129))
            random.shuffle(order)
            seq = greedy_build(order)
            if len(seq) > best_len:
                best_seq = seq
                best_len = len(seq)
                
    return best_seq
