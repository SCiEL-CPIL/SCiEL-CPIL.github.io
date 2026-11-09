import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    candidates = list(range(2, 129))
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    
    def get_products(seq):
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return prods

    def try_insertion(seq, c):
        # Returns (new_seq, success)
        for i in range(len(seq) + 1):
            # Check if inserting c at i keeps the sequence strictly increasing
            if i > 0 and seq[i-1] >= c:
                continue
            if i < len(seq) and c >= seq[i]:
                continue
            
            new_seq = seq[:i] + [c] + seq[i:]
            prods = get_products(new_seq)
            if len(prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                return new_seq, True
        return None, False

    best_seq = primes
    best_len = len(best_seq)
    
    # Multiple restarts with different random orders
    for _ in range(20): # Max 20 restarts
        if time.time() > deadline - 5:
            break
            
        current_seq = primes[:]
        used = set(current_seq)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        
        # Greedy insertion
        progress = True
        while progress:
            progress = False
            for c in remaining:
                if time.time() > deadline - 2:
                    break
                new_seq, success = try_insertion(current_seq, c)
                if success:
                    current_seq = new_seq
                    used.add(c)
                    remaining.remove(c)
                    if len(current_seq) > best_len:
                        best_seq = current_seq[:]
                        best_len = len(current_seq)
                    progress = True
                    break # Restart the loop with the new sequence
            
        if time.time() > deadline - 5:
            break

    return best_seq
