import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to be safe within 60s limit
    
    # Precompute prime factorization vectors for numbers 2-128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    def factorize(x):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while x % p == 0:
                vec[i] += 1
                x //= p
        return vec
    
    # Precompute exponent vectors
    exp_vectors = {}
    for x in range(2, 129):
        exp_vectors[x] = factorize(x)
    
    # 1. Initial Greedy Solution
    # The greedy approach adds x if it doesn't create a duplicate product.
    # We iterate x from 2 to 128.
    values, products = [], set()
    for x in range(2, 129):
        # Calculate all new products ending at x
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_values = values[:]
    best_len = len(values)
    
    # Helper to check validity using exponent vectors
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        
        # Build prefix sum of exponent vectors
        # prefix[i] = sum of exp_vectors[seq[0]] + ... + exp_vectors[seq[i-1]]
        prefix = [[0] * num_primes]
        for i in range(n):
            vec = exp_vectors[seq[i]]
            prev = prefix[-1]
            new_vec = [prev[j] + vec[j] for j in range(num_primes)]
            prefix.append(new_vec)
        
        prods = set()
        for i in range(n):
            for j in range(i, n):
                # Product of seq[i..j] has exponent vector prefix[j+1] - prefix[i]
                exp_vec = tuple(prefix[j+1][k] - prefix[i][k] for k in range(num_primes))
                if exp_vec in prods:
                    return False
                prods.add(exp_vec)
        return True

    # Helper to get sorted list from set
    def get_sorted(s):
        return sorted(s)
    
    # Helper to try inserting a value into a sequence at any valid position
    def try_insert(seq, c):
        n = len(seq)
        for i in range(n + 1):
            # Check if inserting c at i keeps the sequence strictly increasing
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            new_seq = seq[:i] + [c] + seq[i:]
            if is_valid(new_seq):
                return new_seq
        return None

    # Current state
    current_set = set(best_values)
    current_len = best_len
    current_seq = get_sorted(current_set)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Phase 1: Try to add any element that keeps validity using insertion
        added = False
        candidates = [x for x in range(2, 129) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_seq = try_insert(current_seq, x)
            if new_seq is not None:
                current_seq = new_seq
                current_set = set(new_seq)
                current_len += 1
                added = True
                # Update best
                if current_len > best_len:
                    best_len = current_len
                    best_values = current_seq[:]
                break # Restart candidate list after one addition
        
        if added:
            continue # Loop back to try adding more
        
        # Phase 2: Stuck. Perform a kick (remove some elements) and repair.
        # Remove 1-3 random elements
        kick_size = random.randint(1, 3)
        if current_len > kick_size:
            to_remove = random.sample(current_seq, kick_size)
            for x in to_remove:
                current_set.remove(x)
                current_seq.remove(x)
            current_len -= kick_size
            
            # Repair: Greedily add elements back using insertion
            avail = [x for x in range(2, 129) if x not in current_set]
            random.shuffle(avail)
            
            for x in avail:
                if time.time() > deadline:
                    break
                new_seq = try_insert(current_seq, x)
                if new_seq is not None:
                    current_seq = new_seq
                    current_set = set(new_seq)
                    current_len += 1
            
            # If we improved or maintained, we are in a new local optimum.
            # If we dropped length significantly, revert to best_values.
            if current_len < best_len - 2:
                current_set = set(best_values)
                current_len = best_len
                current_seq = best_values[:]

    return best_values
