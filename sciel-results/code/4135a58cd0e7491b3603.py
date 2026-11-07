import time
import random

def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Map number to its index in primes for fast lookup? 
    # Or just compute vector directly.
    
    # Precompute exponent vectors for numbers 2..40
    # vec[n] is a tuple of exponents
    vecs = {}
    for n in range(2, 41):
        v = [0] * num_primes
        m = n
        for i, p in enumerate(primes):
            while m % p == 0:
                v[i] += 1
                m //= p
        # m should be 1 since primes cover all factors <= 40
        vecs[n] = tuple(v)
        
    numbers = list(range(2, 41))
    best_set = []
    best_size = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s buffer for overhead/return
    
    # We will run multiple random restarts
    # Each restart: shuffle numbers, greedily build a dissociated set
    # To improve performance, we can also try different orderings or local search
    
    # Optimization: Represent subset sums as a set of tuples.
    # Adding v: new_sums = {s + v for s in current_sums}
    # Check intersection: if any s in new_sums is in current_sums, invalid.
    # Actually, we just need to check if (current_sums + v) intersects current_sums.
    # Since 0 is in current_sums, v itself must not be in current_sums.
    # More generally, no subset sum of current set can differ from another by v.
    
    # To speed up, we can use a bitset or just set of tuples.
    # Tuples of 12 small ints are hashable and fast.
    
    for _ in range(1000): # Try up to 1000 restarts
        if time.time() > deadline:
            break
            
        random.shuffle(numbers)
        current_set = []
        current_sums = {tuple([0]*num_primes)} # Zero vector
        
        for n in numbers:
            v = vecs[n]
            # Check validity
            # Calculate shifted sums
            # We can check if any (s + v) is in current_sums
            # This is equivalent to checking if v is in {s1 - s2 | s1, s2 in current_sums}
            # But computing the difference set is expensive.
            # Iterating over current_sums and checking membership in current_sums - v is O(2^k).
            
            valid = True
            # Optimization: if len(current_sums) is large, this is slower.
            # But 2^17 is ~130k. 130k checks is fast.
            for s in current_sums:
                # s + v
                sv = tuple(s[i] + v[i] for i in range(num_primes))
                if sv in current_sums:
                    valid = False
                    break
            
            if valid:
                current_set.append(n)
                # Update sums
                # New sums are old sums and old sums + v
                # We must be careful not to modify while iterating
                new_sums = set()
                for s in current_sums:
                    sv = tuple(s[i] + v[i] for i in range(num_primes))
                    new_sums.add(sv)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found 18, we can stop early as it's the max possible distinct integers in 2..40 is 39, but dissociated sets are limited.
            # Actually, can we have more than 18?
            # The number of primes is 12.
            # A dissociated set in Z^12 can be large.
            # However, the coordinates are bounded.
            # Let's just keep searching.
            
    return best_set
