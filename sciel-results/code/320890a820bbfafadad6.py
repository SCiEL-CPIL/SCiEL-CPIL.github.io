import time
import random

def construct():
    # Primes <= 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    # Precompute exponent vectors for numbers 2..40
    # Use integer encoding: pack exponents into a single int
    # Each exponent is at most 5 (2^5=32, 2^6=64>40), so 3 bits per prime
    # 12 primes * 3 bits = 36 bits, fits in a Python int
    vecs = {}
    for n in range(2, 41):
        m = n
        encoded = 0
        for i, p in enumerate(primes):
            exp = 0
            while m % p == 0:
                exp += 1
                m //= p
            encoded |= (exp << (i * 3))
        vecs[n] = encoded
    
    numbers = list(range(2, 41))
    best_set = []
    best_size = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s buffer for overhead/return
    
    def check_valid(sums, v):
        """Check if adding v to current sums creates a collision."""
        for s in sums:
            if (s + v) in sums:
                return False
        return True
    
    def build_sums(aset):
        """Build the set of subset sum encodings for a given set."""
        sums = {0}
        for n in aset:
            v = vecs[n]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
        return sums
    
    # Phase 1: Greedy random restarts to get a good initial solution
    for _ in range(500):
        if time.time() > deadline:
            break
            
        random.shuffle(numbers)
        current_set = []
        current_sums = {0}
        
        for n in numbers:
            v = vecs[n]
            if check_valid(current_sums, v):
                current_set.append(n)
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + v)
                current_sums.update(new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 2: Simulated annealing / local search
    # Start from best_set and try swaps
    if best_size > 0:
        current_set = best_set[:]
        current_sums = build_sums(current_set)
        
        # Try to improve by swapping elements
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random element from current_set to remove
            if len(current_set) == 0:
                break
                
            idx_remove = random.randint(0, len(current_set) - 1)
            n_remove = current_set[idx_remove]
            
            # Pick a random element NOT in current_set to add
            available = [n for n in numbers if n not in current_set]
            if not available:
                break
            n_add = random.choice(available)
            
            # Try the swap
            new_set = current_set[:]
            new_set[idx_remove] = n_add
            new_sums = build_sums(new_set)
            
            # Check if the new set is valid (it should be by construction of build_sums,
            # but we need to ensure no collisions - build_sums doesn't check, it just builds)
            # Actually, we need to verify the swap is valid
            # A swap is valid if:
            # 1. Removing n_remove and adding n_add doesn't create collisions
            # This is equivalent to: the set of subset sums of new_set has size 2^k
            
            # Simpler: just check if new_sums has the expected size
            expected_size = 1 << len(new_set)
            if len(new_sums) == expected_size:
                # Valid swap
                current_set = new_set
                current_sums = new_sums
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
            else:
                # Invalid swap, try a different approach: just try to add elements
                pass
        
        # Also try pure additions: see if we can add any element
        for n in numbers:
            if n not in current_set:
                if check_valid(current_sums, vecs[n]):
                    current_set.append(n)
                    new_sums = set()
                    for s in current_sums:
                        new_sums.add(s + vecs[n])
                    current_sums.update(new_sums)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set[:]
    
    return best_set
