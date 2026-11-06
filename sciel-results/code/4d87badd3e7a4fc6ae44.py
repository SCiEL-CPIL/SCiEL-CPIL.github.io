import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute primes in [2, 128]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)

    # Precompute signatures for all numbers 2..128
    sigs = {}
    for i in range(2, 129):
        sig = [0] * P
        n = i
        for idx, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                sig[idx] += 1
                n //= p
        if n > 1:
            sig[prime_to_idx[n]] += 1
        sigs[i] = tuple(sig)

    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def greedy_construct(skip_set):
        """Greedily construct sequence, skipping numbers in skip_set."""
        seq = []
        products_set = set()
        for x in range(2, 129):
            if x in skip_set:
                continue
            x_sig = sigs[x]
            # Check all suffix products
            current = x_sig
            valid = True
            for s in reversed(sigs[s] for s in seq):
                if current in products_set:
                    valid = False
                    break
                current = add_sig(current, s)
            if current in products_set:
                valid = False
            if valid:
                seq.append(x)
                # Add all suffix products to set
                current = x_sig
                products_set.add(current)
                for s in reversed(sigs[s] for s in seq[:-1]):
                    current = add_sig(current, s)
                    products_set.add(current)
        return seq

    best_seq = greedy_construct(set())
    best_len = len(best_seq)

    # Strategy: Local search with swaps and removals
    # Use incremental validation for speed

    def validate_sequence(seq):
        """Check if sequence is valid. Returns (valid, products_set)."""
        products_set = set()
        for i, x in enumerate(seq):
            current = sigs[x]
            for j in range(i, -1, -1):
                if current in products_set:
                    return False, products_set
                products_set.add(current)
                if j > 0:
                    current = add_sig(current, sigs[seq[j-1]])
        return True, products_set

    def try_insert(seq, pos, x, products_set):
        """Try inserting x at position pos. Returns (success, new_products_set)."""
        # Check if inserting x at pos creates any collision
        # We need to check:
        # 1. All suffix products ending at x (within the new sequence)
        # 2. All suffix products that now include x (from positions before pos)
        
        x_sig = sigs[x]
        
        # Check suffix products ending at x
        current = x_sig
        if current in products_set:
            return False, None
        for j in range(pos - 1, -1, -1):
            current = add_sig(current, sigs[seq[j]])
            if current in products_set:
                return False, None
        
        # Check suffix products that start before pos and end after pos
        # These are products from seq[i]..seq[pos-1] * x * seq[pos+1]..seq[j]
        # Actually, we need to be more careful. Let's just do a full revalidation
        # for simplicity, but with early termination.
        
        new_seq = seq[:pos] + [x] + seq[pos:]
        valid, _ = validate_sequence(new_seq)
        if not valid:
            return False, None
        
        # Rebuild products_set
        new_products_set = set()
        for i, val in enumerate(new_seq):
            current = sigs[val]
            for j in range(i, -1, -1):
                new_products_set.add(current)
                if j > 0:
                    current = add_sig(current, sigs[new_seq[j-1]])
        
        return True, new_products_set

    # Local search: try inserting missing numbers
    current_seq = list(best_seq)
    current_products = validate_sequence(current_seq)[1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [x for x in range(2, 129) if x not in current_seq]
        
        for x in missing:
            if time.time() > deadline:
                break
            for pos in range(len(current_seq) + 1):
                success, new_products = try_insert(current_seq, pos, x, current_products)
                if success:
                    current_seq = current_seq[:pos] + [x] + current_seq[pos:]
                    current_products = new_products
                    improved = True
                    if len(current_seq) > best_len:
                        best_seq = list(current_seq)
                        best_len = len(current_seq)
                    break  # Move to next x

    # Try removals to enable better insertions
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            # Try removing a random element
            if len(current_seq) <= 2:
                break
            idx = random.randint(0, len(current_seq) - 1)
            old_x = current_seq[idx]
            new_seq = current_seq[:idx] + current_seq[idx+1:]
            valid, new_products = validate_sequence(new_seq)
            if valid:
                # Try to reinsert old_x or other missing numbers
                missing = [x for x in range(2, 129) if x not in new_seq]
                for x in missing:
                    for pos in range(len(new_seq) + 1):
                        success, pp = try_insert(new_seq, pos, x, new_products)
                        if success:
                            new_seq = new_seq[:pos] + [x] + new_seq[pos:]
                            new_products = pp
                            break
                if len(new_seq) > len(current_seq):
                    current_seq = new_seq
                    current_products = new_products
                    if len(current_seq) > best_len:
                        best_seq = list(current_seq)
                        best_len = len(current_seq)

    # Final greedy polish
    if time.time() < deadline:
        seq = greedy_construct(set())
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(best_seq)

    return best_seq
