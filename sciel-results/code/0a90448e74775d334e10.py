import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)

    sigs = [None] * 129
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
            idx = prime_to_idx[n]
            sig[idx] += 1
        sigs[i] = tuple(sig)

    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def greedy_construct(skip_set):
        seq_vals = []
        seq_sigs = []
        products_set = set()
        for x in range(2, 129):
            if x in skip_set:
                continue
            x_sig = sigs[x]
            
            # Check validity
            new_tails = [x_sig]
            current_prod = x_sig
            for s in reversed(seq_sigs):
                current_prod = add_sig(current_prod, s)
                new_tails.append(current_prod)
                
            valid = True
            for tail in new_tails:
                if tail in products_set:
                    valid = False
                    break
            
            if valid:
                seq_vals.append(x)
                seq_sigs.append(x_sig)
                products_set.update(new_tails)
        return seq_vals

    def polish(seq):
        seq = list(seq)
        current_set = set(seq)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(2, 129):
                if x in current_set:
                    continue
                
                # Find insertion index
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                insert_idx = lo
                
                if insert_idx > 0 and seq[insert_idx - 1] >= x:
                    continue
                if insert_idx < len(seq) and seq[insert_idx] <= x:
                    continue
                
                # Check validity
                seq_sigs = [sigs[v] for v in seq]
                
                # Compute existing prods
                prods = set()
                for i in range(len(seq)):
                    prod = 1
                    for k in range(i, len(seq)):
                        prod = add_sig(prod, seq_sigs[k])
                        prods.add(prod)
                        
                # Check new tails
                x_sig = sigs[x]
                valid = True
                new_tails_set = set()
                cum = x_sig
                if cum in prods or cum in new_tails_set:
                    valid = False
                else:
                    new_tails_set.add(cum)
                    for j in range(insert_idx - 1, -1, -1):
                        cum = add_sig(cum, seq_sigs[j])
                        if cum in prods or cum in new_tails_set:
                            valid = False
                            break
                        new_tails_set.add(cum)
                
                if valid:
                    seq.insert(insert_idx, x)
                    current_set.add(x)
                    changed = True
                    break
                    
        return seq

    best_seq = []
    
    # Strategy 1: Standard Greedy + Polish
    seq = greedy_construct(set())
    seq = polish(seq)
    if len(seq) > len(best_seq):
        best_seq = seq

    # Strategy 2: Randomized Greedy + Polish
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        skip_set = set()
        for x in range(2, 50):
            if random.random() < 0.2:
                skip_set.add(x)
        seq = greedy_construct(skip_set)
        seq = polish(seq)
        if len(seq) > len(best_seq):
            best_seq = seq

    return best_seq

