import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Precompute gcd relationships
    # conflict[u][v] = True if gcd(u,v) > 1
    # We only care about numbers 1..511
    # Instead of a full matrix, we can use prime factors for quick check
    # But for local search, a set of conflicts or a bitset might be faster.
    # Let's stick to gcd checks but optimize the search.

    def is_coprime(a, b):
        return gcd(a, b) == 1

    def calc_score(A):
        # Using Fraction for exactness in comparisons if needed, but float is faster for scoring
        # The task says "exact rational, reported as a float".
        # For local search decisions, float is sufficient and much faster.
        return sum(1.0 / (512.0 - a) for a in A)

    def calc_score_exact(A):
        s = Fraction(0)
        for a in A:
            s += Fraction(1, 512 - a)
        return s

    # --- Phase 1: Initial Construction ---
    # Greedy: Start with high numbers (large 1/(512-a))
    # But we need pairwise coprime.
    # A strong heuristic: Start with primes? No, composites can be better if they are large.
    # Example: 511 = 7 * 73. 1/(512-511) = 1.
    # If we pick 511, we cannot pick 7 or 73.
    # 1/(512-7) + 1/(512-73) = 1/505 + 1/439 ≈ 0.00198 + 0.00227 ≈ 0.00425 < 1.
    # So 511 is much better than 7 and 73 combined.
    
    # Greedy strategy:
    # 1. Sort candidates by value 1/(512-a) descending (i.e., a descending).
    # 2. Iterate, if coprime to all selected, add it.
    
    A = []
    for a in range(511, 0, -1):
        if all(is_coprime(a, b) for b in A):
            A.append(a)
    
    current_score = calc_score(A)
    
    # --- Phase 2: Local Search ---
    # Moves:
    # 1. Swap: Replace a in A with b not in A, if b is coprime to A\{a} and score improves.
    # 2. Merge: Replace p, q in A with c = p*q if c < 512, c coprime to A\{p,q}, and score improves.
    # 3. Add: Add b not in A if coprime to A (score always improves).
    # 4. 2-for-1 Swap: Replace p, q in A with b1, b2 not in A? (Complex, maybe skip for now or limit)
    # 5. 1-for-2 Split: Replace c in A with p, q such that c = p*q? (c must be composite).
    
    improved = True
    in_A = set(A)
    
    while improved and time.time() < deadline:
        improved = False
        n = len(A)
        
        # 1. Try Adds (fastest, always improves if valid)
        # Check numbers not in A, descending order (highest value first)
        for b in range(511, 0, -1):
            if time.time() > deadline: break
            if b in in_A: continue
            if all(is_coprime(b, a) for a in A):
                A.append(b)
                in_A.add(b)
                current_score += 1.0 / (512.0 - b)
                improved = True
                break # Restart to see if more can be added
        
        if improved: continue
        
        # 2. Try Merges (2-for-1)
        # Iterate over pairs in A
        # Optimization: Only consider pairs where p*q < 512
        # And where the merge actually improves score:
        # 1/(512-pq) > 1/(512-p) + 1/(512-q)
        
        merged = False
        for i in range(len(A)):
            if time.time() > deadline: break
            for j in range(i + 1, len(A)):
                p = A[i]
                q = A[j]
                c = p * q
                if c >= 512: continue
                
                # Check score improvement
                old_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                new_val = 1.0 / (512.0 - c)
                if new_val <= old_val: continue
                
                # Check coprimality with rest of A
                valid = True
                for k in range(len(A)):
                    if k == i or k == j: continue
                    if not is_coprime(c, A[k]):
                        valid = False
                        break
                if valid:
                    # Perform merge
                    new_A = []
                    for k in range(len(A)):
                        if k != i and k != j:
                            new_A.append(A[k])
                    new_A.append(c)
                    
                    A = new_A
                    in_A = set(A)
                    current_score += (new_val - old_val)
                    improved = True
                    merged = True
                    break
            if merged: break
        
        if improved: continue
        
        # 3. Try Swaps (1-for-1)
        # Replace a in A with b not in A
        # We want b > a (higher value) and b coprime to A\{a}
        
        swapped = False
        for i in range(len(A)):
            if time.time() > deadline: break
            a = A[i]
            # Try b from 511 down to a+1
            # Optimization: We can pre-check if b is coprime to A\{a}
            for b in range(511, a, -1):
                if b in in_A: continue
                if b == a: continue # Just in case, though range excludes it
                
                # Check coprimality
                valid = True
                for j in range(len(A)):
                    if j == i: continue
                    if not is_coprime(b, A[j]):
                        valid = False
                        break
                
                if valid:
                    # Score improvement is guaranteed since b > a
                    A[i] = b
                    in_A.remove(a)
                    in_A.add(b)
                    current_score += (1.0 / (512.0 - b)) - (1.0 / (512.0 - a))
                    improved = True
                    swapped = True
                    break
            if swapped: break
        
        if improved: continue
        
        # 4. Try Splits (1-for-2)
        # Replace c in A with p, q such that c = p * q
        # This is only useful if 1/(512-p) + 1/(512-q) > 1/(512-c)
        # Note: This increases the number of elements.
        # c must be composite.
        # p, q must be coprime to each other (they are factors of c, so if c is square-free or we pick distinct prime factors? No, p and q just need to be coprime to each other and to A\{c}).
        # Actually, if c = p*q, gcd(p,q) might not be 1. e.g. c=4, p=2, q=2. gcd(2,2)=2. Not coprime.
        # So we need p, q coprime. This implies c must be a product of two coprime integers > 1.
        # i.e. c must have at least two distinct prime factors.
        
        split = False
        for i in range(len(A)):
            if time.time() > deadline: break
            c = A[i]
            
            # Find factors p, q of c such that p*q=c, gcd(p,q)=1, p,q > 1
            # And p, q not in A\{c} (or if in A, we'd be duplicating, which is not allowed. "distinct" integers)
            # Wait, the problem says "distinct pairwise coprime integers".
            # So p and q must not be in A\{c}.
            
            # Iterate divisors of c
            found_pair = False
            for p in range(2, int(c**0.5) + 1):
                if c % p != 0: continue
                q = c // p
                if p >= q: break # Avoid duplicates, though order doesn't matter for set
                if gcd(p, q) != 1: continue
                
                # Check if p, q are in A
                if p in in_A or q in in_A: continue
                
                # Check coprimality with A\{c}
                valid = True
                for j in range(len(A)):
                    if j == i: continue
                    if not is_coprime(p, A[j]) or not is_coprime(q, A[j]):
                        valid = False
                        break
                
                if valid:
                    # Check score
                    old_val = 1.0 / (512.0 - c)
                    new_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                    if new_val > old_val:
                        # Perform split
                        new_A = []
                        for j in range(len(A)):
                            if j != i:
                                new_A.append(A[j])
                        new_A.append(p)
                        new_A.append(q)
                        
                        A = new_A
                        in_A = set(A)
                        current_score += (new_val - old_val)
                        improved = True
                        split = True
                        found_pair = True
                        break
            if split: break
        
        if improved: continue
        
        # 5. Random Perturbation / Restart to escape local optima?
        # If stuck, maybe do a few random swaps and then re-optimize?
        # Given the time limit, let's try a simple random restart if we've been stuck for a while.
        # Or just accept the current local optimum.
        
        # Let's try a few random 1-for-1 swaps that don't necessarily improve score immediately,
        # then run the local search again.
        # This is risky with time, so let's do it sparingly.
        
        if time.time() < deadline - 10:
            # Try 10 random perturbations
            for _ in range(10):
                if time.time() > deadline: break
                
                # Pick a random element to swap out
                idx = random.randint(0, len(A) - 1)
                old_a = A[idx]
                
                # Pick a random candidate b
                # Prefer high b
                candidates = [b for b in range(511, 0, -1) if b not in in_A]
                if not candidates: continue
                
                # Weighted random choice towards higher b
                # Simple: just pick random
                b = random.choice(candidates)
                
                # Check validity
                valid = True
                for j in range(len(A)):
                    if j == idx: continue
                    if not is_coprime(b, A[j]):
                        valid = False
                        break
                
                if not valid: continue
                
                # Perform swap
                A[idx] = b
                in_A.remove(old_a)
                in_A.add(b)
                current_score = calc_score(A)
                
                # Re-run local search briefly
                improved_local = True
                while improved_local and time.time() < deadline - 2:
                    improved_local = False
                    # Adds
                    for b_add in range(511, 0, -1):
                        if b_add in in_A: continue
                        if all(is_coprime(b_add, a) for a in A):
                            A.append(b_add)
                            in_A.add(b_add)
                            current_score += 1.0 / (512.0 - b_add)
                            improved_local = True
                            break
                    if improved_local: continue
                    
                    # Merges
                    merged = False
                    for i in range(len(A)):
                        for j in range(i + 1, len(A)):
                            p, q = A[i], A[j]
                            c = p * q
                            if c >= 512: continue
                            old_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                            new_val = 1.0 / (512.0 - c)
                            if new_val <= old_val: continue
                            valid = True
                            for k in range(len(A)):
                                if k == i or k == j: continue
                                if not is_coprime(c, A[k]):
                                    valid = False
                                    break
                            if valid:
                                new_A = [A[k] for k in range(len(A)) if k != i and k != j]
                                new_A.append(c)
                                A = new_A
                                in_A = set(A)
                                current_score += (new_val - old_val)
                                improved_local = True
                                merged = True
                                break
                        if merged: break
                    if improved_local: continue
                    
                    # Swaps
                    swapped = False
                    for i in range(len(A)):
                        a = A[i]
                        for b_swap in range(511, a, -1):
                            if b_swap in in_A: continue
                            valid = True
                            for j in range(len(A)):
                                if j == i: continue
                                if not is_coprime(b_swap, A[j]):
                                    valid = False
                                    break
                            if valid:
                                A[i] = b_swap
                                in_A.remove(a)
                                in_A.add(b_swap)
                                current_score += (1.0 / (512.0 - b_swap)) - (1.0 / (512.0 - a))
                                improved_local = True
                                swapped = True
                                break
                        if swapped: break
                    if improved_local: continue
                    
                    # Splits
                    split = False
                    for i in range(len(A)):
                        c = A[i]
                        for p in range(2, int(c**0.5) + 1):
                            if c % p != 0: continue
                            q = c // p
                            if p >= q: break
                            if gcd(p, q) != 1: continue
                            if p in in_A or q in in_A: continue
                            valid = True
                            for j in range(len(A)):
                                if j == i: continue
                                if not is_coprime(p, A[j]) or not is_coprime(q, A[j]):
                                    valid = False
                                    break
                            if valid:
                                old_val = 1.0 / (512.0 - c)
                                new_val = 1.0 / (512.0 - p) + 1.0 / (512.0 - q)
                                if new_val > old_val:
                                    new_A = [A[j] for j in range(len(A)) if j != i]
                                    new_A.append(p)
                                    new_A.append(q)
                                    A = new_A
                                    in_A = set(A)
                                    current_score += (new_val - old_val)
                                    improved_local = True
                                    split = True
                                    break
                        if split: break
                    if improved_local: continue

    return A
