import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    MAX_L = 100000
    # We need indices 1..MAX_L. Index 0 is dummy.
    # Size MAX_L + 1 is sufficient.
    
    def get_L(counts):
        # Find first index i >= 1 where counts[i] == 0
        # counts[1:] are the relevant parts
        zero_indices = np.where(counts[1:] == 0)[0]
        if len(zero_indices) == 0:
            return MAX_L
        return int(zero_indices[0]) + 1

    def optimize_set(prime_set, num_restarts=10):
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        P_sorted = sorted(prime_set)
        best_L = 0
        best_residues = {p: 0 for p in P_sorted}
        
        # We will try a few random initializations
        # and also a "greedy" initialization where we try to cover early numbers
        # but the local search should handle that.
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize residues
            residues = {p: random.randint(0, p - 1) for p in P_sorted}
            
            # Build counts
            counts = np.zeros(MAX_L + 1, dtype=np.int32)
            for p in P_sorted:
                r = residues[p]
                counts[r::p] += 1
            
            current_L = get_L(counts)
            
            # Local Search
            no_improve = 0
            max_no_improve = 20 # If we can't improve 20 times in a row, we're likely stuck
            
            while time.time() < deadline - 5 and no_improve < max_no_improve:
                if current_L == MAX_L:
                    break
                    
                n = current_L # First uncovered
                best_move = None
                best_new_L = current_L
                
                # Try changing each prime's residue to cover n
                for p in P_sorted:
                    r_new = n % p
                    if residues[p] == r_new:
                        continue
                    
                    r_old = residues[p]
                    
                    # Apply
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    
                    new_L = get_L(counts)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_move = (p, r_old, r_new)
                    
                    # Revert
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    
                if best_move:
                    p, r_old, r_new = best_move
                    residues[p] = r_new
                    counts[r_old::p] -= 1
                    counts[r_new::p] += 1
                    current_L = best_new_L
                    no_improve = 0
                else:
                    no_improve += 1
                    # If stuck, we could try a random kick, but let's just break
                    # and rely on other restarts
                    break
                    
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                
        return best_L, [[p, best_residues[p]] for p in P_sorted]

    # Generate Candidate Sets
    candidates = []
    
    def build_set(base_primes, mode='smallest'):
        s = sum(Fraction(1, p) for p in base_primes)
        if s > 1:
            return []
        final_set = list(base_primes)
        if mode == 'smallest':
            pool = [p for p in primes if p not in base_primes]
        else: # largest
            pool = [p for p in reversed(primes) if p not in base_primes]
            
        for p in pool:
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return final_set

    # 1. Greedy Smallest
    candidates.append(build_set([], 'smallest'))
    # 2. Greedy Largest
    candidates.append(build_set([], 'largest'))
    # 3. Base {2}, fill smallest
    candidates.append(build_set([2], 'smallest'))
    # 4. Base {3}, fill smallest
    candidates.append(build_set([3], 'smallest'))
    # 5. Base {2,3}, fill smallest
    candidates.append(build_set([2,3], 'smallest'))
    # 6. Base {5}, fill smallest
    candidates.append(build_set([5], 'smallest'))
    # 7. Base {2,5}, fill smallest
    candidates.append(build_set([2,5], 'smallest'))
    # 8. Base {3,5}, fill smallest
    candidates.append(build_set([3,5], 'smallest'))
    
    # Add some random sets
    random.seed(42)
    for _ in range(5):
        s = Fraction(0)
        res = []
        available = list(primes)
        random.shuffle(available)
        for p in available:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)

    best_L = 0
    best_classes = []
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
        if not P:
            continue
        
        # Adjust restarts based on time
        # If we have lots of time, more restarts.
        # Let's say 20 restarts per set.
        # 10 sets * 20 restarts = 200 local searches.
        # Each local search might take ~1-10s?
        # If it takes 1s, 200s is too long.
        # We need to be careful.
        
        # Let's estimate cost.
        # N=100,000.
        # p=2, update cost 50,000.
        # p=1000, update cost 100.
        # Avg update cost maybe 1000-5000 ops.
        # get_L cost 100,000 ops (numpy where).
        # So one step is ~100,000 ops.
        # If 100 steps per search, 10M ops. ~0.01s in numpy?
        # Python loop overhead is the killer.
        # The loop `for p in P_sorted` runs ~100 times.
        # Inside, we do numpy ops.
        # 100 numpy ops per step.
        # If 100 steps, 10,000 numpy ops.
        # This should be fast, maybe 0.1s per search.
        # 200 searches -> 20s. Safe.
        
        L, classes = optimize_set(P, num_restarts=20)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes
