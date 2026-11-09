"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import math

def construct():
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Select a high-density prime set
    # {3, 5, 7, 11, 13, 17, 19, 23} sum(1/p) ~ 0.997
    candidate_primes = [3, 5, 7, 11, 13, 17, 19, 23]
    
    # Verify budget
    total_budget = sum(Fraction(1, p) for p in candidate_primes)
    if total_budget > 1:
        # Fallback to a safer set if needed, but this one is valid
        candidate_primes = [2, 3, 7, 43]
        total_budget = sum(Fraction(1, p) for p in candidate_primes)

    best_L = 0
    best_classes = []
    
    L_MAX = 100000
    NUM_RESTARTS = 10
    LOCAL_SEARCH_ITERS = 200
    
    def greedy_repair(primes, residues, L_limit):
        n_primes = len(primes)
        cover_count = [0] * (L_limit + 1)
        
        for i, p in enumerate(primes):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            while start <= L_limit:
                cover_count[start] += 1
                start += p
        
        current_L = 0
        while current_L < L_limit and cover_count[current_L + 1] > 0:
            current_L += 1
            
        if current_L == L_limit:
            return L_limit, residues[:]

        while current_L < L_limit:
            target = current_L + 1
            found = False
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for idx in indices:
                p = primes[idx]
                old_r = residues[idx]
                new_r = target % p
                
                if old_r == new_r:
                    continue
                    
                valid = True
                k = old_r
                if k == 0:
                    k = p
                while k <= current_L:
                    if cover_count[k] == 1:
                        valid = False
                        break
                    k += p
                
                if valid:
                    k = old_r
                    if k == 0:
                        k = p
                    while k <= L_limit:
                        cover_count[k] -= 1
                        k += p
                        
                    k = new_r
                    if k == 0:
                        k = p
                    while k <= L_limit:
                        cover_count[k] += 1
                        k += p
                        
                    residues[idx] = new_r
                    current_L = target
                    found = True
                    break
            
            if not found:
                break
                
        return current_L, residues[:]

    def local_search(primes, residues, L_limit, iters):
        n_primes = len(primes)
        cover_count = [0] * (L_limit + 1)
        
        for i, p in enumerate(primes):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            while start <= L_limit:
                cover_count[start] += 1
                start += p
        
        current_L = 0
        while current_L < L_limit and cover_count[current_L + 1] > 0:
            current_L += 1
            
        if current_L == L_limit:
            return L_limit, residues[:]

        best_L = current_L
        best_residues = residues[:]
        
        for _ in range(iters):
            if current_L == L_limit:
                break
                
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            
            if random.random() < 0.5:
                new_r = (current_L + 1) % p
            else:
                new_r = random.randint(0, p - 1)
                
            if new_r == old_r:
                continue
                
            valid = True
            k = old_r
            if k == 0:
                k = p
            while k <= current_L:
                if cover_count[k] == 1:
                    valid = False
                    break
                k += p
                
            if valid:
                k = old_r
                if k == 0:
                    k = p
                while k <= L_limit:
                    cover_count[k] -= 1
                    k += p
                    
                k = new_r
                if k == 0:
                    k = p
                while k <= L_limit:
                    cover_count[k] += 1
                    k += p
                    
                residues[idx] = new_r
                
                while current_L < L_limit and cover_count[current_L + 1] > 0:
                    current_L += 1
                    
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                
        return best_L, best_residues

    for _ in range(NUM_RESTARTS):
        residues = [random.randint(0, p - 1) for p in candidate_primes]
        
        L1, res1 = greedy_repair(candidate_primes, residues, L_MAX)
        L2, res2 = local_search(candidate_primes, res1, L_MAX, LOCAL_SEARCH_ITERS)
        
        if L2 > best_L:
            best_L = L2
            best_classes = [[p, r] for p, r in zip(candidate_primes, res2)]
            
        if best_L >= 999:
            break

    total = sum(Fraction(1, p) for p, r in best_classes)
    assert total <= 1, f"Budget exceeded: {total}"
    
    return best_classes
# EVOLVE-BLOCK-END
