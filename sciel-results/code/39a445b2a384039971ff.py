"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    N = 512
    # Candidates: 1..511
    # Value of a: 1/(512-a)
    # Conflict: gcd(a,b) > 1
    
    # Precompute prime factors for each number
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    # For speed, precompute gcd conflicts using prime factors
    # Two numbers conflict iff they share a prime factor
    pf = [set() for _ in range(N)]  # pf[a] = prime factors of a, for a in 0..511
    for a in range(1, N):
        pf[a] = prime_factors(a)
    
    def conflicts(a, b):
        """Return True if a and b share a prime factor (i.e., not coprime)"""
        if a == 1 or b == 1:
            return False
        # Check if any prime factor is shared
        for p in pf[a]:
            if p in pf[b]:
                return True
        return False
    
    def conflicts_with_set(a, chosen_set):
        """Check if a conflicts with any in chosen_set"""
        if a == 1:
            return False
        for b in chosen_set:
            if b == 1:
                continue
            for p in pf[a]:
                if p in pf[b]:
                    return True
        return False
    
    # Greedy initialization
    chosen = set()
    # Sort by value descending
    candidates = list(range(N-1, 0, -1))  # 511 down to 1
    for a in candidates:
        if not conflicts_with_set(a, chosen):
            chosen.add(a)
    
    def score(s):
        return sum(1.0/(N-a) for a in s)
    
    best_score = score(chosen)
    best_chosen = set(chosen)
    
    # Local search
    unchosen = set(range(1, N)) - chosen
    
    while time.time() < DEADLINE:
        improved = False
        
        # Try adding any unchosen number that doesn't conflict
        for a in list(unchosen):
            if not conflicts_with_set(a, chosen):
                new_score = best_score + 1.0/(N-a)
                if new_score > best_score:
                    chosen.add(a)
                    unchosen.remove(a)
                    best_score = new_score
                    best_chosen = set(chosen)
                    improved = True
        
        # Try 1-for-1 swaps: remove one chosen, add one unchosen
        for a in list(chosen):
            val_a = 1.0/(N-a)
            # Try removing a
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding any unchosen that fits
            for b in list(unchosen):
                if b == a:
                    continue
                if not conflicts_with_set(b, chosen):
                    new_score = best_score - val_a + 1.0/(N-b)
                    if new_score > best_score:
                        chosen.add(b)
                        unchosen.remove(b)
                        best_score = new_score
                        best_chosen = set(chosen)
                        improved = True
                        break  # restart inner loop
            # Restore a if we didn't swap
            if a not in chosen and a in unchosen:
                chosen.add(a)
                unchosen.remove(a)
        
        # Try 2-for-1 swaps: remove two chosen, add one unchosen
        chosen_list = list(chosen)
        for i in range(len(chosen_list)):
            if time.time() > DEADLINE:
                break
            a1 = chosen_list[i]
            for j in range(i+1, len(chosen_list)):
                a2 = chosen_list[j]
                val_a1 = 1.0/(N-a1)
                val_a2 = 1.0/(N-a2)
                
                chosen.discard(a1)
                chosen.discard(a2)
                unchosen.add(a1)
                unchosen.add(a2)
                
                # Try adding any unchosen that fits
                for b in list(unchosen):
                    if b == a1 or b == a2:
                        continue
                    if not conflicts_with_set(b, chosen):
                        new_score = best_score - val_a1 - val_a2 + 1.0/(N-b)
                        if new_score > best_score:
                            chosen.add(b)
                            unchosen.remove(b)
                            best_score = new_score
                            best_chosen = set(chosen)
                            improved = True
                            break
                # Restore
                if a1 not in chosen:
                    chosen.add(a1)
                    unchosen.discard(a1)
                if a2 not in chosen:
                    chosen.add(a2)
                    unchosen.discard(a2)
        
        # Try 1-for-2 swaps: remove one chosen, add two unchosen
        for a in list(chosen):
            if time.time() > DEADLINE:
                break
            val_a = 1.0/(N-a)
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding pairs of unchosen
            unchosen_list = [b for b in unchosen if b != a]
            for i in range(len(unchosen_list)):
                if time.time() > DEADLINE:
                    break
                b1 = unchosen_list[i]
                if not conflicts_with_set(b1, chosen):
                    chosen.add(b1)
                    unchosen.remove(b1)
                    
                    # Try adding a second
                    for b2 in list(unchosen):
                        if b2 == a or b2 == b1:
                            continue
                        if not conflicts_with_set(b2, chosen):
                            new_score = best_score - val_a + 1.0/(N-b1) + 1.0/(N-b2)
                            if new_score > best_score:
                                chosen.add(b2)
                                unchosen.remove(b2)
                                best_score = new_score
                                best_chosen = set(chosen)
                                improved = True
                                break
                    # Restore b1 if not swapped
                    if b1 in chosen and b1 not in best_chosen:
                        # Keep it if it helped, else restore
                        pass
                    # Actually let me redo this more carefully
                    # Restore b1 if we didn't keep it
                    if b1 not in best_chosen:
                        chosen.discard(b1)
                        unchosen.add(b1)
            
            # Restore a if not swapped
            if a not in chosen:
                chosen.add(a)
                unchosen.discard(a)
        
        if not improved:
            break
    
    return list(best_chosen)
# EVOLVE-BLOCK-END
