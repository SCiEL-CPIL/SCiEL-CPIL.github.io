"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
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
    
    # Precompute prime factors
    pf = [set() for _ in range(N)]  # pf[a] = prime factors of a, for a in 0..511
    for a in range(1, N):
        pf[a] = prime_factors(a)
    
    # Precompute conflict bitmasks for each number
    # conflict_mask[a] has bit b set if gcd(a,b) > 1
    conflict_mask = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and any(p in pf[b] for p in pf[a]):
                mask |= (1 << b)
        conflict_mask[a] = mask
    
    # Greedy initialization - pick high value numbers first
    chosen = set()
    # Sort by value descending (higher a means higher value 1/(512-a))
    candidates = list(range(N-1, 0, -1))  # 511 down to 1
    for a in candidates:
        # Check if a conflicts with any chosen
        can_add = True
        for b in chosen:
            if (conflict_mask[a] >> b) & 1:
                can_add = False
                break
        if can_add:
            chosen.add(a)
    
    def exact_score(s):
        total = Fraction(0, 1)
        for a in s:
            total += Fraction(1, N - a)
        return total
    
    best_score = exact_score(chosen)
    best_chosen = set(chosen)
    
    # Local search with exact rational comparison
    unchosen = set(range(1, N)) - chosen
    
    def can_add(a, current_set):
        """Check if a can be added to current_set without conflict"""
        for b in current_set:
            if (conflict_mask[a] >> b) & 1:
                return False
        return True
    
    while time.time() < DEADLINE:
        improved = False
        
        # Try adding any unchosen number that doesn't conflict
        for a in sorted(unchosen, key=lambda x: N-x):  # higher value first
            if can_add(a, chosen):
                new_score = best_score + Fraction(1, N-a)
                if new_score > best_score:
                    chosen.add(a)
                    unchosen.remove(a)
                    best_score = new_score
                    best_chosen = set(chosen)
                    improved = True
        
        # Try 1-for-1 swaps: remove one chosen, add one unchosen
        chosen_list = list(chosen)
        for a in chosen_list:
            val_a = Fraction(1, N-a)
            # Try removing a
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding any unchosen that fits
            for b in sorted(unchosen, key=lambda x: N-x):
                if b == a:
                    continue
                if can_add(b, chosen):
                    new_score = best_score - val_a + Fraction(1, N-b)
                    if new_score > best_score:
                        chosen.add(b)
                        unchosen.remove(b)
                        best_score = new_score
                        best_chosen = set(chosen)
                        improved = True
                        break
            # Restore a if we didn't swap
            if a not in chosen:
                chosen.add(a)
                unchosen.discard(a)
        
        # Try 2-for-1 swaps: remove two chosen, add one unchosen
        chosen_list = list(chosen)
        for i in range(len(chosen_list)):
            if time.time() > DEADLINE:
                break
            a1 = chosen_list[i]
            for j in range(i+1, len(chosen_list)):
                a2 = chosen_list[j]
                val_a1 = Fraction(1, N-a1)
                val_a2 = Fraction(1, N-a2)
                
                chosen.discard(a1)
                chosen.discard(a2)
                unchosen.add(a1)
                unchosen.add(a2)
                
                # Try adding any unchosen that fits
                for b in sorted(unchosen, key=lambda x: N-x):
                    if b == a1 or b == a2:
                        continue
                    if can_add(b, chosen):
                        new_score = best_score - val_a1 - val_a2 + Fraction(1, N-b)
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
        chosen_list = list(chosen)
        for a in chosen_list:
            if time.time() > DEADLINE:
                break
            val_a = Fraction(1, N-a)
            chosen.remove(a)
            unchosen.add(a)
            
            # Try adding pairs of unchosen
            unchosen_list = [b for b in unchosen if b != a]
            unchosen_list.sort(key=lambda x: N-x)
            for i in range(len(unchosen_list)):
                if time.time() > DEADLINE:
                    break
                b1 = unchosen_list[i]
                if can_add(b1, chosen):
                    chosen.add(b1)
                    unchosen.remove(b1)
                    
                    # Try adding a second
                    for b2 in unchosen_list[i+1:]:
                        if b2 == a or b2 == b1:
                            continue
                        if can_add(b2, chosen):
                            new_score = best_score - val_a + Fraction(1, N-b1) + Fraction(1, N-b2)
                            if new_score > best_score:
                                chosen.add(b2)
                                unchosen.remove(b2)
                                best_score = new_score
                                best_chosen = set(chosen)
                                improved = True
                                break
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
