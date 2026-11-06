import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    N = 512
    max_a = N - 1
    
    # Pre-compute weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
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
    
    pfactors = [set() for _ in range(max_a + 1)]
    for a in range(1, max_a + 1):
        pfactors[a] = prime_factors(a)
    
    # Precompute coprime matrix using prime factor intersection
    coprime = [[False] * (max_a + 1) for _ in range(max_a + 1)]
    for a in range(1, max_a + 1):
        coprime[a][a] = False
        for b in range(a + 1, max_a + 1):
            coprime[a][b] = (len(pfactors[a] & pfactors[b]) == 0)
            coprime[b][a] = coprime[a][b]
            
    # Beam Search - process in descending weight order (which is ascending a order)
    beam_width = 300
    beams = [(0.0, frozenset())]
    
    # Process numbers from highest weight (smallest a) to lowest weight (largest a)
    # But we want to try including high-weight items first
    # Actually, let's process in descending weight: a=1 has weight 1/511, a=511 has weight 1/1
    # So process a from 1 to 511 (ascending) to get high weights first
    for a in range(1, max_a + 1):
        w = weights[a]
        new_beams = []
        
        for score, chosen in beams:
            # Option 1: Skip a
            new_beams.append((score, chosen))
            
            # Option 2: Include a
            can_include = True
            for b in chosen:
                if not coprime[a][b]:
                    can_include = False
                    break
            if can_include:
                new_beams.append((score + w, chosen | {a}))
                
        # Prune - keep top beam_width
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        
        # Time check
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin - 60:
            break
            
    # Best beam
    best_beam = max(beams, key=lambda x: x[0])
    best_score = best_beam[0]
    best_chosen = list(best_beam[1])
    chosen_set = set(best_chosen)
    
    # Greedy Fill - add any coprime number that improves score
    changed = True
    while changed:
        changed = False
        for a in range(1, max_a + 1):  # ascending = highest weight first
            if a in chosen_set:
                continue
            can_add = True
            for b in best_chosen:
                if not coprime[a][b]:
                    can_add = False
                    break
            if can_add:
                best_chosen.append(a)
                chosen_set.add(a)
                best_score += weights[a]
                changed = True
        if time.time() - start_time > deadline - safety_margin:
            break
            
    # Local Search: 1-Opt Swaps (replace one element with a better one)
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(best_chosen):
            for b in range(1, max_a + 1):
                if b in chosen_set:
                    continue
                if weights[b] <= weights[a]:
                    continue
                # Check if b is coprime to all others (excluding a)
                can_add_b = True
                for j, c in enumerate(best_chosen):
                    if j == i:
                        continue
                    if not coprime[b][c]:
                        can_add_b = False
                        break
                if can_add_b:
                    # Swap
                    best_chosen[i] = b
                    chosen_set.remove(a)
                    chosen_set.add(b)
                    best_score += weights[b] - weights[a]
                    improved = True
                    break
            if time.time() - start_time > deadline - safety_margin:
                break
        if time.time() - start_time > deadline - safety_margin:
            break
            
    # 2-Opt: Swap two elements with two better ones
    improved = True
    while improved:
        improved = False
        n = len(best_chosen)
        for i in range(n):
            for j in range(i + 1, n):
                a1, a2 = best_chosen[i], best_chosen[j]
                for b1 in range(1, max_a + 1):
                    if b1 in chosen_set:
                        continue
                    if weights[b1] <= weights[a1]:
                        continue
                    for b2 in range(1, max_a + 1):
                        if b2 in chosen_set or b2 == b1:
                            continue
                        if weights[b2] <= weights[a2]:
                            continue
                        # Check b1 and b2 are coprime to all others (excluding a1, a2)
                        # and to each other
                        if not coprime[b1][b2]:
                            continue
                        can_add = True
                        for k, c in enumerate(best_chosen):
                            if k == i or k == j:
                                continue
                            if not coprime[b1][c] or not coprime[b2][c]:
                                can_add = False
                                break
                        if can_add:
                            # Swap
                            best_chosen[i] = b1
                            best_chosen[j] = b2
                            chosen_set.discard(a1)
                            chosen_set.discard(a2)
                            chosen_set.add(b1)
                            chosen_set.add(b2)
                            best_score += weights[b1] + weights[b2] - weights[a1] - weights[a2]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if time.time() - start_time > deadline - safety_margin:
                break
        if time.time() - start_time > deadline - safety_margin:
            break
            
    # Drop-and-Refill
    improved = True
    while improved:
        improved = False
        for a in list(best_chosen):
            # Remove a
            best_chosen.remove(a)
            chosen_set.remove(a)
            temp_score = best_score - weights[a]
            temp_chosen = list(best_chosen)
            temp_set = set(chosen_set)
            
            # Greedily refill in weight order
            for b in range(1, max_a + 1):
                if b in temp_set:
                    continue
                can_add = True
                for c in temp_chosen:
                    if not coprime[b][c]:
                        can_add = False
                        break
                if can_add:
                    temp_chosen.append(b)
                    temp_set.add(b)
                    temp_score += weights[b]
                    
            if temp_score > best_score + 1e-12:
                best_chosen = temp_chosen
                chosen_set = temp_set
                best_score = temp_score
                improved = True
            else:
                # Restore
                best_chosen.append(a)
                chosen_set.add(a)
                
            if time.time() - start_time > deadline - safety_margin:
                break
        if time.time() - start_time > deadline - safety_margin:
            break
            
    return best_chosen
