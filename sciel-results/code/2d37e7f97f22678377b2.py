import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    N = 512
    max_a = N - 1
    
    # Pre-compute weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
    # Precompute coprime matrix
    # Using a list of sets of prime factors might be faster for checking?
    # Or just gcd. gcd is fast in C.
    # Let's stick to gcd for simplicity and correctness, but precompute a matrix if needed.
    # 511x511 matrix is ~260k entries. Small.
    coprime = [[False] * (max_a + 1) for _ in range(max_a + 1)]
    for a in range(1, max_a + 1):
        for b in range(a + 1, max_a + 1):
            coprime[a][b] = (gcd(a, b) == 1)
            coprime[b][a] = coprime[a][b]
            
    # Beam Search
    beam_width = 150
    beams = [(0.0, [])]
    
    # Iterate a from max_a down to 1
    for a in range(max_a, 0, -1):
        w = weights[a]
        new_beams = []
        
        for score, chosen in beams:
            # Option 1: Skip a
            new_beams.append((score, chosen))
            
            # Option 2: Include a
            # Check coprime
            can_include = True
            for b in chosen:
                if not coprime[a][b]:
                    can_include = False
                    break
            if can_include:
                new_beams.append((score + w, chosen + [a]))
                
        # Prune
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        
        # Time check
        if time.time() - start_time > deadline - safety_margin - 30: # Reserve 30s for polish
            break
            
    # Best beam
    best_beam = max(beams, key=lambda x: x[0])
    best_score = best_beam[0]
    best_chosen = list(best_beam[1])
    chosen_set = set(best_chosen)
    
    # Greedy Fill
    changed = True
    while changed:
        changed = False
        for a in range(max_a, 0, -1):
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
            
    # Local Search: Drop-and-Refill and 1-Opt
    improved = True
    while improved:
        improved = False
        # 1-Opt: Swap
        for i, a in enumerate(best_chosen):
            for b in range(max_a, 0, -1):
                if b in chosen_set:
                    continue
                # Check if b is coprime to all others
                can_add_b = True
                for j, c in enumerate(best_chosen):
                    if j == i:
                        continue
                    if not coprime[b][c]:
                        can_add_b = False
                        break
                if can_add_b and weights[b] > weights[a]:
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
            
        # Drop-and-Refill
        for a in list(best_chosen):
            # Remove a
            best_chosen.remove(a)
            chosen_set.remove(a)
            temp_score = best_score - weights[a]
            temp_chosen = list(best_chosen)
            temp_set = set(chosen_set)
            
            # Greedily refill
            for b in range(max_a, 0, -1):
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
                # Note: best_score is already correct if we didn't update it
                # But we need to ensure best_chosen is consistent.
                # Since we only update best_chosen if improved, and we restored a,
                # the set is back to original.
                
            if time.time() - start_time > deadline - safety_margin:
                break
        if time.time() - start_time > deadline - safety_margin:
            break
            
    return best_chosen
