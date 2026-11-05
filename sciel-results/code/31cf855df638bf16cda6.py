"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_triples():
    """Precompute all sets of 3 distinct elements {a,b,c} in [1,180] 
    such that 1/a = 1/b + 1/c for some permutation.
    Returns:
        triples: List of (a, b, c) tuples.
        element_to_triples: List where index i contains list of triple indices involving element i+1.
    """
    triples = []
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    triples.append((a, b, c))
    
    element_to_triples = [[] for _ in range(181)]
    for i, (a, b, c) in enumerate(triples):
        element_to_triples[a].append(i)
        element_to_triples[b].append(i)
        element_to_triples[c].append(i)
        
    return triples, element_to_triples

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    triples, element_to_triples = _precompute_triples()
    num_triples = len(triples)
    
    if num_triples == 0:
        return list(range(1, 181))
        
    # Simulated Annealing
    in_set = [True] * 181
    k_counts = [3] * num_triples
    current_violations = num_triples
    current_size = 180
    
    LAMBDA = 100.0
    
    def get_energy(size, violations):
        return -size + LAMBDA * violations
    
    current_energy = get_energy(current_size, current_violations)
    
    best_size = 0
    best_in_set = [False] * 181
    
    T = 50.0
    cooling = 0.995
    
    steps = 0
    max_steps = 200000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        x = random.randint(1, 180)
        if not in_set[x]:
            continue
            
        y = random.randint(1, 180)
        if in_set[y]:
            continue
            
        if x == y:
            continue
            
        delta_violations = 0
        for t_idx in element_to_triples[x]:
            k = k_counts[t_idx]
            if k == 3:
                delta_violations -= 1
            k_counts[t_idx] = k - 1
            
        for t_idx in element_to_triples[y]:
            k = k_counts[t_idx]
            if k == 2:
                delta_violations += 1
            k_counts[t_idx] = k + 1
            
        delta_size = 0
        delta_energy = LAMBDA * delta_violations
        
        if delta_energy < 0 or random.random() < pow(2.71828, -delta_energy / T):
            in_set[x] = False
            in_set[y] = True
            current_violations += delta_violations
            current_energy += delta_energy
            
            if current_violations == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set[:]
        else:
            for t_idx in element_to_triples[x]:
                k_counts[t_idx] += 1
            for t_idx in element_to_triples[y]:
                k_counts[t_idx] -= 1
            
        T *= cooling
        
    if best_size == 0:
        if current_violations > 0:
             while current_violations > 0:
                 found = False
                 for t_idx in range(num_triples):
                     if k_counts[t_idx] == 3:
                         a, b, c = triples[t_idx]
                         candidates = [z for z in (a,b,c) if in_set[z]]
                         if candidates:
                             rem = random.choice(candidates)
                             in_set[rem] = False
                             current_size -= 1
                             for t_idx2 in element_to_triples[rem]:
                                 k_counts[t_idx2] -= 1
                                 if k_counts[t_idx2] == 2:
                                     current_violations -= 1
                             found = True
                             break
                 if not found:
                     break
                 
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set[:]
            
    k_counts_final = [0] * num_triples
    for i in range(1, 181):
        if best_in_set[i]:
            for t_idx in element_to_triples[i]:
                k_counts_final[t_idx] += 1
                
    final_violations = sum(1 for k in k_counts_final if k == 3)
    final_size = sum(best_in_set[1:])
    
    if final_violations > 0:
        while final_violations > 0:
             found = False
             for t_idx in range(num_triples):
                 if k_counts_final[t_idx] == 3:
                     a, b, c = triples[t_idx]
                     candidates = [z for z in (a,b,c) if best_in_set[z]]
                     if candidates:
                         rem = random.choice(candidates)
                         best_in_set[rem] = False
                         final_size -= 1
                         for t_idx2 in element_to_triples[rem]:
                             k_counts_final[t_idx2] -= 1
                             if k_counts_final[t_idx2] == 2:
                                 final_violations -= 1
                         found = True
                         break
             if not found:
                 break
             
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for y in range(1, 181):
            if best_in_set[y]:
                continue
            creates_violation = False
            for t_idx in element_to_triples[y]:
                if k_counts_final[t_idx] == 2:
                    creates_violation = True
                    break
            
            if not creates_violation:
                best_in_set[y] = True
                final_size += 1
                for t_idx in element_to_triples[y]:
                    k_counts_final[t_idx] += 1
                improved = True
                break
                
    return [i for i in range(1, 181) if best_in_set[i]]

# EVOLVE-BLOCK-END
