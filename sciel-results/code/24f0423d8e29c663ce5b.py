"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict
import math

def construct():
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Generate all forbidden triples
    triples = []
    triple_set = set()
    
    for b_prime in range(1, 181):
        for c_prime in range(b_prime + 1, 181):
            if math.gcd(b_prime, c_prime) != 1:
                continue
            
            max_k = 180 // ((b_prime + c_prime) * b_prime)
            if max_k < 1:
                continue
                
            for k in range(1, max_k + 1):
                b = k * (b_prime + c_prime) * b_prime
                c = k * (b_prime + c_prime) * c_prime
                a = k * b_prime * c_prime
                
                if a <= N and b <= N and c <= N:
                    triple = tuple(sorted((a, b, c)))
                    if triple not in triple_set:
                        triple_set.add(triple)
                        triples.append(triple)
                        
    if not triples:
        return numbers

    # Map element to list of triple indices
    elem_to_triples = defaultdict(set)
    for i, t in enumerate(triples):
        for x in t:
            elem_to_triples[x].add(i)

    best_removed = set()
    best_removed_size = N
    
    def get_lower_bound(active_indices):
        if not active_indices:
            return 0
        used_elements = set()
        count = 0
        for idx in active_indices:
            t = triples[idx]
            if not (set(t) & used_elements):
                used_elements.update(t)
                count += 1
        return count

    class TimeoutException(Exception):
        pass

    def branch_and_bound(removed_set, active_set):
        nonlocal best_removed, best_removed_size
        
        if time.time() - start_time > 50:
            raise TimeoutException()
            
        lb = get_lower_bound(active_set)
        if len(removed_set) + lb >= best_removed_size:
            return
            
        if not active_set:
            if len(removed_set) < best_removed_size:
                best_removed_size = len(removed_set)
                best_removed = set(removed_set)
            return
            
        # Heuristic: Pick triple with element having highest frequency in active_set
        freq = defaultdict(int)
        for idx in active_set:
            for x in triples[idx]:
                freq[x] += 1
                
        max_freq = 0
        best_idx = next(iter(active_set))
        for idx in active_set:
            for x in triples[idx]:
                if freq[x] > max_freq:
                    max_freq = freq[x]
                    best_idx = idx
                    break

        a, b, c = triples[best_idx]
        
        branches = [(a, freq[a]), (b, freq[b]), (c, freq[c])]
        branches.sort(key=lambda x: x[1], reverse=True)
        
        for elem, _ in branches:
            new_removed = removed_set | {elem}
            to_remove = elem_to_triples[elem]
            new_active_set = active_set - to_remove
            branch_and_bound(new_removed, new_active_set)

    start_time = time.time()
    initial_active = set(range(len(triples)))
    
    try:
        branch_and_bound(set(), initial_active)
    except TimeoutException:
        pass

    result = [x for x in numbers if x not in best_removed]
    if not result:
        return [1]
    return sorted(result)

# EVOLVE-BLOCK-END
