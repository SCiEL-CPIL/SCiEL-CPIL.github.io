import math
import time
import sys

def construct():
    sys.setrecursionlimit(10000)
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    orig_masks = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            b = j + 1
            g = math.gcd(a, b)
            if (a // g) * b <= 512:
                m |= (1 << j)
        orig_masks[i] = m
        
    degrees = [bin(m).count('1') for m in orig_masks]
    sorted_indices = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    new_masks = [0] * N
    for new_i in range(N):
        orig_i = sorted_indices[new_i]
        m = 0
        orig_m = orig_masks[orig_i]
        for new_j in range(N):
            orig_j = sorted_indices[new_j]
            if (orig_m >> orig_j) & 1:
                m |= (1 << new_j)
        new_masks[new_i] = m

    # Greedy
    candidates = (1 << N) - 1
    greedy_clique = []
    while candidates:
        v = (candidates & -candidates).bit_length() - 1
        greedy_clique.append(v)
        candidates &= new_masks[v]
        
    best_size = len(greedy_clique)
    best_clique = list(greedy_clique)

    # Helper for popcount
    if hasattr(int, 'bit_count'):
        def bit_count(x):
            return x.bit_count()
    else:
        def bit_count(x):
            return bin(x).count('1')
            
    def search(candidates, current_clique):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if len(current_clique) + bit_count(candidates) <= best_size:
            return
            
        if candidates == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
            
        v = (candidates & -candidates).bit_length() - 1
        
        current_clique.append(v)
        search(candidates & new_masks[v], current_clique)
        current_clique.pop()
        
        search(candidates & ~(1 << v), current_clique)

    initial_candidates = (1 << N) - 1
    search(initial_candidates, [])
    
    result = [sorted_indices[i] + 1 for i in best_clique]
    
    if not result:
        return [1]
        
    return result
