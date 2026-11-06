import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Any element in the set (other than 255) must share a factor with 255
    # so must be divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i != 255 and (i % 3 == 0 or i % 5 == 0 or i % 17 == 0)]
    
    n = len(candidates)
    
    # Build adjacency using bitmasks for fast set operations
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    all_mask = (1 << n) - 1
    
    # Bron-Kerbosch with pivot to find maximum clique
    best_clique = []
    
    def bron_kerbosch(R, P_mask, X_mask):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if P_mask == 0 and X_mask == 0:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        if P_mask == 0:
            return
        
        # Pivot selection: choose vertex in P ∪ X with maximum degree in P
        # For each vertex in P ∪ X, count its neighbors in P
        union_px = P_mask | X_mask
        pivot = -1
        max_deg = -1
        temp = union_px
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            deg = bin(adj_mask[v] & P_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Candidates: vertices in P not adjacent to pivot
        if pivot >= 0:
            candidates_mask = P_mask & ~adj_mask[pivot]
        else:
            candidates_mask = P_mask
        
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            new_R = R + [v]
            new_P_mask = P_mask & adj_mask[v]
            new_X_mask = X_mask & adj_mask[v]
            bron_kerbosch(new_R, new_P_mask, new_X_mask)
            
            P_mask ^= lsb
            X_mask |= lsb
    
    # Use bitmask for P and X for efficiency
    P_mask = all_mask
    X_mask = 0
    bron_kerbosch([], P_mask, X_mask)
    
    # The clique among candidates, plus 255
    result = [candidates[i] for i in best_clique] + [255]
    
    return result
