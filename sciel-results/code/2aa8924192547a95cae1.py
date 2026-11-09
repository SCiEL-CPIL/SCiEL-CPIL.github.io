"""Construct a minimum maximal no-isosceles set on an N x N grid."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time

def _precompute_isosceles(n):
    """
    Precompute isosceles conflicts.
    Iso[a][b] is a bitmask of points c such that {a, b, c} form an isosceles triangle.
    """
    n2 = n * n
    points = [(x, y) for x in range(n) for y in range(n)]
    
    # Squared distances
    dist2 = np.zeros((n2, n2), dtype=np.int32)
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            dist2[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2

    # Iso[a][b] will be an integer bitmask
    Iso = [[0] * n2 for _ in range(n2)]
    
    # Iterate over all triples i < j < k
    for i in range(n2):
        for j in range(i + 1, n2):
            d_ij = dist2[i, j]
            if d_ij == 0:
                continue
            for k in range(j + 1, n2):
                d_ik = dist2[i, k]
                d_jk = dist2[j, k]
                
                # Check if isosceles
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    # Set bits for all permutations
                    Iso[i][j] |= (1 << k)
                    Iso[j][i] |= (1 << k)
                    
                    Iso[i][k] |= (1 << j)
                    Iso[k][i] |= (1 << j)
                    
                    Iso[j][k] |= (1 << i)
                    Iso[k][j] |= (1 << i)
                    
                    # Self bits? 
                    # If {a,b,c} is isosceles, does a form isosceles with itself?
                    # The problem implies distinct points. 
                    # However, if we include self-bits, it might interfere with validity check.
                    # Validity: ConflictMask & S_mask == 0.
                    # If Iso[a][b] has bit a set, and a is in S, then ConflictMask has bit a.
                    # S_mask has bit a. Intersection non-zero. Invalid.
                    # But {a,b,a} is not a triangle. 
                    # We should NOT set self bits in Iso.
                    # The loop i<j<k ensures distinct i,j,k.
                    # So Iso[i][j] will only have bits for k != i, k != j.
                    # This is correct.
    
    return points, Iso

def construct():
    n = N
    n2 = n * n
    points, Iso = _precompute_isosceles(n)
    
    AllMask = (1 << n2) - 1
    
    def mask_to_list(mask):
        return [i for i in range(n2) if (mask >> i) & 1]
    
    def list_to_mask(lst):
        mask = 0
        for i in lst:
            mask |= (1 << i)
        return mask
    
    def get_metrics(S_list):
        """
        Returns:
        valid: bool
        maximal: bool
        size: int
        unblocked_mask: int
        """
        if not S_list:
            return True, False, 0, AllMask
        
        S_mask = list_to_mask(S_list)
        ConflictMask = 0
        
        # Calculate ConflictMask
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                ConflictMask |= Iso[a][b]
        
        # Check Validity: No point in S should be in ConflictMask
        # (ConflictMask contains points c that form isosceles with some pair a,b in S)
        # If c is in S, then {a,b,c} is isosceles.
        valid = ((ConflictMask & S_mask) == 0)
        
        # Check Maximality: All points not in S must be in ConflictMask
        # Unblocked points are those not in S and not in ConflictMask
        Unblocked = AllMask & ~S_mask & ~ConflictMask
        maximal = (Unblocked == 0)
        
        return valid, maximal, len(S_list), Unblocked

    # Greedy initialization
    def greedy_init():
        S = []
        available = list(range(n2))
        random.shuffle(available)
        for p in available:
            # Check if p can be added to S
            # p is valid if p is not in ConflictMask of current S
            # And adding p doesn't create isosceles with existing pairs?
            # Actually, if p is not in ConflictMask(S), it means no pair in S forms isosceles with p.
            # So p can be added.
            if not S:
                S.append(p)
                continue
            
            S_mask = list_to_mask(S)
            ConflictMask = 0
            L = len(S)
            for i in range(L):
                a = S[i]
                for j in range(i + 1, L):
                    b = S[j]
                    ConflictMask |= Iso[a][b]
            
            if (ConflictMask >> p) & 1:
                continue # p is blocked/conflicting
            else:
                S.append(p)
                
        # Ensure maximality? Greedy adds until no more can be added, so it is maximal.
        # But we should verify.
        valid, maximal, size, _ = get_metrics(S)
        return S

    S = greedy_init()
    
    best_S = list(S)
    best_size = len(S)
    
    # Simulated Annealing / Local Search
    rng = random.Random(42)
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_S = list(S)
    current_valid, current_maximal, current_size, current_unblocked = get_metrics(current_S)
    
    # We want to minimize size, subject to valid and maximal.
    # Cost function: size * 100 + (unblocked_count * 10 if not maximal else 0)
    # Actually, if we allow non-maximal states, we need a penalty.
    # Let's define cost = size + penalty * unblocked_count
    # We want to find a state with unblocked_count=0 and minimal size.
    
    def get_cost(S_list):
        v, m, sz, ub = get_metrics(S_list)
        if not v:
            return float('inf')
        ub_count = bin(ub).count('1')
        return sz + 10.0 * ub_count
        
    current_cost = get_cost(current_S)
    
    iterations = 50000
    for it in range(iterations):
        if T < T_min:
            break
            
        # Choose move
        if rng.random() < 0.5:
            # Remove
            if not current_S:
                continue
            idx = rng.randint(0, len(current_S) - 1)
            p = current_S[idx]
            new_S = current_S[:idx] + current_S[idx+1:]
        else:
            # Add
            # Pick a random point not in S
            # To make it faster, pick from unblocked or random
            if len(current_S) == n2:
                continue
            # Find available points
            # We can just pick random 1..n2 and check if in S
            # Or pick from unblocked to help maximality
            # Let's pick random
            p = rng.randint(0, n2 - 1)
            if p in current_S:
                continue
            new_S = current_S + [p]
        
        new_valid, new_maximal, new_size, new_unblocked = get_metrics(new_S)
        if not new_valid:
            continue
            
        new_cost = new_size + 10.0 * bin(new_unblocked).count('1')
        
        delta = new_cost - current_cost
        
        if delta < 0:
            current_S = new_S
            current_cost = new_cost
            current_valid, current_maximal, current_size, current_unblocked = new_valid, new_maximal, new_size, new_unblocked
            if new_maximal and new_size < best_size:
                best_S = list(new_S)
                best_size = new_size
        else:
            # Accept with probability
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                current_S = new_S
                current_cost = new_cost
                current_valid, current_maximal, current_size, current_unblocked = new_valid, new_maximal, new_size, new_unblocked
                if new_maximal and new_size < best_size:
                    best_S = list(new_S)
                    best_size = new_size
                    
        T *= cooling
        
        # Optional: If we found a very small set, we can restart or just continue
        # To improve convergence, we can periodically reset to best_S if stuck
        if it % 1000 == 0 and current_cost > get_cost(best_S):
             # Soft reset
             if rng.random() < 0.1:
                 current_S = list(best_S)
                 current_cost = get_cost(current_S)
                 current_valid, current_maximal, current_size, current_unblocked = get_metrics(current_S)

    # Final check
    v, m, sz, _ = get_metrics(best_S)
    if not m:
        # Fallback to a valid maximal set if best_S is not maximal (should not happen if we only update on maximal)
        # But greedy_init returns maximal.
        pass
        
    result = [points[i] for i in best_S]
    return result

# EVOLVE-BLOCK-END
