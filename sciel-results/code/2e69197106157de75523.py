import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55
    
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)
    
    # Precompute masks
    masks = [0] * n_quads
    for i, q in enumerate(all_quads):
        m = 0
        for x in q:
            m |= (1 << (x - 1))
        masks[i] = m
        
    # Precompute Conf[i][j] is expensive to store fully.
    # We will compute it on the fly or cache it.
    # Given the small number of pairs in F, on-the-fly is fine.
    
    def get_conflicts(i, j):
        """
        Returns a list of indices k such that {i, j, k} form a sunflower.
        """
        M_i = masks[i]
        M_j = masks[j]
        K = M_i & M_j
        U = M_i | M_j
        V = (~U) & 0xFFFF
        
        pop_k = bin(K).count('1')
        size_p = 4 - pop_k
        
        # Generate all subsets P of V of size size_p
        # V has 16 - popcount(U) bits.
        # popcount(U) = 8 - pop_k.
        # So V has 8 + pop_k bits.
        
        confs = []
        # Iterate subsets of V
        sub = V
        while True:
            if bin(sub).count('1') == size_p:
                M_k = K | sub
                # Find index of M_k
                # We need a mapping from mask to index
                confs.append(M_k)
            if sub == 0:
                break
            sub = (sub - 1) & V
            
        return confs

    # Map mask to index for quick lookup
    mask_to_idx = {masks[i]: i for i in range(n_quads)}
    
    best_family = []
    best_size = 0
    
    num_restarts = 1000
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        F = []
        in_F = [False] * n_quads
        Forbidden = set()
        
        # Candidate pool
        candidates = list(range(n_quads))
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline:
                break
                
            if in_F[idx] or idx in Forbidden:
                continue
                
            # Check if adding idx is valid
            # It is valid if idx not in Forbidden
            # We just need to update Forbidden
            
            F.append(idx)
            in_F[idx] = True
            
            # Update Forbidden with conflicts of idx with all existing in F
            M_idx = masks[idx]
            for j in F:
                if j == idx:
                    continue
                M_j = masks[j]
                
                K = M_idx & M_j
                U = M_idx | M_j
                V = (~U) & 0xFFFF
                
                pop_k = bin(K).count('1')
                size_p = 4 - pop_k
                
                sub = V
                while True:
                    if bin(sub).count('1') == size_p:
                        M_k = K | sub
                        if M_k in mask_to_idx:
                            k_idx = mask_to_idx[M_k]
                            if not in_F[k_idx]:
                                Forbidden.add(k_idx)
                    if sub == 0:
                        break
                    sub = (sub - 1) & V
            
            if len(F) > best_size:
                best_size = len(F)
                best_family = F[:]
                
    result = [sorted(all_quads[i]) for i in best_family]
    return result
