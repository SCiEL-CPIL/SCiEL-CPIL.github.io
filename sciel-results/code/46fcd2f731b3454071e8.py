# EVOLVE-BLOCK-START
import sys
import time
from math import gcd

def construct():
    n = 512
    start_time = time.time()
    timeout = 110.0  # Leave 10s buffer
    
    # 1. Precompute weights and coprime masks
    weights = [0.0] * n
    for i in range(1, n):
        weights[i] = 1.0 / (n - i)
    
    coprime_masks = [0] * n
    for i in range(1, n):
        mask = 0
        for j in range(1, n):
            if i != j and gcd(i, j) == 1:
                mask |= (1 << (j - 1))
        coprime_masks[i] = mask

    # 2. Greedy Initial Solution
    def greedy_solve():
        chosen = []
        cur_mask = (1 << 511) - 1
        score = 0.0
        
        while cur_mask:
            best_node = -1
            best_w = -1.0
            m = cur_mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                node = bit_idx + 1
                if weights[node] > best_w:
                    best_w = weights[node]
                    best_node = node
                m ^= lsb
                
            if best_node == -1:
                break
                
            chosen.append(best_node)
            score += weights[best_node]
            cur_mask &= coprime_masks[best_node]
            
        return score, chosen

    best_score, best_set = greedy_solve()
    
    # 3. Branch and Bound
    def calc_bound(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            node = bit_idx + 1
            s += weights[node]
            m ^= lsb
        return s

    def bb(current_score, current_mask, current_list):
        nonlocal best_score, best_set
        
        if time.time() - start_time > timeout:
            return
            
        bound = current_score + calc_bound(current_mask)
        if bound <= best_score:
            return
        
        if current_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set = current_list[:]
            return
        
        # Find next node: highest weight
        best_node = -1
        best_w = -1.0
        m = current_mask
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            node = bit_idx + 1
            if weights[node] > best_w:
                best_w = weights[node]
                best_node = node
            m ^= lsb
            
        if best_node == -1:
            return
            
        # Branch 1: Include
        new_mask_include = current_mask & coprime_masks[best_node]
        current_list.append(best_node)
        bb(current_score + weights[best_node], new_mask_include, current_list)
        current_list.pop()
        
        # Branch 2: Exclude
        bit_to_remove = 1 << (best_node - 1)
        new_mask_exclude = current_mask & ~bit_to_remove
        bb(current_score, new_mask_exclude, current_list)

    init_mask = (1 << 511) - 1
    bb(0.0, init_mask, [])
    
    return best_set

# EVOLVE-BLOCK-END
