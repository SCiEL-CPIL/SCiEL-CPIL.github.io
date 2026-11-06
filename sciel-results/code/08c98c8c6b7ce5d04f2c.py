import time
import sys
import random
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute conflict: conflict[i][j] is True if lcm(vertices[i], vertices[j]) <= N
    # We'll store as bitmasks: conflict[i] has bit j set if i and j conflict
    conflict = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
                
    weights = [1.0 / v for v in vertices]
    
    # Order vertices by weight descending (smaller v first for higher weight)
    order = list(range(n))
    
    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            s += weights[idx]
            m ^= lsb
        return s
    
    def is_valid_add(mask, v):
        # Check if v can be added to mask
        return (mask & conflict[v]) == 0
    
    def greedy_init(order_list, mask_start=0):
        """Greedy add vertices in given order to mask_start"""
        mask = mask_start
        for v in order_list:
            if not (mask & (1 << v)) and is_valid_add(mask, v):
                mask |= (1 << v)
        return mask
    
    def local_search(mask, val):
        """Local search with remove+refill and swap moves"""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try remove and refill
            m = mask
            while m:
                lsb = m & -m
                u_idx = (lsb.bit_length() - 1)
                m ^= lsb
                
                S_candidate = mask & ~lsb
                val_candidate = val - weights[u_idx]
                
                # Greedy refill
                new_mask = S_candidate
                new_val = val_candidate
                for v in order:
                    if new_mask & (1 << v):
                        continue
                    if is_valid_add(new_mask, v):
                        new_mask |= (1 << v)
                        new_val += weights[v]
                
                if new_val > val:
                    val = new_val
                    mask = new_mask
                    improved = True
                    break
            
            # If no improvement from remove, try swap moves
            if not improved and time.time() < deadline:
                # Try swapping each vertex in mask with each vertex not in mask
                m_in = mask
                while m_in and time.time() < deadline:
                    lsb_in = m_in & -m_in
                    u_idx = (lsb_in.bit_length() - 1)
                    m_in ^= lsb_in
                    
                    # Try removing u and adding various w
                    S_no_u = mask & ~lsb_in
                    val_no_u = val - weights[u_idx]
                    
                    # Find best w to add
                    best_w_val = val_no_u
                    best_w_mask = S_no_u
                    best_w_idx = -1
                    
                    m_not = (~mask) & ((1 << n) - 1)
                    while m_not and time.time() < deadline:
                        lsb_not = m_not & -m_not
                        w_idx = (lsb_not.bit_length() - 1)
                        m_not ^= lsb_not
                        
                        if is_valid_add(S_no_u, w_idx):
                            candidate_val = val_no_u + weights[w_idx]
                            if candidate_val > best_w_val:
                                best_w_val = candidate_val
                                best_w_mask = S_no_u | lsb_not
                                best_w_idx = w_idx
                    
                    if best_w_val > val:
                        val = best_w_val
                        mask = best_w_mask
                        improved = True
                        break
        
        return mask, val
    
    # Multiple restarts with different orderings
    best_mask = 0
    best_val = 0.0
    
    # First: standard greedy
    mask = greedy_init(order)
    val = get_value(mask)
    if val > best_val:
        best_val = val
        best_mask = mask
    
    # Local search on the greedy result
    mask, val = local_search(best_mask, best_val)
    if val > best_val:
        best_val = val
        best_mask = mask
    
    # Try different orderings for more restarts
    random.seed(42)
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        # Create a shuffled order, but bias towards smaller numbers
        # Use a random permutation
        shuffled = list(range(n))
        random.shuffle(shuffled)
        
        mask = greedy_init(shuffled)
        val = get_value(mask)
        
        # Local search
        mask, val = local_search(mask, val)
        
        if val > best_val:
            best_val = val
            best_mask = mask
    
    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = (lsb.bit_length() - 1)
        result.append(vertices[idx])
        m ^= lsb
    
    if not result:
        return [2]
        
    return result
