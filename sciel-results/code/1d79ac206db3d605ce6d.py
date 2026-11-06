# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (limit is 90s)
    
    N = 128
    nums = list(range(1, N + 1))
    n = N
    
    # Precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    # Precompute bad_triples_with[x] = list of (a, b) such that {a, b, x} is a bad triple.
    # a, b, x are 0-indexed (0..127).
    # A triple {a, b, x} is bad if lcm(a+1, b+1) == lcm(a+1, x+1) == lcm(b+1, x+1).
    
    bad_triples_with = [[] for _ in range(n)]
    
    for x in range(n):
        x_val = x + 1
        for a in range(n):
            if a == x:
                continue
            a_val = a + 1
            L_ax = get_lcm(a_val, x_val)
            for b in range(a + 1, n):
                if b == x:
                    continue
                b_val = b + 1
                L_ab = get_lcm(a_val, b_val)
                L_bx = get_lcm(b_val, x_val)
                if L_ax == L_ab == L_bx:
                    bad_triples_with[x].append((a, b))
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(x, mask):
        """
        Check if element x (0-indexed) can be added to the set represented by mask.
        mask is a bitmask where bit i is 1 if element i is in the set.
        Returns True if no bad triple is formed.
        """
        for a, b in bad_triples_with[x]:
            # Check if both a and b are in mask
            if (mask & (1 << a)) and (mask & (1 << b)):
                return False
        return True
    
    # Greedy initialization
    def greedy_solve(order):
        mask = 0
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, mask):
                mask |= (1 << x)
        return mask
    
    # Try multiple random and heuristic orderings to find a good initial solution
    best_mask = 0
    best_size = 0
    
    # Heuristic 1: Natural order
    m1 = greedy_solve(range(n))
    s1 = popcount(m1)
    if s1 > best_size:
        best_size = s1
        best_mask = m1
        
    # Heuristic 2: Reverse order
    m2 = greedy_solve(range(n-1, -1, -1))
    s2 = popcount(m2)
    if s2 > best_size:
        best_size = s2
        best_mask = m2
        
    # Heuristic 3: Sort by degree (number of bad triples involving x)
    degrees = [len(bad_triples_with[i]) for i in range(n)]
    order_deg = sorted(range(n), key=lambda i: degrees[i])
    m3 = greedy_solve(order_deg)
    s3 = popcount(m3)
    if s3 > best_size:
        best_size = s3
        best_mask = m3
        
    # Heuristic 4: Random restarts
    import random
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        mr = greedy_solve(random_order)
        sr = popcount(mr)
        if sr > best_size:
            best_size = sr
            best_mask = mr
            
    # Local Search: Try to improve best_mask by swapping elements
    # We will iterate: remove one element from best_mask, try to add any other element, keep if size increases.
    improved_mask = best_mask
    improved_size = best_size
    progress = True
    
    while progress and time.time() < deadline:
        progress = False
        current_list = [i for i in range(n) if (improved_mask >> i) & 1]
        
        for idx, rem_idx in enumerate(current_list):
            if time.time() > deadline:
                break
            
            # Remove rem_idx
            temp_mask = improved_mask & ~(1 << rem_idx)
            temp_size = improved_size - 1
            
            # Try to add any element not in temp_mask
            added_any = False
            for add_idx in range(n):
                if (temp_mask >> add_idx) & 1:
                    continue
                if can_add(add_idx, temp_mask):
                    new_mask = temp_mask | (1 << add_idx)
                    new_size = temp_size + 1
                    if new_size > improved_size:
                        improved_mask = new_mask
                        improved_size = new_size
                        progress = True
                        added_any = True
                        break # Restart inner loop with new improved_mask
            if added_any:
                # Restart outer loop because mask changed
                break
        
    # Convert mask to list of 1-indexed integers
    result = [i + 1 for i in range(n) if (improved_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
