import time
import random

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Candidates: We want to try large numbers first to keep sums large and sparse
    # The directive mentions "binary representation" and "smallest such integers".
    # This is ambiguous. I will stick to the effective heuristic: 
    # Try multiple orders, specifically favoring large numbers which reduce divisibility risk.
    
    candidates = list(range(1, 257))
    
    # Define several orders
    orders = [
        sorted(candidates, reverse=True), # Largest first
        sorted(candidates),               # Smallest first
    ]
    
    # Add some random shuffles biased towards large numbers?
    # Or just standard shuffles.
    for _ in range(100):
        random.shuffle(candidates)
        orders.append(candidates[:])
        
    for order in orders:
        if time.time() - start_time > 55:
            break
        
        current_set = []
        sums = set()
        
        for x in order:
            # Generate new sums
            # new_sums = {s + x for s in sums} U {x}
            # We need to check if adding x creates any divisibility conflict.
            
            # Optimization: 
            # If we add x, the new sums are S' = S U {s+x for s in S} U {x}.
            # We must ensure:
            # 1. No s in S' divides t in S' (s != t).
            
            # Let's construct the set of potential new sums and check against existing ones.
            # Actually, it's safer to just generate the full new set of sums and check all pairs?
            # No, that's $O(2^{2k})$.
            
            # Better: Check new sums against old sums, and new sums against each other.
            
            new_sums_list = [x] + [s + x for s in sums]
            new_sums_set = set(new_sums_list)
            
            valid = True
            
            # Check new vs old
            for s_new in new_sums_list:
                for s_old in sums:
                    if s_new == s_old:
                        # Equal sums from different subsets?
                        # The task says: "Two different subsets with the same sum still make the set invalid".
                        # If s_new == s_old, it means there are two subsets with same sum.
                        # One is the subset generating s_old (from old set).
                        # One is the subset generating s_new (involving x).
                        # This implies a collision.
                        valid = False
                        break
                    if s_new % s_old == 0 or s_old % s_new == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Check new vs new
            # We need to check pairs in new_sums_list.
            # To avoid O(N^2) for N=2^k, we can be smarter?
            # For k up to 8, N=255. 255^2/2 = 32k checks. Acceptable.
            for i in range(len(new_sums_list)):
                s1 = new_sums_list[i]
                for j in range(i + 1, len(new_sums_list)):
                    s2 = new_sums_list[j]
                    if s1 == s2:
                        valid = False
                        break
                    if s1 % s2 == 0 or s2 % s1 == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(x)
                sums.update(new_sums_set)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
