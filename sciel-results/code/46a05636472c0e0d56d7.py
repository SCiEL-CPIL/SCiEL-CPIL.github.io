import time
import random

def construct():
    start = time.time()
    deadline = start + 88.0
    n = 256
    
    def full_validate(S_list):
        S_set = set(S_list)
        n2 = len(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(a, S_set):
        """
        Check if a can be added to S_set.
        Assumes S_set contains only elements > a? No, S_set can contain anything, 
        but the condition is a < b <= c. 
        So we only care about b, c in S_set such that b > a.
        """
        # We need to check if there exist b, c in S_set 
        # such that a < b <= c and (b+c) % a == 0.
        
        # Iterate over multiples of a
        # b+c can be at most 2*n
        # b+c must be > 2*a (since b>a, c>=b -> b+c > 2a)
        # So smallest multiple is 3*a (if 3a <= 2n)
        
        max_sum = 2 * n
        # k*a > 2a => k >= 3
        for k in range(3, max_sum // a + 1):
            m = k * a
            # We need b in S_set, c in S_set
            # b <= c
            # b + c = m
            # b > a
            # b <= m // 2
            # c = m - b
            # c >= b
            # c <= n
            
            # b ranges from max(a+1, m-n) to m//2
            b_min = max(a + 1, m - n)
            b_max = m // 2
            
            # Check if any b in [b_min, b_max] is in S_set AND (m-b) is in S_set
            # To speed up, we can just iterate b in S_set? 
            # Or iterate b in range?
            # Range is at most n/2 = 128.
            # Iterating range is O(128).
            # Iterating S_set is O(|S|).
            # |S| is around 100.
            # Range is bounded by m//2 - b_min.
            
            for b in range(b_min, b_max + 1):
                if b in S_set:
                    c = m - b
                    if c in S_set:
                        return False
        return True

    def greedy_solve(order):
        S_set = set()
        # S_list not strictly needed for can_add if we use set
        for a in order:
            if can_add(a, S_set):
                S_set.add(a)
        return S_set

    best_S = set(range(171, n + 1))
    best_size = len(best_S)
    
    # 1. Try the standard upper tail
    # 2. Try multi-start greedy with random orders
    
    # Pre-generate a few fixed good orders?
    # Descending order is a good heuristic.
    order_desc = list(range(n, 0, -1))
    S_desc = greedy_solve(order_desc)
    if len(S_desc) > best_size:
        best_S = S_desc
        best_size = len(S_desc)
        
    # Ascending order (likely bad, but check)
    order_asc = list(range(1, n + 1))
    S_asc = greedy_solve(order_asc)
    if len(S_asc) > best_size:
        best_S = S_asc
        best_size = len(S_asc)

    # Randomized Greedy Restarts
    # Focus on the range where most elements will be picked, e.g., [50, 256]
    # But we must consider all 1..256 as candidates.
    
    while time.time() < deadline - 5:
        # Random permutation
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_candidate = greedy_solve(candidates)
        
        if len(S_candidate) > best_size:
            best_S = S_candidate
            best_size = len(S_candidate)
            
            # Optional: Local search on the new best
            # Try to add any remaining elements
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for a in range(1, n + 1):
                    if a not in best_S:
                        if can_add(a, best_S):
                            best_S.add(a)
                            best_size += 1
                            changed = True
                            break # Restart loop to check new state
                            
    # Final validation
    S_final = sorted(best_S)
    if len(S_final) > 5 and not full_validate(S_final):
        # Fallback to known good set
        S_final = list(range(171, n + 1))
        
    return S_final
