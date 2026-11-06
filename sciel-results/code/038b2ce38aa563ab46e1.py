import sys
from functools import lru_cache

def construct():
    n = 256
    # We search for the maximum size subset of [1, n]
    # using Branch and Bound.
    # State: (current_index, current_set_tuple)
    # To make it fast, we only consider adding elements in descending order.
    # We maintain the set as a tuple or bitmask.
    # Given n=256, bitmask is 256 bits. Python int handles this.
    
    best_size = 0
    best_set = 0  # bitmask
    
    # Precompute sums? No, check on the fly.
    # To check if x can be added to set S:
    # For all b, c in S with x < b <= c, x must not divide b+c.
    
    def can_add(x, S):
        # S is a bitmask of elements already in the set (all > x)
        # Iterate through bits in S
        # Since S only contains elements > x, we just check all pairs.
        # Optimization: if S has fewer than 2 elements, True.
        # We need to check if there exist b, c in S such that b+c is multiple of x.
        # b, c > x.
        # Multiples of x: 3x, 4x, ... up to 2n.
        # For each multiple m, check if m = b+c for some b,c in S.
        
        # Get list of elements in S
        # This is slow if done every time.
        # Alternative: Precompute for each x, the set of pairs (b,c) that conflict.
        # But S changes.
        
        # Heuristic: Just check multiples.
        # Max sum is 2n = 512.
        # Min sum is 2(x+1).
        # Multiples of x in range:
        start_k = (2*(x+1) + x - 1) // x + 1 # ceil(2(x+1)/x) + 1?
        # Actually, we need b+c > 2x.
        # Smallest multiple > 2x is 3x.
        for k in range(3, 2*n//x + 1):
            m = k * x
            if m > 2*n:
                break
            # Check if m can be formed by two elements in S
            # b + c = m, b < c (or b<=c), b,c in S
            # b < m/2
            # Iterate b from max(x+1, m-256) to m//2
            # If b in S and (m-b) in S, then conflict.
            
            # Optimize:
            # We can iterate b in S?
            # If S is sparse, iterating S is better.
            # If S is dense, iterating range is better.
            
            # Let's iterate b in the possible range.
            min_b = max(x + 1, m - n)
            max_b = m // 2
            if min_b > max_b:
                continue
            
            # Check if any b in [min_b, max_b] is in S and m-b is in S
            # Since S is a bitmask, we can extract bits.
            # But checking each b is O(n).
            # Total O(n * n) per add? Too slow.
            
            pass
        return True

    # This naive approach is too slow for B&B.
    # We need a faster conflict check.
    
    # Alternative:
    # The set S is always a suffix [K, n] in the greedy path.
    # If we assume the optimal set is a suffix, we just need to find K.
    # But it might not be a suffix.
    
    # Given the time limit, I will implement a greedy algorithm that mimics B&B
    # (trying to add, if fails, skip) and hope it finds 86.
    # If the B&B logic is too complex, I will simplify to a greedy repair.
    
    S = set()
    # Add from n down to 1
    for x in range(n, 0, -1):
        # Check if x can be added to S
        # S contains elements > x
        conflict = False
        # Check multiples of x
        # b, c in S
        # Since S is built from large numbers, S is likely dense at the top.
        
        # To speed up, we can store S in a list and use a set for O(1) lookup.
        # But we need to check pairs.
        
        # If |S| is small, check all pairs.
        # If |S| is large, check multiples.
        
        # Let's use the multiple check.
        for k in range(3, 513 // x + 1):
            m = k * x
            if m > 512:
                break
            # Check if m = b+c for b,c in S
            # b < c, b >= x+1
            # b in [max(x+1, m-256), m//2]
            min_b = max(x + 1, m - 256)
            max_b = m // 2
            if min_b > max_b:
                continue
            
            # Check if any b in this range is in S
            # And m-b is in S
            # We can iterate b in this range
            # Range size is at most 256
            for b in range(min_b, max_b + 1):
                if b in S and (m - b) in S:
                    conflict = True
                    break
            if conflict:
                break
        
        if not conflict:
            S.add(x)
            
    return list(S)

# EVOLVE-BLOCK-END
