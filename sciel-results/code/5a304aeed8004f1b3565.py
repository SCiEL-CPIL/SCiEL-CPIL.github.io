import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    
    def is_valid(S):
        """Check if set S satisfies: for all a<b<=c in S, a does NOT divide b+c."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                if (2 * b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if adding x to set S maintains the property."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: x is the smallest element 'a' in a triple
        # Need: no b <= c in S with x < b and x | (b+c)
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            if (2 * b) % x == 0:
                return False
            for j in range(i + 1, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element 'b' in a triple
        # Need: no a < x in S and c >= x in S with a | (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element 'c' in a triple
        # Need: no a < b in S with b <= x and a | (b+x)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy_from_largest(order):
        """Build a set by greedily adding elements in the given order."""
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S, max_swaps=200):
        """Try to improve S by swapping elements out/in."""
        S = set(S)
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            swaps_done += 1
            S_list = sorted(S)
            candidates_out = list(S_list)
            random.shuffle(candidates_out)
            candidates_in = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(candidates_in)
            
            for out_elem in candidates_out:
                if time.time() > deadline - 2.0:
                    return S
                S_test = S - {out_elem}
                for in_elem in candidates_in:
                    if can_add(S_test, in_elem):
                        S = S_test | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy from largest to smallest
    S1 = greedy_from_largest(range(n, 0, -1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)
    
    # Strategy 2: Greedy from smallest to largest
    S2 = greedy_from_largest(range(1, n + 1))
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = set(S2)
    
    # Strategy 3: Multiple random restarts with different orderings
    for restart in range(100):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        # Create a random ordering with bias toward larger elements
        order = list(range(1, n + 1))
        # Mix: put larger elements first with some randomness
        large_half = list(range(n // 2 + 1, n + 1))
        small_half = list(range(1, n // 2 + 1))
        rng.shuffle(large_half)
        rng.shuffle(small_half)
        
        # Different mixing strategies
        strategy = restart % 4
        if strategy == 0:
            order = large_half + small_half
        elif strategy == 1:
            order = small_half + large_half
        elif strategy == 2:
            # Interleave
            order = []
            li, si = 0, 0
            while li < len(large_half) or si < len(small_half):
                if li < len(large_half) and (si >= len(small_half) or rng.random() < 0.6):
                    order.append(large_half[li])
                    li += 1
                else:
                    order.append(small_half[si])
                    si += 1
        else:
            rng.shuffle(order)
        
        S = greedy_from_largest(order)
        
        # Local search to improve
        if len(S) >= best_size - 5:
            S_improved = local_search(S, max_swaps=50)
            S = S_improved
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 4: Start from upper tail and expand
    for k in [128, 160, 192, 200, 210]:
        if time.time() > deadline - 3.0:
            break
        
        S = set(range(k, n + 1))
        if not is_valid(S):
            # Fix by removing problematic elements
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        # Try to add smaller elements
        candidates = [x for x in range(1, k) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Local search
        S = local_search(S, max_swaps=30)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final verification and return
    if not best_set:
        best_set = set(range(192, n + 1))
    
    # Verify the final set
    if not is_valid(best_set):
        # Fallback: rebuild from scratch
        best_set = greedy_from_largest(range(n, 0, -1))
    
    return sorted(best_set)
