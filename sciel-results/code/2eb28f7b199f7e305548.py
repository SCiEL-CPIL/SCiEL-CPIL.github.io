import time
import random

def construct():
    n = 256
    start_time = time.time()
    # 90s limit, keep 5s buffer for overhead/safety
    deadline = start_time + 85.0 

    def is_valid_addition(x, S_list):
        """
        Check if adding x to the sorted list S_list maintains the property:
        For every a < b <= c in the resulting set, a does NOT divide b+c.
        S_list contains elements of S, sorted ascending.
        x is not in S_list.
        """
        s_len = len(S_list)
        
        # We need to check three cases based on the position of x in the triple (a, b, c)
        # where a < b <= c.
        
        # Case 1: x is 'a'
        # Triple: (x, b, c) with x < b <= c.
        # b, c are from S_list.
        # Condition: x does NOT divide (b+c).
        # Since x is the smallest, b and c must be > x.
        # We iterate over pairs (b, c) from S_list where b <= c and b > x.
        
        # Find index of first element > x
        # Since S_list is sorted, we can use bisect or just loop.
        # For efficiency in a tight loop, simple loop with break is often fine for n=256.
        
        # Optimization: If x is very small, there are many pairs.
        # If x is large, there are few pairs.
        
        # Let's find indices i, j such that S_list[i] > x and S_list[j] >= S_list[i].
        # Actually, just iterate all b in S_list where b > x.
        # For each b, iterate c in S_list where c >= b.
        
        # To speed up, we can pre-filter S_list to elements > x?
        # But S_list changes.
        
        # Let's just loop.
        # Note: S_list is sorted.
        
        # Case 1: x is a
        # Find start index for b > x
        start_b = 0
        while start_b < s_len and S_list[start_b] <= x:
            start_b += 1
            
        for i in range(start_b, s_len):
            b = S_list[i]
            # b > x
            for j in range(i, s_len):
                c = S_list[j]
                # c >= b is guaranteed
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b'
        # Triple: (a, x, c) with a < x <= c.
        # a, c from S_list.
        # Condition: a does NOT divide (x+c).
        # a must be < x.
        # c must be >= x.
        
        # Subcase 2a: c = x?
        # The triple is (a, x, x). This requires x to be in the set?
        # No, the triple consists of values from the set.
        # If the set contains a, x, x... wait, sets have unique elements.
        # The problem statement says "For every a<b<=c in the set".
        # This implies a, b, c are elements of the set.
        # If b=c=x, then x is an element.
        # So we must check triples where b=x and c=x.
        # This requires checking if any a < x in S divides (x+x) = 2x.
        
        # Subcase 2b: c > x.
        # Triple (a, x, c) with a < x < c.
        # Check if a divides (x+c).
        
        # Find indices for a < x
        end_a = 0
        while end_a < s_len and S_list[end_a] < x:
            end_a += 1
            
        # a is in S_list[0 : end_a]
        
        # c must be >= x.
        # Since x is not in S_list, c must be > x.
        # So c is in S_list[start_b : s_len] (same range as b in Case 1, since b > x)
        
        # Check (a, x, x)
        for i in range(end_a):
            a = S_list[i]
            if (2 * x) % a == 0:
                return False
        
        # Check (a, x, c) with c > x
        for i in range(end_a):
            a = S_list[i]
            for j in range(start_b, s_len):
                c = S_list[j]
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c'
        # Triple: (a, b, x) with a < b <= x.
        # a, b from S_list.
        # Condition: a does NOT divide (b+x).
        # a < b.
        # b <= x.
        
        # b must be < x (since x not in S).
        # So b is in S_list[0 : start_b] (elements < x).
        # a must be < b.
        
        # Iterate b in S_list where b < x.
        for j in range(start_b):
            b = S_list[j]
            # b < x
            # a must be < b.
            # a is in S_list[0 : index_of_b]
            # Find index of b
            # Since sorted, we can just loop a from 0 to j-1
            
            for i in range(j):
                a = S_list[i]
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy(order):
        S = set()
        S_list = []
        for x in order:
            if time.time() > deadline:
                break
            # Check validity
            if is_valid_addition(x, S_list):
                S.add(x)
                # Maintain sorted list for efficiency
                # Simple insertion
                # Or just sort at the end? 
                # is_valid_addition requires sorted list.
                # Insertion into sorted list is O(N). N <= 256.
                # Binary search for position
                lo, hi = 0, len(S_list)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if S_list[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                S_list.insert(lo, x)
        return S, S_list

    best_S = set()
    best_S_list = []
    
    # 1. Descending order (Primary strategy)
    # This tends to pack large numbers which are "inert" to divisibility by smaller numbers
    # (if small numbers are not present) or have fewer multiples.
    S_desc, S_list_desc = greedy(range(n, 0, -1))
    if len(S_desc) > len(best_S):
        best_S = S_desc
        best_S_list = S_list_desc

    # 2. Ascending order
    if time.time() < deadline:
        S_asc, S_list_asc = greedy(range(1, n + 1))
        if len(S_asc) > len(best_S):
            best_S = S_asc
            best_S_list = S_list_asc

    # 3. Randomized Greedy Restarts
    # Use the remaining time to try many random orderings.
    # Each greedy pass is O(N^3) worst case for validation, but usually much faster.
    # N=256. N^3 = 16M. 
    # With optimizations, it's likely O(N^2) or less per step on average.
    # Let's estimate: 256 steps. Validation is O(K^2) where K is current size.
    # Total work ~ sum(K^2) ~ N^3/3.
    # If we can do ~100-500 restarts in 60s, that's good.
    
    # Let's also try "Block" strategies:
    # E.g., even numbers, odd numbers, primes?
    # But random shuffles cover a lot of space.
    
    rng = random.Random(42)
    
    # We will run random restarts until deadline
    restart_count = 0
    while time.time() < deadline - 2.0: # Keep 2s for final polish
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        S_rand, S_list_rand = greedy(order)
        
        if len(S_rand) > len(best_S):
            best_S = S_rand
            best_S_list = S_list_rand
            # If we found a significantly better set, we might want to polish it immediately?
            # Or just continue.
        
        restart_count += 1
        
        # If we have a very good set (e.g., > 100), we might reduce random restarts
        # and spend time on local search?
        # For now, just keep trying.
        
    # 4. Local Search / Polish
    # Try to add any element from [1, n] not in best_S to best_S.
    # This is a simple hill-climbing step.
    if time.time() < deadline - 0.5:
        current_S = set(best_S)
        current_S_list = sorted(current_S)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try to add each missing element
            for x in range(1, n + 1):
                if x in current_S:
                    continue
                if is_valid_addition(x, current_S_list):
                    current_S.add(x)
                    # Update sorted list
                    lo, hi = 0, len(current_S_list)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if current_S_list[mid] < x:
                            lo = mid + 1
                        else:
                            hi = mid
                    current_S_list.insert(lo, x)
                    improved = True
                    # Reset loop to try again, as adding one might allow another?
                    # Actually, if adding x was valid, it doesn't make other additions invalid
                    # unless the new x creates a conflict with a previously rejected element.
                    # So we should restart the scan or just continue.
                    # Continuing is fine, but we might miss elements that become valid
                    # only after adding x? No, adding elements only adds constraints.
                    # If y was invalid before, it remains invalid.
                    # So we can just break and exit, or continue to check others.
                    # Wait, if y was invalid because of a, and we add x, y is still invalid.
                    # So we don't need to re-check rejected elements.
                    # We just need to check the remaining candidates.
                    pass
        
        if len(current_S) > len(best_S):
            best_S = current_S
            best_S_list = current_S_list

    if not best_S:
        return [256]
    
    return list(best_S)

# Quick sanity check / self-test if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    sorted_S = sorted(S)
    for i, a in enumerate(sorted_S):
        for j, b in enumerate(sorted_S):
            for k, c in enumerate(sorted_S):
                if a < b <= c:
                    if (b + c) % a == 0:
                        valid = False
                        print(f"Invalid: {a}, {b}, {c}")
    print(f"Valid: {valid}")
