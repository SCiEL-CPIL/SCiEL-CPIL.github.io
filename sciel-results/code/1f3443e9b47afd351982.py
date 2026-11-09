import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    N = 128

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        """Check if three distinct elements form a bad triple."""
        return gcd_table[a][b] == gcd_table[b][c] == gcd_table[a][c]

    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        # Optimization: Only pairs (a, b) in selected_set where gcd(a,b) == gcd(x,a) can be problematic.
        # If gcd(a,b) != gcd(x,a), then it's impossible for gcd(x,b) to equal both.
        # Specifically, we need gcd(a,b) == gcd(x,a) == gcd(x,b).
        # So we only need to check pairs (a,b) where gcd(a,b) == gcd(x,a).
        
        # Precompute gcd with x for all in selected_set
        # But selected_set is a set, so we need to iterate.
        
        # Let's iterate pairs (a, b) in selected_set
        # To speed up, we can group elements by their gcd with x?
        # No, the condition involves gcd(a,b) as well.
        
        # Let's stick to the O(k^2) check but with early termination and maybe some pruning.
        # Pruning: If gcd(x, a) == 1, then for a bad triple, gcd(a,b) must be 1 and gcd(x,b) must be 1.
        # So if gcd(x,a)==1, we only need to check pairs (a,b) where gcd(a,b)==1.
        
        # A more general optimization:
        # For a fixed x, let g_a = gcd(x, a).
        # We need to find a, b in S such that gcd(a,b) == g_a and gcd(x,b) == g_a.
        
        # Let's just do the direct check but optimized with local variables.
        sel_list = list(selected_set)
        n_sel = len(sel_list)
        
        for i in range(n_sel):
            a = sel_list[i]
            ga_x = gcd_table[x][a]
            # If ga_x is 1, then we need b such that gcd(a,b)=1 and gcd(x,b)=1.
            # If ga_x > 1, we need b such that gcd(a,b)=ga_x and gcd(x,b)=ga_x.
            
            for j in range(i + 1, n_sel):
                b = sel_list[j]
                if gcd_table[a][b] == ga_x and gcd_table[x][b] == ga_x:
                    return False
        return True

    def is_valid(selected_set):
        """Check if the set has no bad triples."""
        lst = list(selected_set)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i + 1, n):
                b = lst[j]
                ga = gcd_table[a][b]
                for k in range(j + 1, n):
                    c = lst[k]
                    if ga == gcd_table[a][c] == gcd_table[b][c]:
                        return False
        return True

    # Phase 1: Greedy construction with multiple orderings
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Define specific orderings
    nums = list(range(1, N + 1))
    
    # 1. Ascending
    orderings = [nums[:], nums[::-1]]
    
    # 2. By number of divisors
    def div_count(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    orderings.append(sorted(nums, key=lambda x: -div_count(x)))
    orderings.append(sorted(nums, key=lambda x: div_count(x)))
    
    # 3. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n)]
    orderings.append(primes + composites)
    orderings.append(composites + primes)
    
    # 4. Random restarts
    for _ in range(50):
        o = nums[:]
        rng.shuffle(o)
        orderings.append(o)
        
    for order in orderings:
        if time.time() > deadline - 30:
            break
            
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
                
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    # Phase 2: Local search - try to grow by adding elements
    if time.time() < deadline - 10:
        current = set(best_set)
        
        # Try adding in various orders until no more can be added
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            remaining = [x for x in range(1, N + 1) if x not in current]
            rng.shuffle(remaining)
            for x in remaining:
                if can_add(x, current):
                    current.add(x)
                    improved = True
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Phase 3: Local search with swaps (remove 1, add 1 or 2)
    # This is the most expensive part. We try to replace an element to allow more additions.
    if time.time() < deadline - 5:
        current = set(best_set)
        
        max_swap_iters = 50
        for it in range(max_swap_iters):
            if time.time() > deadline - 5:
                break
            
            found_swap = False
            # Try removing each element and see if we can add more
            for remove in list(current):
                if time.time() > deadline - 5:
                    break
                    
                temp = set(current)
                temp.discard(remove)
                
                # Try to add elements to temp
                added_count = 0
                remaining = [x for x in range(1, N + 1) if x not in temp and x != remove]
                rng.shuffle(remaining)
                
                for x in remaining:
                    if can_add(x, temp):
                        temp.add(x)
                        added_count += 1
                        # We only need to add 1 to maintain size, 2 to increase size
                        # But let's see how many we can add
                        if added_count >= 2:
                            break
                            
                # If we added at least 1, the size is maintained or increased
                # If we added 1, size is same. If 2, size is +1.
                # We only accept if size increases OR if we are stuck and want to explore?
                # Since we already did exhaustive add in Phase 2, any swap that allows an add is good.
                # But a swap that maintains size might allow future adds.
                # Let's accept if len(temp) > len(current) (strict improvement)
                # Or if len(temp) == len(current) but we haven't tried this configuration?
                # To keep it simple and fast, only accept strict improvements in size.
                if len(temp) > len(current):
                    current = temp
                    found_swap = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    break # Restart the loop with the new current
            
            if not found_swap:
                break
    
    # Phase 4: Final polish
    if time.time() < deadline:
        current = set(best_set)
        remaining = [x for x in range(1, N + 1) if x not in current]
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(x, current):
                current.add(x)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    return sorted(best_set)
