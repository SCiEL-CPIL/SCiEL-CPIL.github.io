import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    def is_valid_set(S):
        """Check if S (as a set) satisfies the property: for all a<b<=c in S, a ∤ (b+c)."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            # For all j > i, k >= j: check (S_list[j] + S_list[k]) % a != 0
            for j in range(i + 1, m):
                b = S_list[j]
                # Check b+b (j=k=j)
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to set S without violating the property."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                pass
            else:
                # a < x
                # Case 1: x is 'c' (largest). Need a < b <= x for b in S.
                for j in range(m):
                    b = S_list[j]
                    if a < b <= x:
                        if (b + x) % a == 0:
                            return False
                # Case 2: x is 'b' (middle). Need a < x <= c for c in S.
                for j in range(m):
                    c = S_list[j]
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        # Case 3: x is 'a' (smallest). Need x < b <= c for b, c in S.
        for i in range(m):
            b = S_list[i]
            if b > x:
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True
    
    def can_remove(S, x):
        """Check if removing x from S still leaves a valid set."""
        S_without = S - {x}
        return is_valid_set(S_without)
    
    def try_swap(S, remove_x, add_y):
        """Try to swap remove_x out of S and add_y in. Returns True if successful."""
        if remove_x not in S or add_y in S:
            return False
        S_new = (S - {remove_x}) | {add_y}
        if is_valid_set(S_new):
            S.clear()
            S.update(S_new)
            return True
        return False
    
    def local_search(S, max_iterations=5000):
        """Perform local search by trying to add elements and swap elements."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 3.0:
            improved = False
            iterations += 1
            S_list = sorted(S)
            
            # Try adding elements
            for x in range(1, n + 1):
                if time.time() > deadline - 3.0:
                    break
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
                    break
            
            # Try swapping: remove one element, add another
            if not improved:
                for remove_x in S_list:
                    if time.time() > deadline - 3.0:
                        break
                    for add_y in range(1, n + 1):
                        if time.time() > deadline - 3.0:
                            break
                        if add_y not in S and add_y != remove_x:
                            S_temp = (S - {remove_x}) | {add_y}
                            if is_valid_set(S_temp):
                                S.clear()
                                S.update(S_temp)
                                improved = True
                                break
                    if improved:
                        break
        return S
    
    best_size = 0
    best_S = set()
    
    # Strategy 1: Start with upper half, greedily add from below
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 1000 + 42)
        
        # Start with a random subset of the upper portion
        base_start = n - rng.randint(40, 80)
        S = set(range(base_start, n + 1))
        
        # Verify and fix if needed
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        # Greedily add remaining elements in random order
        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Try to add more in another pass
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 2: Start with a random subset, greedily add
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 2000 + 99)
        
        # Start with a random subset of size ~30
        S = set(rng.sample(range(1, n + 1), rng.randint(20, 40)))
        
        # Verify and fix if needed
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        # Greedily add remaining elements in random order
        candidates = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Try to add more in another pass
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 3: Start with all odd numbers (they have good properties), then add evens
    if time.time() < deadline - 5.0:
        S = set(range(1, n + 1, 2))  # All odd numbers
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(2, n + 1, 2) if x not in S]
        rng = random.Random(12345)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 4: Start with numbers > n/3, then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(86, n + 1))  # Numbers > 256/3 ≈ 85.33
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, 86) if x not in S]
        rng = random.Random(54321)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 5: Start with numbers in (n/4, n/2], then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(65, 129))  # Numbers in (64, 128]
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, 65) if x not in S]
        rng = random.Random(77777)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 6: Start with numbers in (n/5, n/3], then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(52, 86))  # Numbers in (51, 85]
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, 52) if x not in S]
        rng = random.Random(88888)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Apply local search
        S = local_search(S, max_iterations=2000)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    if best_size == 0:
        # Fallback: return a valid set (upper half)
        return list(range(129, 257))
    
    # Final verification
    if not is_valid_set(best_S):
        # Should not happen, but fallback
        S = set(range(129, 257))
        return sorted(S)
    
    # Final local search on the best set found
    if time.time() < deadline - 2.0:
        best_S_copy = set(best_S)
        best_S_copy = local_search(best_S_copy, max_iterations=3000)
        if len(best_S_copy) > best_size:
            best_S = best_S_copy
            best_size = len(best_S)
    
    return sorted(best_S)
