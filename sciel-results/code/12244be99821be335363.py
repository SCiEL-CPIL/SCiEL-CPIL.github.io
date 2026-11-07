import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_valid(set_elements):
        n = len(set_elements)
        if n == 0:
            return True
        
        # Generate all non-empty subset sums
        sums = set()
        # Bitmask iteration
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += set_elements[i]
            if s in sums:
                return False
            sums.add(s)
        
        sums_list = list(sums)
        # Check divisibility
        # Optimization: sort sums. If a | b and a != b, then b >= 2a.
        # So for each a, we only need to check b where b >= 2a.
        sums_list.sort()
        for i in range(len(sums_list)):
            a = sums_list[i]
            # b must be a multiple of a, b != a
            # Since sums are distinct, b > a.
            # Smallest multiple is 2a.
            for j in range(i + 1, len(sums_list)):
                b = sums_list[j]
                if b >= 2 * a:
                    if b % a == 0:
                        return False
                else:
                    # b < 2a, so b cannot be a multiple of a (since b > a)
                    # But wait, if b < 2a, b % a != 0 is guaranteed if b != a?
                    # Yes, because the next multiple is 2a.
                    pass
        return True

    def get_sums(set_elements):
        n = len(set_elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += set_elements[i]
            sums.add(s)
        return sums

    best_size = 0
    best_set = []
    
    # Candidate pool: focus on larger numbers to reduce divisibility risk
    # But we need distinct sums, so they shouldn't be too correlated.
    # Primes are good.
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    primes = [n for n in range(100, 257) if is_prime(n)]
    others = [n for n in range(100, 257) if not is_prime(n)]
    candidates = primes + others
    random.shuffle(candidates)
    
    # Try to build a set of size 8
    target_size = 8
    
    # Phase 1: Random Greedy Construction
    for attempt in range(2000):
        if time.time() > deadline - 10:
            break
            
        current_set = []
        random.shuffle(candidates)
        
        for c in candidates:
            if len(current_set) >= target_size:
                break
            test_set = current_set + [c]
            if is_valid(test_set):
                current_set = test_set
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            if best_size >= 10: # Unlikely, but good if it happens
                break
                
    # Phase 2: Local Search / Swaps to improve size or find larger sets
    # We can try to increase size from best_size
    if best_size > 0:
        current_set = best_set[:]
        current_size = best_size
        
        # Try to add elements
        while time.time() < deadline - 5:
            improved = False
            # Shuffle candidates to vary search
            random.shuffle(candidates)
            for c in candidates:
                if c in current_set:
                    continue
                test_set = current_set + [c]
                if is_valid(test_set):
                    current_set = test_set
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                    break # Restart adding loop with new set
            
            if not improved:
                # Try swapping to unlock additions?
                # Simple swap: remove one, try to add two?
                # Or just remove one and see if we can add a different one that allows growth?
                # Let's just do random swaps to escape local minima
                if len(current_set) >= 2:
                    idx1 = random.randint(0, len(current_set) - 1)
                    # Find a candidate to swap in
                    swapped = False
                    for c in candidates:
                        if c in current_set:
                            continue
                        test_set = current_set[:]
                        test_set[idx1] = c
                        if is_valid(test_set):
                            # Check if this swap allows adding more later?
                            # Just accept if valid, hoping it helps
                            current_set = test_set
                            swapped = True
                            break
                    if not swapped:
                        break # Stuck
                    
    return [int(x) for x in best_set]
