import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0
    
    def is_valid_addition(s_sorted, candidate):
        n = len(s_sorted)
        # Find insertion point
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if s_sorted[mid] < candidate:
                lo = mid + 1
            else:
                hi = mid
        pos = lo
        
        # Case 1: candidate is 'a'
        # b, c in s_sorted[pos:]
        # Note: s_sorted[pos:] contains elements >= candidate.
        # Since candidate is not in s_sorted, all elements in s_sorted[pos:] are > candidate.
        # We need b <= c.
        L = s_sorted[pos:]
        for i in range(len(L)):
            b = L[i]
            for j in range(i, len(L)):
                c = L[j]
                if (b + c) % candidate == 0:
                    return False
        
        # Case 2: candidate is 'b'
        # a in s_sorted[:pos], c in s_sorted[pos:] U {candidate}
        for i in range(pos):
            a = s_sorted[i]
            # c = candidate
            if (2 * candidate) % a == 0:
                return False
            # c in s_sorted[pos:]
            target = (-candidate) % a
            for k in range(pos, n):
                if s_sorted[k] % a == target:
                    return False
        
        # Case 3: candidate is 'c'
        # a < b in s_sorted[:pos]
        for i in range(pos):
            a = s_sorted[i]
            target = (-candidate) % a
            for j in range(i + 1, pos):
                if s_sorted[j] % a == target:
                    return False
        
        return True

    def insert_sorted(s_sorted, x):
        lo, hi = 0, len(s_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            if s_sorted[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        s_sorted.insert(lo, x)

    def greedy(order):
        s = []
        for x in order:
            if is_valid_addition(s, x):
                insert_sorted(s, x)
        return s

    # Initial Greedy Strategies
    best_s = []
    best_len = 0
    
    # Strategy 1: High to Low
    s1 = greedy(list(range(N, 0, -1)))
    if len(s1) > best_len:
        best_s = s1
        best_len = len(s1)
        
    # Strategy 2: Low to High
    s2 = greedy(list(range(1, N + 1)))
    if len(s2) > best_len:
        best_s = s2
        best_len = len(s2)
        
    # Strategy 3: Randomized
    for _ in range(10):
        if time.time() - start_time > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Bias towards larger numbers
        biased_order = sorted(order, key=lambda x: x * random.uniform(0.5, 1.5), reverse=True)
        s = greedy(biased_order)
        if len(s) > best_len:
            best_s = s
            best_len = len(s)

    current_s = best_s[:]
    current_set = set(current_s)
    
    no_improve = 0
    max_no_improve = 100
    
    while time.time() - start_time < deadline:
        if no_improve > max_no_improve:
            break
        
        improved = False
        
        # 1. Try to add any single element
        missing = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(missing)
        for x in missing:
            if is_valid_addition(current_s, x):
                insert_sorted(current_s, x)
                current_set.add(x)
                improved = True
                break
        
        if improved:
            no_improve = 0
            continue
        
        # 2. Try swap: remove 1, add 2
        if len(current_s) > 0:
            # Try a few random removals
            for _ in range(5):
                if time.time() - start_time > deadline:
                    break
                rem = random.choice(current_s)
                temp_s = [x for x in current_s if x != rem]
                temp_set = current_set - {rem}
                
                missing = [x for x in range(1, N + 1) if x not in temp_set]
                random.shuffle(missing)
                
                added = 0
                new_s = temp_s[:]
                new_set = temp_set.copy()
                
                for x in missing:
                    if added >= 2:
                        break
                    if is_valid_addition(new_s, x):
                        insert_sorted(new_s, x)
                        new_set.add(x)
                        added += 1
                
                if added >= 2:
                    current_s = new_s
                    current_set = new_set
                    improved = True
                    break
        
        if not improved:
            # 3. Try swap: remove 2, add 3
            if len(current_s) > 2:
                for _ in range(5):
                    if time.time() - start_time > deadline:
                        break
                    rem1, rem2 = random.sample(current_s, 2)
                    rem_set = {rem1, rem2}
                    temp_s = [x for x in current_s if x not in rem_set]
                    temp_set = current_set - rem_set
                    
                    missing = [x for x in range(1, N + 1) if x not in temp_set]
                    random.shuffle(missing)
                    
                    added = 0
                    new_s = temp_s[:]
                    new_set = temp_set.copy()
                    
                    for x in missing:
                        if added >= 3:
                            break
                        if is_valid_addition(new_s, x):
                            insert_sorted(new_s, x)
                            new_set.add(x)
                            added += 1
                    
                    if added >= 3:
                        current_s = new_s
                        current_set = new_set
                        improved = True
                        break
                        
        if not improved:
            # 4. Perturbation
            k = random.randint(3, 6)
            if len(current_s) > k:
                rem_vals = random.sample(current_s, k)
                rem_set = set(rem_vals)
                perturbed_s = [x for x in current_s if x not in rem_set]
                perturbed_set = current_set - rem_set
                
                missing = [x for x in range(1, N + 1) if x not in perturbed_set]
                random.shuffle(missing)
                
                for x in missing:
                    if time.time() - start_time > deadline:
                        break
                    if is_valid_addition(perturbed_s, x):
                        insert_sorted(perturbed_s, x)
                        perturbed_set.add(x)
                
                if len(perturbed_s) > len(current_s):
                    current_s = perturbed_s
                    current_set = perturbed_set
                    improved = True
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            
    return current_s
