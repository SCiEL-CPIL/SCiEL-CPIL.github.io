import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_len = 0
    best_set = set()
    best_products = set()
    
    def check_validity_from_scratch(numbers):
        if not numbers:
            return True, set()
        nums = sorted(numbers)
        n = len(nums)
        products = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= nums[j]
                if p in products:
                    return False, set()
                products.add(p)
        return True, products

    def get_valid_additions(current_set, existing_products):
        valid_adds = []
        current_list = sorted(current_set)
        n = len(current_list)
        
        for x in range(2, 129):
            if x in current_set:
                continue
            
            k = 0
            while k < n and current_list[k] < x:
                k += 1
            
            new_products = set()
            valid = True
            
            for i in range(k + 1):
                p_left = 1
                for m in range(i, k):
                    p_left *= current_list[m]
                
                p_mid = p_left * x
                if p_mid in existing_products or p_mid in new_products:
                    valid = False
                    break
                new_products.add(p_mid)
                
                p_right = p_mid
                for m in range(k, n):
                    p_right *= current_list[m]
                    if p_right in existing_products or p_right in new_products:
                        valid = False
                        break
                    new_products.add(p_right)
                
                if not valid:
                    break
            
            if valid:
                valid_adds.append((x, new_products))
        
        return valid_adds

    # Initial Greedy
    s = set()
    prods = set()
    for x in range(2, 129):
        if x in s:
            continue
        # Check validity
        # We can use a simplified check for greedy init
        # Since we are adding in increasing order, k is always n.
        # So we only check blocks ending at n.
        # Blocks are [i..n] for i in 0..n.
        
        # Actually, let's just use the full check for simplicity in init
        # It's fast enough.
        new_s = s | {x}
        ok, new_prods = check_validity_from_scratch(new_s)
        if ok:
            s = new_s
            prods = new_prods
            
    best_set = s
    best_len = len(s)
    best_products = prods

    # Local Search Loop
    while time.time() < deadline:
        current_set = set(best_set)
        current_products = set(best_products)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            valid_adds = get_valid_additions(current_set, current_products)
            
            if valid_adds:
                # Pick random
                x, new_prods = random.choice(valid_adds)
                current_set.add(x)
                current_products.update(new_prods)
                improved = True
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = set(current_set)
                    best_products = set(current_products)
            else:
                # Stuck. Try a random swap?
                # Or break and restart.
                break
        
        # Restart with perturbation
        # Skip a random element
        if time.time() >= deadline:
            break
            
        skip = random.randint(2, 128)
        s = set()
        prods = set()
        for x in range(2, 129):
            if x == skip or x in s:
                continue
            new_s = s | {x}
            ok, new_prods = check_validity_from_scratch(new_s)
            if ok:
                s = new_s
                prods = new_prods
                
        if len(s) > best_len:
            best_len = len(s)
            best_set = set(s)
            best_products = set(prods)
            
        # Local search on the new s
        current_set = set(s)
        current_products = set(prods)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            valid_adds = get_valid_additions(current_set, current_products)
            if valid_adds:
                x, new_prods = random.choice(valid_adds)
                current_set.add(x)
                current_products.update(new_prods)
                improved = True
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = set(current_set)
                    best_products = set(current_products)

    return sorted(best_set)
