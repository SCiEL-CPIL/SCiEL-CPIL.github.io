import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def is_valid(seq):
        if not seq:
            return True
        n = len(seq)
        prods = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in prods:
                    return False
                prods.add(prod)
        return True

    def greedy_seed():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = greedy_seed()
    best_len = len(best_seq)
    
    all_numbers = list(range(2, 129))
    
    current_seq = list(best_seq)
    current_set = set(current_seq)
    current_len = len(current_seq)
    unused_list = [x for x in all_numbers if x not in current_set]
    
    def try_add(x):
        nonlocal current_seq, current_len, current_set, unused_list
        if x in current_set:
            return False
        lo, hi = 0, len(current_seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if current_seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        pos = lo
        new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        if is_valid(new_seq):
            current_seq = new_seq
            current_set.add(x)
            current_len += 1
            unused_list.remove(x) # O(N)
            return True
        return False

    def try_remove(x):
        nonlocal current_seq, current_len, current_set, unused_list
        if x not in current_set:
            return False
        current_set.remove(x)
        current_seq.remove(x)
        current_len -= 1
        unused_list.append(x)
        return True

    stuck_counter = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        if unused_list:
            x = random.choice(unused_list)
            if try_add(x):
                if current_len > best_len:
                    best_len = current_len
                    best_seq = list(current_seq)
                stuck_counter = 0
                continue
        
        # Swap / Perturb
        if current_len > 0:
            x = random.choice(current_seq)
            try_remove(x)
            
            # Try to add a new one
            added = False
            if unused_list:
                # Try a few random candidates
                for _ in range(10):
                    if not unused_list: break
                    y = random.choice(unused_list)
                    if try_add(y):
                        added = True
                        if current_len > best_len:
                            best_len = current_len
                            best_seq = list(current_seq)
                        stuck_counter = 0
                        break
            
            if not added:
                # Restore x
                try_add(x)
                stuck_counter += 1
        else:
            stuck_counter += 1

        if stuck_counter > 500:
            current_seq = list(best_seq)
            current_set = set(current_seq)
            current_len = len(current_seq)
            unused_list = [x for x in all_numbers if x not in current_set]
            stuck_counter = 0

    return best_seq
