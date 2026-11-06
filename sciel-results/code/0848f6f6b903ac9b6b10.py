import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for 90s limit

    def get_products(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def try_insert_fast(seq, val, existing_prods):
        """
        Try to insert val into seq at its sorted position.
        existing_prods is the set of all products of consecutive blocks in seq.
        Returns (new_seq, new_prods) if valid, None otherwise.
        """
        n = len(seq)
        pos = 0
        while pos < n and seq[pos] < val:
            pos += 1
        if pos < n and seq[pos] == val:
            return None
        
        if val in existing_prods:
            return None
        
        left_prods = []
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            left_prods.append(p)
        
        right_prods = []
        p = 1
        for j in range(pos, n):
            right_prods.append(p)
            p *= seq[j]
        
        for lp in left_prods:
            prod = lp * val
            if prod in existing_prods:
                return None
        
        for rp in right_prods:
            prod = val * rp
            if prod in existing_prods:
                return None
        
        for lp in left_prods:
            for rp in right_prods:
                prod = lp * val * rp
                if prod in existing_prods:
                    return None
        
        new_prods_list = [val]
        for lp in left_prods:
            new_prods_list.append(lp * val)
        for rp in right_prods:
            new_prods_list.append(val * rp)
        for lp in left_prods:
            for rp in right_prods:
                new_prods_list.append(lp * val * rp)
        
        new_prods_set = set(new_prods_list)
        if len(new_prods_set) != len(new_prods_list):
            return None
        
        new_seq = seq[:pos] + [val] + seq[pos:]
        new_prods = existing_prods | new_prods_set
        return (new_seq, new_prods)

    def greedy_construct(order=None):
        if order is None:
            order = list(range(2, 129))
        
        values = []
        prods = set()
        for x in order:
            result = try_insert_fast(values, x, prods)
            if result is not None:
                values, prods = result
        return values

    # 1. Initial Greedy Construction with natural order
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    all_values = set(range(2, 129))
    
    def get_excluded(seq):
        return all_values - set(seq)
    
    current_seq = best_seq
    current_excluded = get_excluded(current_seq)
    current_prods = get_products(current_seq)
    
    max_kicks = 50000
    kick_count = 0
    
    while time.time() < deadline and kick_count < max_kicks:
        kick_count += 1
        
        # Phase 1: Try to insert any excluded element
        improved = False
        excl_list = list(current_excluded)
        random.shuffle(excl_list)
        
        for e in excl_list:
            if time.time() > deadline:
                break
            result = try_insert_fast(current_seq, e, current_prods)
            if result is not None:
                current_seq, current_prods = result
                current_excluded.remove(e)
                improved = True
                break
        
        if improved:
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(best_seq)
            continue
            
        # Phase 2: If no direct insertion, try a "kick"
        k = random.randint(1, 6)
        if len(current_seq) < k:
            k = max(1, len(current_seq) // 2)
            
        if k > 0:
            to_remove = random.sample(current_seq, k)
            to_remove_set = set(to_remove)
            
            kicked_seq = [x for x in current_seq if x not in to_remove_set]
            kicked_prods = get_products(kicked_seq)
            
            candidates = current_excluded | to_remove_set
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            new_seq = kicked_seq
            new_prods = kicked_prods
            new_excluded = all_values - set(new_seq)
            
            for e in cand_list:
                if time.time() > deadline:
                    break
                if e in new_excluded:
                    result = try_insert_fast(new_seq, e, new_prods)
                    if result is not None:
                        new_seq, new_prods = result
                        new_excluded.remove(e)
            
            if len(new_seq) >= len(current_seq) - 1:
                current_seq = new_seq
                current_excluded = new_excluded
                current_prods = new_prods
                
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(best_seq)

    # Final phase: Try multiple fresh restarts with different orderings if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        restart_count = 0
        while time.time() < deadline - 1 and restart_count < 20:
            restart_count += 1
            vals = list(range(2, 129))
            random.shuffle(vals)
            values = []
            prods = set()
            for x in vals:
                result = try_insert_fast(values, x, prods)
                if result is not None:
                    values, prods = result
            if len(values) > best_len:
                best_seq = values
                best_len = len(best_seq)

    # Polish pass: try single-element swaps systematically
    if time.time() < deadline - 2:
        seq = list(best_seq)
        prods = get_products(seq)
        excluded = all_values - set(seq)
        
        # Try removing each element and inserting each excluded element
        for i in range(len(seq)):
            if time.time() > deadline - 1:
                break
            removed = seq[i]
            # Remove element i
            temp_seq = seq[:i] + seq[i+1:]
            temp_prods = get_products(temp_seq)
            
            # Try inserting each excluded element
            for e in sorted(excluded):
                if time.time() > deadline - 1:
                    break
                if e == removed:
                    continue
                result = try_insert_fast(temp_seq, e, temp_prods)
                if result is not None:
                    new_seq, new_prods = result
                    if len(new_seq) > best_len:
                        best_seq = new_seq
                        best_len = len(new_seq)
                        seq = list(new_seq)
                        prods = new_prods
                        excluded = all_values - set(seq)
                        break

    return best_seq
