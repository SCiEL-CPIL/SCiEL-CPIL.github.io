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

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

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
        
        # Check if inserting val creates any duplicate products
        # New products are:
        # 1. val itself (singleton)
        # 2. Products ending at pos-1 and starting with val: seq[i..pos-1] * val for i in [0..pos-1]
        # 3. Products starting at pos and ending with val: val * seq[pos..j] for j in [pos..n-1]
        # 4. Products that include val and extend both sides: seq[i..pos-1] * val * seq[pos..j]
        
        # Check val itself
        if val in existing_prods:
            return None
        
        # Check products involving val
        # For each possible block that includes the new element at position pos
        # A block [i..j] in new_seq that includes pos corresponds to:
        #   - If i == pos and j == pos: just val
        #   - If i < pos and j == pos: seq[i..pos-1] * val
        #   - If i == pos and j > pos: val * seq[pos..j]
        #   - If i < pos and j > pos: seq[i..pos-1] * val * seq[pos..j]
        
        # Precompute prefix products from pos-1 going left
        # and suffix products from pos going right
        left_prods = []  # left_prods[k] = product of seq[pos-1-k .. pos-1] for k=0..pos-1
        p = 1
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            left_prods.append(p)
        
        right_prods = []  # right_prods[k] = product of seq[pos .. pos+k] for k=0..n-1-pos
        p = 1
        for j in range(pos, n):
            right_prods.append(p)
            p *= seq[j]
        
        # Check all new products
        # Singleton: val
        # Left extensions: left_prods[k] * val
        for lp in left_prods:
            prod = lp * val
            if prod in existing_prods:
                return None
        
        # Right extensions: val * right_prods[k]
        for rp in right_prods:
            prod = val * rp
            if prod in existing_prods:
                return None
        
        # Both sides: left_prods[i] * val * right_prods[j]
        for lp in left_prods:
            for rp in right_prods:
                prod = lp * val * rp
                if prod in existing_prods:
                    return None
        
        # Also need to check that the new products don't conflict with each other
        # But since all new products include val and old products don't, 
        # and we checked against existing_prods, we just need to check new vs new.
        # Actually, two different new blocks could have the same product.
        # Let's collect all new products and check for duplicates among them.
        
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
        
        # All checks passed, construct new sequence and product set
        new_seq = seq[:pos] + [val] + seq[pos:]
        new_prods = existing_prods | new_prods_set
        return (new_seq, new_prods)

    def greedy_construct(order=None):
        """
        Constructs a sequence by greedily adding elements from 2 to 128.
        If order is provided, use that order; otherwise use natural order.
        """
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
    
    # Precompute the set of all possible values
    all_values = set(range(2, 129))
    
    # Helper to get excluded elements
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
            # If we improved, update best if longer
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(best_seq)
            # Continue searching from this new state
            continue
            
        # Phase 2: If no direct insertion, try a "kick"
        # Remove k random elements and then greedily re-fill
        
        # Choose k between 1 and 6
        k = random.randint(1, 6)
        if len(current_seq) < k:
            k = max(1, len(current_seq) // 2)
            
        if k > 0:
            # Select k random elements to remove
            to_remove = random.sample(current_seq, k)
            to_remove_set = set(to_remove)
            
            # Remove them and rebuild product set
            kicked_seq = [x for x in current_seq if x not in to_remove_set]
            kicked_prods = get_products(kicked_seq)
            
            # Now, greedily try to insert elements from the union of 
            # previously excluded and the removed ones
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
            
            # Accept if length is not reduced by more than 1
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
            # Random permutation of values
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

    return best_seq
