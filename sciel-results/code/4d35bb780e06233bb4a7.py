import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
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

    def try_insert(seq, val):
        # seq is sorted
        # find position
        pos = 0
        while pos < len(seq) and seq[pos] < val:
            pos += 1
        # check if val is already in seq
        if pos < len(seq) and seq[pos] == val:
            return None
        new_seq = seq[:pos] + [val] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None

    # 1. Greedy Construction
    values = []
    for x in range(2, 129):
        new_seq = try_insert(values, x)
        if new_seq is not None:
            values = new_seq
    
    best_seq = values
    best_len = len(values)
    
    # 2. Local Search
    # We will try to improve best_seq
    # State: best_seq
    # Moves: Insert excluded, Swap (remove 1, add 1)
    
    excluded = set(range(2, 129)) - set(best_seq)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to insert any excluded element
        # Shuffle excluded to vary order
        excl_list = list(excluded)
        random.shuffle(excl_list)
        
        for e in excl_list:
            if time.time() > deadline:
                break
            new_seq = try_insert(best_seq, e)
            if new_seq is not None:
                best_seq = new_seq
                excluded.remove(e)
                best_len += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try swaps (remove x, add e) to escape local optima
        # We want to find a move that might enable future inserts.
        # We don't gain length, but we change the configuration.
        # We accept a swap if it's valid.
        # To avoid cycles, we might track visited states or just rely on the fact that
        # we will try inserts again.
        
        seq_list = list(best_seq)
        random.shuffle(seq_list)
        
        swap_made = False
        for x in seq_list:
            if time.time() > deadline:
                break
            # Try to remove x
            temp_seq = [v for v in best_seq if v != x]
            
            # Try to add any e
            for e in excl_list:
                if time.time() > deadline:
                    break
                new_seq = try_insert(temp_seq, e)
                if new_seq is not None:
                    best_seq = new_seq
                    excluded.remove(e)
                    excluded.add(x)
                    # Note: length is same.
                    swap_made = True
                    break
            if swap_made:
                break
        
        if not swap_made:
            # Try a more aggressive perturbation: Remove 2, Add 1 or 2?
            # Or just restart greedy with a different randomization?
            # Since the space is small, if we can't insert or swap, we might be stuck.
            # Let's try removing 2 random elements and then doing a greedy fill on the gap?
            # Actually, just break and return best found.
            # Or try a random restart of the whole process?
            # Given the small search space, local search should suffice.
            
            # One more idea: Try to insert e, and if it fails, check if removing a specific conflicting element helps.
            # The try_insert already checks validity.
            
            # If no move works, we might be at a local maximum.
            # We can try to "kick" by removing a random element and then trying to insert anything.
            if seq_list:
                x = random.choice(seq_list)
                temp_seq = [v for v in best_seq if v != x]
                # Try to insert any excluded element into temp_seq
                for e in excl_list:
                    if time.time() > deadline:
                        break
                    new_seq = try_insert(temp_seq, e)
                    if new_seq is not None:
                        best_seq = new_seq
                        excluded.remove(e)
                        excluded.add(x)
                        # Length decreased by 1? No, removed 1, added 1.
                        # Wait, if we remove x and add e, length is same.
                        # If we remove x and add NOTHING, length decreases.
                        # We only add if valid.
                        # So this is a swap.
                        break

    return best_seq
