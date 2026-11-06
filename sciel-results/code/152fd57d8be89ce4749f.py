import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s margin for 60s limit

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

    def greedy_construct(exclude_set=None):
        """
        Constructs a sequence by greedily adding elements from 2 to 128.
        If exclude_set is provided, those elements are skipped.
        """
        if exclude_set is None:
            exclude_set = set()
        
        values = []
        for x in range(2, 129):
            if x in exclude_set:
                continue
            new_seq = try_insert(values, x)
            if new_seq is not None:
                values = new_seq
        return values

    # 1. Initial Greedy Construction
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # 2. Local Search with Kicks
    # We maintain the best sequence found so far.
    # We try to improve it.
    
    # Precompute the set of all possible values
    all_values = set(range(2, 129))
    
    # Helper to get excluded elements
    def get_excluded(seq):
        return all_values - set(seq)
    
    current_seq = best_seq
    current_excluded = get_excluded(current_seq)
    
    # Random seed for reproducibility in this run, but we can use system time if needed
    # random.seed(int(time.time()))
    
    max_kicks = 20000
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
            new_seq = try_insert(current_seq, e)
            if new_seq is not None:
                current_seq = new_seq
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
        
        # Choose k between 1 and 4
        k = random.randint(1, 4)
        if len(current_seq) < k:
            k = len(current_seq)
            
        if k > 0:
            # Select k random elements to remove
            to_remove = random.sample(current_seq, k)
            to_remove_set = set(to_remove)
            
            # Remove them
            kicked_seq = [x for x in current_seq if x not in to_remove_set]
            
            # Now, greedily try to insert elements from the union of 
            # previously excluded and the removed ones
            candidates = current_excluded | to_remove_set
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            new_seq = kicked_seq
            new_excluded = all_values - set(new_seq)
            
            for e in cand_list:
                if time.time() > deadline:
                    break
                if e in new_excluded:
                    ins_seq = try_insert(new_seq, e)
                    if ins_seq is not None:
                        new_seq = ins_seq
                        new_excluded.remove(e)
            
            # If the new sequence is valid (it should be by construction), 
            # we can accept it if it's better or equal length?
            # To escape local optima, we should accept moves that don't decrease length 
            # too much, or even decrease length if it leads to a better state later.
            # However, for simplicity and safety, let's only accept if length is >= current length - 1
            # or if it allows for future improvements.
            # A common heuristic: accept if len(new_seq) >= len(current_seq)
            # Or use simulated annealing style.
            # Let's try: accept if len(new_seq) >= len(current_seq)
            # If not, we might want to accept with some probability if it's close.
            # Given the small space, let's be strict: only accept if length is not reduced.
            # But this might trap us. 
            # Alternative: Accept if len(new_seq) == len(current_seq) (swap-like)
            # or if len(new_seq) > len(current_seq).
            # If len(new_seq) < len(current_seq), we are losing length.
            # Let's accept if len(new_seq) >= len(current_seq) - 1 to allow some exploration.
            
            if len(new_seq) >= len(current_seq) - 1:
                current_seq = new_seq
                current_excluded = new_excluded
                
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(best_seq)
                    # Reset kick counter? No, just keep going.
            else:
                # Revert to previous state? 
                # Since we modified current_seq, we need to restore it if we reject.
                # But we didn't modify current_seq in place, we created new_seq.
                # So current_seq is still the old one.
                pass

    return best_seq
