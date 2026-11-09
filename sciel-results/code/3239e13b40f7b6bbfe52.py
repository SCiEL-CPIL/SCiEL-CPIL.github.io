import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit
    
    # Fixed seed for reproducibility of the "best" search path
    # The evaluator might run this once.
    random.seed(42)

    def is_valid(seq):
        """
        Check if all products of non-empty consecutive blocks are distinct.
        seq is a sorted list of integers.
        """
        if not seq:
            return True
        n = len(seq)
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def get_sorted_addition(seq, x):
        """
        Returns the new sorted list if x is added to seq.
        Does not check validity.
        """
        # Since seq is sorted, we can insert efficiently
        new_seq = seq + [x]
        new_seq.sort()
        return new_seq

    def try_insertion(seq, x):
        """
        Try to insert x into seq. Return new seq if valid, else None.
        """
        if x in seq:
            return None
        new_seq = get_sorted_addition(seq, x)
        if is_valid(new_seq):
            return new_seq
        return None

    # 1. Generate Initial Candidates
    
    # Greedy Forward
    seq_fwd = []
    for x in range(2, 129):
        # Check if adding x to current seq_fwd is valid
        # Optimize: Only check new products? For now, full check is fine.
        candidate = get_sorted_addition(seq_fwd, x)
        if is_valid(candidate):
            seq_fwd = candidate
            
    best_seq = seq_fwd[:]
    best_len = len(seq_fwd)
    
    # Greedy Reverse (try adding larger numbers first)
    seq_rev = []
    for x in range(128, 1, -1):
        candidate = get_sorted_addition(seq_rev, x)
        if is_valid(candidate):
            seq_rev = candidate
            
    if len(seq_rev) > best_len:
        best_seq = seq_rev[:]
        best_len = len(seq_rev)
        
    # Random Greedy Restarts
    for _ in range(20):
        seq_rand = []
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        for x in candidates:
            candidate = get_sorted_addition(seq_rand, x)
            if is_valid(candidate):
                seq_rand = candidate
        if len(seq_rand) > best_len:
            best_seq = seq_rand[:]
            best_len = len(seq_rand)
            
    # 2. Iterated Local Search (ILS)
    current_seq = best_seq[:]
    current_len = best_len
    
    # Pre-compute the set of numbers not in current_seq for fast access?
    # We'll just generate them on the fly as it's small.
    
    while time.time() < deadline:
        # Phase 1: Local Optimum Search (Hill Climbing via Insertion)
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            missing = [x for x in range(2, 129) if x not in current_seq]
            random.shuffle(missing)
            
            for x in missing:
                new_seq = try_insertion(current_seq, x)
                if new_seq is not None:
                    current_seq = new_seq
                    current_len += 1
                    if current_len > best_len:
                        best_len = current_len
                        best_seq = current_seq[:]
                    added_any = True
                    break # Restart the loop as the set of missing changed
        
        if time.time() > deadline:
            break
            
        # Phase 2: Kick (Perturbation)
        # Remove 1 to 3 random elements to escape local optimum
        kick_size = random.randint(1, 3)
        if current_len > kick_size:
            to_remove = random.sample(current_seq, kick_size)
            # Create a new sequence without these elements
            # It's easier to filter
            remove_set = set(to_remove)
            current_seq = [x for x in current_seq if x not in remove_set]
            current_len = len(current_seq)
            
            # Repair: Try to add elements back
            # This is effectively re-running Phase 1 but we start from a smaller set
            # We use the same "add if valid" logic
            added_any = True
            while added_any and time.time() < deadline:
                added_any = False
                missing = [x for x in range(2, 129) if x not in current_seq]
                random.shuffle(missing)
                
                for x in missing:
                    new_seq = try_insertion(current_seq, x)
                    if new_seq is not None:
                        current_seq = new_seq
                        current_len += 1
                        if current_len > best_len:
                            best_len = current_len
                            best_seq = current_seq[:]
                        added_any = True
                        break
                        
            # If we dropped below best significantly, revert to best to avoid wasting time
            # Or keep going? ILS usually keeps the perturbed state if it's a new basin.
            # But if we can't even recover close to best, we are in a bad basin.
            # Let's revert if we are more than 2 away from best.
            if current_len < best_len - 2:
                current_seq = best_seq[:]
                current_len = best_len

    return best_seq
