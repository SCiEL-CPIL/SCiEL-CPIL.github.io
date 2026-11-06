import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    N = 256
    best_seq = []
    best_len = 0

    def check_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def improve(seq):
        """Local search to maximize length starting from a valid seq."""
        nonlocal best_seq, best_len
        current = seq[:]
        current_set = set(current)
        
        # If current is already better, update best
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current[:]

        while time.time() < deadline - 2:
            # 1. Try to add any element
            # Generate list of candidates not in current_set
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                break  # All numbers used
            
            # Shuffle to avoid bias towards smaller/larger numbers
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                # Check if adding x maintains validity
                # We can optimize by checking sums involving x
                # But O(k^2) is fast enough for k < 100
                new_seq = sorted(current + [x])
                if check_valid(new_seq):
                    current = new_seq
                    current_set.add(x)
                    if len(current) > best_len:
                        best_len = len(current)
                        best_seq = current[:]
                    added = True
                    break  # Restart search with new length to try more additions
            
            if added:
                continue  # Go back to top to try adding more
            
            # 2. If no addition possible, try swaps to escape local optimum
            # Try a limited number of random swaps
            swap_made = False
            # Number of swaps to try depends on time, but 100 is a good heuristic
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                if not current:
                    break
                
                # Pick random element to remove
                i_val = random.choice(current)
                # Pick random element to add (from candidates)
                # Note: candidates list might be stale if we modified current_set?
                # We didn't modify current_set in the failed addition loop.
                # But we should ensure j_val is not in current_set.
                # candidates is built from current_set at the start of the loop iteration.
                # Since we didn't change current_set, candidates is still valid.
                
                j_val = random.choice(candidates)
                
                # Perform swap
                new_set = current_set - {i_val} | {j_val}
                new_seq = sorted(new_set)
                
                if check_valid(new_seq):
                    current = new_seq
                    current_set = new_set
                    swap_made = True
                    break  # Restart outer loop to check for new additions
            
            if not swap_made:
                # Stuck in local optimum for this configuration
                # Break out to try a different initial sequence
                break

    # Phase 1: Greedy constructions with different orderings
    # 1. Ascending order
    seq = []
    for x in range(1, N + 1):
        new_seq = sorted(seq + [x])
        if check_valid(new_seq):
            seq = new_seq
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
    if best_seq:
        improve(best_seq)

    # 2. Descending order
    seq = []
    for x in range(N, 0, -1):
        new_seq = sorted(seq + [x])
        if check_valid(new_seq):
            seq = new_seq
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]
    if best_seq:
        improve(best_seq)

    # 3. Random orderings
    # Run several random greedy restarts
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        seq = []
        for x in order:
            new_seq = sorted(seq + [x])
            if check_valid(new_seq):
                seq = new_seq
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        if best_seq:
            improve(best_seq)

    return best_seq
