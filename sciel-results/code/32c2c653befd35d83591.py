import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for the 60s limit
    
    best_len = 0
    best_seq = []
    
    def update_best(prefixes):
        nonlocal best_len, best_seq
        seq_len = len(prefixes) - 1
        if seq_len > best_len:
            best_len = seq_len
            # Reconstruct the sequence of integers
            # S_i = P_i - P_{i-1}
            best_seq = [prefixes[i] - prefixes[i-1] for i in range(1, len(prefixes))]

    def backtrack(prefixes, used_diffs):
        nonlocal deadline
        
        # Time guard
        if time.time() > deadline:
            return False
            
        # Update best solution
        update_best(prefixes)
        
        last_p = prefixes[-1]
        
        # Generate valid candidates for the next prefix sum
        # P_new must be in [last_p + 1, last_p + 256]
        # And P_new - P_i must not be in used_diffs for all i < len(prefixes)
        
        candidates = []
        for gap in range(1, 257):
            P_new = last_p + gap
            valid = True
            for i in range(len(prefixes)):
                d = P_new - prefixes[i]
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(P_new)
                
        if not candidates:
            return True # Dead end
            
        # Randomize the order of candidates to explore different parts of the tree
        random.shuffle(candidates)
        
        for P_new in candidates:
            if time.time() > deadline:
                return False
                
            # Calculate new differences
            new_diffs = [P_new - p for p in prefixes]
            
            # Commit
            prefixes.append(P_new)
            for d in new_diffs:
                used_diffs.add(d)
                
            # Recurse
            result = backtrack(prefixes, used_diffs)
            if result == False: # Timeout
                return False
                
            # Undo
            prefixes.pop()
            for d in new_diffs:
                used_diffs.discard(d)
                
        return True

    # Run multiple restarts with different random seeds
    # This allows the search to escape local optima and explore different branches
    restart_id = 0
    while time.time() < deadline:
        restart_id += 1
        random.seed(restart_id)
        
        # Initialize state
        prefixes = [0]
        used_diffs = set()
        
        # Run backtracking search
        backtrack(prefixes, used_diffs)
        
        # If the search finished without timeout, we can continue to the next restart
        # to potentially find a better solution with a different ordering.
        
    return best_seq
