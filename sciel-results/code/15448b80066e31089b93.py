import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_valid_extension(seq, sums, x):
        """Check if x can be appended to seq without creating duplicate consecutive sums."""
        n = len(seq)
        total = x
        if total in sums:
            return None
        new_sums = [total]
        for i in range(n - 1, -1, -1):
            total += seq[i]
            if total in sums:
                return None
            new_sums.append(total)
        return new_sums
    
    def greedy_construct(strategy, seed_val=0):
        """Greedy construction with given strategy. Returns best sequence found."""
        if seed_val:
            random.seed(seed_val)
        
        seq = []
        sums = set()
        
        if strategy == 'asc':
            candidates = list(range(1, 257))
        elif strategy == 'desc':
            candidates = list(range(256, 0, -1))
        else:  # random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        for x in candidates:
            if seq and x <= seq[-1]:
                continue
            if not seq and x < 1:
                continue
            
            new_sums = is_valid_extension(seq, sums, x)
            if new_sums is not None:
                seq.append(x)
                sums.update(new_sums)
        
        return seq
    
    # Try multiple strategies
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # baseline length 9
    best_len = len(best_seq)
    
    # Strategy 1: Greedy ascending
    seq1 = greedy_construct('asc')
    if len(seq1) > best_len:
        best_seq = seq1
        best_len = len(seq1)
    
    # Strategy 2: Greedy descending (start with large numbers)
    seq2 = greedy_construct('desc')
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        seq_r = greedy_construct('random', seed)
        if len(seq_r) > best_len:
            best_seq = seq_r
            best_len = len(seq_r)
    
    # Strategy 4: Backtracking search from a good prefix
    # Use the best sequence found so far as a guide
    if time.time() < deadline - 10:
        def backtrack(seq, sums, last_val):
            nonlocal best_seq, best_len, deadline
            
            if time.time() > deadline:
                return
            
            if len(seq) > best_len:
                best_seq = seq[:]
                best_len = len(seq)
            
            # Pruning
            if len(seq) + (256 - last_val) <= best_len:
                return
            
            # Try adding next elements in increasing order
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                
                new_sums = is_valid_extension(seq, sums, x)
                if new_sums is not None:
                    seq.append(x)
                    sums.update(new_sums)
                    
                    backtrack(seq, sums, x)
                    
                    # Backtrack
                    for s in new_sums:
                        sums.discard(s)
                    seq.pop()
        
        # Try starting with different small prefixes
        for start_val in [1, 2, 3, 4, 5]:
            if time.time() > deadline:
                break
            backtrack([start_val], {start_val}, start_val)
    
    # Strategy 5: More randomized restarts if time permits
    remaining = deadline - time.time()
    if remaining > 5:
        for seed in range(100, 200):
            if time.time() > deadline:
                break
            seq_r = greedy_construct('random', seed)
            if len(seq_r) > best_len:
                best_seq = seq_r
                best_len = len(seq_r)
    
    # Strategy 6: Local search - try to insert elements into existing sequences
    if time.time() < deadline - 2:
        # Try to extend the best sequence found so far by inserting elements
        def try_insertions(seq, max_time):
            """Try to insert elements into the sequence to make it longer."""
            current_seq = seq[:]
            current_sums = set()
            
            # Compute all consecutive block sums
            n = len(current_seq)
            for i in range(n):
                total = 0
                for j in range(i, n):
                    total += current_seq[j]
                    current_sums.add(total)
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                # Try inserting each possible value at each position
                for val in range(1, 257):
                    if val in current_seq:
                        continue
                    for pos in range(len(current_seq) + 1):
                        # Check if inserting val at pos maintains strictly increasing
                        if pos > 0 and val <= current_seq[pos - 1]:
                            continue
                        if pos < len(current_seq) and val >= current_seq[pos]:
                            continue
                        
                        # Check if insertion creates duplicate sums
                        new_seq = current_seq[:pos] + [val] + current_seq[pos:]
                        n_new = len(new_seq)
                        
                        # Only need to check sums that include the new element
                        # Sums entirely before or after pos are unchanged
                        # Sums that cross pos need checking
                        conflict = False
                        
                        # Sums starting at or before pos and ending at or after pos
                        # These are: [i..j] where i <= pos <= j in new_seq
                        # In terms of old seq: sums that span the insertion point
                        # Plus the element itself, and sums with elements to the left/right
                        
                        # Check the new element alone
                        if val in current_sums:
                            conflict = True
                            break
                        
                        # Check sums ending at the new element (starting from left)
                        total = val
                        for i in range(pos - 1, -1, -1):
                            total += current_seq[i]
                            if total in current_sums:
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        # Check sums starting at the new element (going right)
                        total = val
                        for j in range(pos, len(current_seq)):
                            total += current_seq[j]
                            if total in current_sums:
                                conflict = True
                                break
                        if conflict:
                            continue
                        
                        # Check sums spanning the new element (left part + val + right part)
                        for i in range(pos - 1, -1, -1):
                            left_sum = current_seq[i]
                            total_left = left_sum + val
                            if total_left in current_sums:
                                conflict = True
                                break
                            # Extend right
                            total_span = total_left
                            for j in range(pos, len(current_seq)):
                                total_span += current_seq[j]
                                if total_span in current_sums:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            continue
                        
                        # Insertion is valid - do it
                        current_seq = new_seq
                        # Recompute sums
                        current_sums = set()
                        n_new = len(current_seq)
                        for i in range(n_new):
                            total = 0
                            for j in range(i, n_new):
                                total += current_seq[j]
                                current_sums.add(total)
                        improved = True
                        break
                    if improved:
                        break
            
            return current_seq
        
        # Try to improve the best sequence
        improved_seq = try_insertions(best_seq, deadline - 2)
        if len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = best_len
    
    return best_seq
