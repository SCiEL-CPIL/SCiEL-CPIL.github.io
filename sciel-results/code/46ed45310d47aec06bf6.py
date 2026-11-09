import time
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety
    
    best_sequence = []
    best_len = 0
    
    # We will search for the longest sequence.
    # Since the range is [1, 256], the max length is 256.
    # We can use a global flag to stop if we find a very long sequence, 
    # but we don't know the max.
    
    # Optimization: 
    # The set of used sums grows quadratically.
    # Checking intersection is O(L) if we generate the new sums list.
    
    def dfs(current_seq, used_sums, last_val):
        nonlocal best_sequence, best_len
        
        if time.time() > deadline:
            return
            
        current_len = len(current_seq)
        if current_len > best_len:
            best_len = current_len
            best_sequence = current_seq[:]
            
        # If we have used all possible values, stop
        if last_val == 256:
            return
            
        # Heuristic: If the theoretical max extension is small, prune?
        # Max possible additional elements is 256 - last_val.
        # If current_len + (256 - last_val) <= best_len, we can't beat it.
        # This is a weak bound but helps at deep levels.
        if current_len + (256 - last_val) <= best_len:
            return

        # Generate new sums if we add x
        # New sums are: x, x + last_val, x + last_val + prev_val, ...
        
        # We try candidates from last_val + 1 to 256.
        # To find longer sequences, it's often better to try smaller candidates first?
        # Or larger? 
        # Smaller candidates keep the cumulative sums lower, potentially avoiding 
        # collisions with sums that would be formed by larger numbers later.
        # Let's try ascending order.
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Calculate new sums
            new_sums = []
            total = x
            # We need to check if any new sum is in used_sums
            # Also, new_sums must be distinct (they are, as argued)
            
            # Efficient check:
            # The new sums are x, x+last, x+last+prev, ...
            # We can compute them and check against used_sums.
            
            # Optimization: 
            # If x is in used_sums, skip.
            if x in used_sums:
                continue
                
            # Check x + last_val
            total += last_val
            if total in used_sums:
                continue
            # We need to check all suffix sums.
            # Let's build the list of new sums to update used_sums if valid.
            
            new_sums_list = [x]
            curr_sum = x
            # Iterate backwards through current_seq
            for val in reversed(current_seq):
                curr_sum += val
                if curr_sum in used_sums:
                    break
                new_sums_list.append(curr_sum)
            else:
                # No collision found
                # Check if new_sums_list has duplicates? 
                # Since current_seq is strictly increasing positive, 
                # the suffix sums x, x+last, x+last+prev... are strictly increasing. 
                # So no internal duplicates.
                
                # Update state
                new_used_sums = used_sums | set(new_sums_list)
                
                dfs(current_seq + [x], new_used_sums, x)
                
                # Backtrack: used_sums is immutable in this recursion style (passed by value/copy)
                # But copying sets is expensive.
                
    # Initial call
    # We can start with an empty sequence.
    # However, starting with 1 is always optimal? 
    # If we skip 1, we can never use it later. 
    # Does using 1 prevent a longer sequence? 
    # Generally, including the smallest available number is safe for maximizing length 
    # because it minimizes the sums, leaving more "room" for distinct sums.
    # Let's start with 1.
    
    # But what if starting with 1 leads to a dead end sooner than starting with 2?
    # Unlikely, but DFS will explore other paths if we allow it.
    # To be safe and fast, let's just do a full DFS from empty, 
    # but prune aggressively.
    
    # Actually, a greedy initialization is good.
    # Let's run the greedy first to get a good lower bound.
    greedy_seq, greedy_sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(greedy_seq):
            total += a
            tails.append(total)
        if not greedy_sums.intersection(tails):
            greedy_seq.append(x)
            greedy_sums.update(tails)
            
    best_sequence = greedy_seq
    best_len = len(greedy_seq)
    
    # Now use DFS to try to improve.
    # We can restart DFS from the greedy sequence? 
    # Or do a full DFS with the greedy bound?
    # A full DFS from scratch might be too slow if the branching factor is high.
    # But the constraint is very tight.
    
    # Let's try a DFS that starts from the greedy sequence and tries to "swap" or "extend"?
    # No, standard DFS from root with pruning is better.
    # We use the greedy length as the initial best_len.
    
    # To speed up DFS, we can order the candidates.
    # We'll use the standard DFS.
    
    # One optimization: 
    # Instead of copying the set, we can use a "mark" array or similar? 
    # But sets are fine for small L.
    
    # Let's refine the DFS to be faster.
    # We can precompute suffix sums of the current sequence?
    # No, the sequence changes.
    
    # Let's try a Beam Search approach as suggested in inspirations?
    # Beam search keeps top K candidates at each depth.
    # This is often faster and finds good solutions.
    
    # Let's implement a Beam Search.
    
    def beam_search(width=100):
        nonlocal best_sequence, best_len
        
        # Each state: (sequence, used_sums, last_val)
        # We sort states by potential: 
        # 1. Length (prefer longer)
        # 2. Last value (prefer smaller last value to allow more room? or larger to fill gaps?)
        #    Smaller last value means more numbers available > last_val.
        #    So prefer smaller last_val.
        
        beam = [([], set(), 0)]
        
        max_depth = 0
        while beam:
            new_beam = []
            for seq, sums, last in beam:
                # If time is up, break
                if time.time() > deadline:
                    return
                    
                # Try to extend
                for x in range(last + 1, 257):
                    # Check validity
                    # Generate new sums
                    new_sums_list = [x]
                    curr_sum = x
                    valid = True
                    if x in sums:
                        continue
                    for val in reversed(seq):
                        curr_sum += val
                        if curr_sum in sums:
                            valid = False
                            break
                        new_sums_list.append(curr_sum)
                        
                    if not valid:
                        continue
                        
                    new_sums = sums | set(new_sums_list)
                    new_seq = seq + [x]
                    new_last = x
                    
                    # Update best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_sequence = new_seq
                        
                    # Add to new_beam
                    new_beam.append((new_seq, new_sums, new_last))
                    
            if not new_beam:
                break
                
            # Prune new_beam to size 'width'
            # Sort by:
            # 1. Length descending
            # 2. Last value ascending (smaller last = more room)
            # 3. Maybe sum of last few elements?
            
            new_beam.sort(key=lambda state: (-len(state[0]), state[2]))
            
            # Keep top 'width'
            beam = new_beam[:width]
            
            # Optional: If all states in beam have same length and we can't extend further?
            # We can check if the max possible length is reached.
            
    # Run beam search
    beam_search(width=50)
    
    return best_sequence
