import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Greedy construction - try multiple strategies
    def greedy_construct():
        seq = []
        prods = set()
        for x in range(2, 129):
            # Check if x can be appended
            new_prods = []
            current_prod = x
            valid = True
            
            if x in prods:
                valid = False
            else:
                new_prods.append(x)
                # Multiply by elements of seq in reverse
                for a in reversed(seq):
                    current_prod *= a
                    new_prods.append(current_prod)
                    if current_prod in prods:
                        valid = False
                        break
            
            if valid:
                seq.append(x)
                prods.update(new_prods)
        return seq, prods

    # Beam search with better state management
    def beam_search(initial_seq, initial_prods, beam_width=300, max_time=50):
        # State: (seq_tuple, prods_set)
        beam = [(tuple(initial_seq), initial_prods)]
        
        best_seq = list(initial_seq)
        best_len = len(best_seq)
        best_prods = set(initial_prods)

        seen_states = set([tuple(initial_seq)])
        
        time_limit = start_time + max_time
        
        while time.time() < time_limit:
            next_beam = []
            seen_in_next = set()
            
            for seq_t, prods in beam:
                last = seq_t[-1] if seq_t else 1
                start_val = last + 1
                
                # Prune: if we can't beat best_len, skip
                if len(seq_t) + (128 - start_val + 1) <= best_len:
                    continue
                
                # Try appending each possible next element
                for x in range(start_val, 129):
                    # Check if x can be appended
                    p = x
                    valid = True
                    if x in prods:
                        valid = False
                    else:
                        for a in reversed(seq_t):
                            p *= a
                            if p in prods:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    new_seq_t = seq_t + (x,)
                    if new_seq_t in seen_in_next:
                        continue
                    seen_in_next.add(new_seq_t)
                    
                    # Compute new products
                    new_prods_list = [x]
                    p = x
                    for a in reversed(seq_t):
                        p *= a
                        new_prods_list.append(p)
                    new_prods = prods.union(new_prods_list)
                    
                    if len(new_seq_t) > best_len:
                        best_len = len(new_seq_t)
                        best_seq = list(new_seq_t)
                        best_prods = new_prods
                    
                    next_beam.append((new_seq_t, new_prods))
            
            if not next_beam:
                break
            
            # Sort by length desc, then by last element asc for diversity
            next_beam.sort(key=lambda s: (-len(s[0]), s[0][-1] if s[0] else 0))
            beam = next_beam[:beam_width]
            
            # If we've exhausted all possibilities or beam is empty, break
            if len(beam) < beam_width:
                can_continue = False
                for seq_t, prods in beam:
                    if seq_t and seq_t[-1] < 128:
                        can_continue = True
                        break
                if not can_continue:
                    break
        
        return best_seq, best_prods

    # Local polish: try inserting missing numbers
    def polish_sequence(seq, prods, time_limit=15):
        current_seq = list(seq)
        current_prods = set(prods)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to insert each number from 2 to 128 not in current_seq
            for x in range(2, 129):
                if x in current_seq:
                    continue
                
                # Try inserting x at every position
                for i in range(len(current_seq) + 1):
                    # Check if inserting x at i is valid
                    # New blocks involving x:
                    # 1. [x]
                    # 2. [x, next] ... [x, next, ..., end]
                    # 3. [prev, x] ... [start, ..., prev, x]
                    
                    new_prods_set = set()
                    
                    # 1. x
                    new_prods_set.add(x)
                    
                    # 2. Right side: x * suffixes of current_seq[i:]
                    p = x
                    for j in range(i, len(current_seq)):
                        new_prods_set.add(p)
                        p *= current_seq[j]
                        
                    # 3. Left side: prefixes of current_seq[:i] * x
                    p = x
                    for j in range(i-1, -1, -1):
                        new_prods_set.add(p)
                        p *= current_seq[j]
                    
                    # Check for collisions with existing prods
                    if not new_prods_set.isdisjoint(current_prods):
                        continue
                        
                    # Check for collisions within new_prods_set
                    # The number of new products should equal the number of new blocks
                    # which is: 1 (just x) + (len(current_seq) - i) (right) + i (left)
                    expected_new_count = 1 + (len(current_seq) - i) + i
                    if len(new_prods_set) != expected_new_count:
                        continue
                    
                    # Valid insertion!
                    new_seq = current_seq[:i] + [x] + current_seq[i:]
                    new_prods = current_prods.union(new_prods_set)
                    
                    current_seq = new_seq
                    current_prods = new_prods
                    improved = True
                    break  # Restart the loop for x, as seq changed
                
                # If we improved, break out of x loop and restart
                if improved:
                    break
        
        return current_seq, current_prods

    # Swap-based local search: try swapping adjacent elements
    def swap_local_search(seq, prods, time_limit=10):
        current_seq = list(seq)
        current_prods = set(prods)
        time_limit_abs = start_time + time_limit
        
        improved = True
        while improved and time.time() < time_limit_abs:
            improved = False
            for i in range(len(current_seq) - 1):
                # Try swapping current_seq[i] and current_seq[i+1]
                # But we need strictly increasing, so only swap if result is still increasing
                if current_seq[i] >= current_seq[i+1]:
                    continue
                
                # Check if swapping would maintain strictly increasing
                # This is tricky - we'd need to revalidate the whole sequence
                # For now, skip this as it's complex
                pass
        
        return current_seq, current_prods

    # Step 1: Greedy construction
    greedy_seq, greedy_prods = greedy_construct()
    best_seq, best_prods = greedy_seq, greedy_prods
    best_len = len(best_seq)

    # Step 2: Beam search
    if time.time() < deadline - 20:
        beam_seq, beam_prods = beam_search(greedy_seq, greedy_prods)
        if len(beam_seq) > best_len:
            best_seq, best_prods, best_len = beam_seq, beam_prods, len(beam_seq)

    # Step 3: Polish
    if time.time() < deadline - 5:
        polished_seq, polished_prods = polish_sequence(best_seq, best_prods)
        if len(polished_seq) > best_len:
            best_seq, best_prods, best_len = polished_seq, polished_prods, len(polished_seq)

    # Ensure strictly increasing and in [2, 128]
    result = sorted(set(best_seq))
    result = [x for x in result if 2 <= x <= 128]
    
    return result
