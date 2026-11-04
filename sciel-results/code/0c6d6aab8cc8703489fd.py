import time
import random
import itertools
import math

def construct(q, n, radius):
    # Specific to q=3, n=6, radius=1
    # General fallback for other params if needed, but task is specific.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to simple greedy or tetracode
        # Tetracode sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    # Map word tuple to index 0..728
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls
    # ball[i] is a list of indices covered by word i
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.add(to_idx(w))
            w[pos] = orig
        balls.append(list(covered))
        
    # Convert balls to sets for faster intersection/difference if needed, 
    # but lists are small (13 elements). 
    # Let's keep as lists for iteration, but maybe sets for membership?
    # balls_set[i] = set(balls[i])
    balls_set = [set(b) for b in balls]

    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # counts[word_idx] = number of codewords in current set covering word_idx
    counts = [0] * total_words
    for c in initial_code:
        for w in balls[c]:
            counts[w] += 1
            
    # Calculate initial uncovered count
    uncovered_count = sum(1 for w in counts if w == 0)
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    current_code = list(initial_code)
    current_counts = list(counts)
    code_set = set(current_code)
    
    # Simulated Annealing Parameters
    T = 100.0 # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    penalty = 1000 # Large penalty for uncovered words
    
    # Helper to calculate delta_uncovered for a swap (out, in)
    def calc_delta_uncovered(c_out, c_in, counts_arr):
        # Words in B(c_out) \ B(c_in) with count 1 will become uncovered
        # Words in B(c_in) \ B(c_out) with count 0 will become covered
        # Intersection: count change 0, status unchanged regarding 0/1 transition?
        # If w in intersection:
        #   old count k. new count k.
        #   If k==1, it was covered, stays covered.
        #   If k==0, it was uncovered, stays uncovered.
        # So only non-intersecting parts matter.
        
        delta = 0
        # Iterate B(c_out)
        for w in balls[c_out]:
            if w in balls_set[c_in]:
                continue
            if counts_arr[w] == 1:
                delta += 1 # Becomes uncovered
        # Iterate B(c_in)
        for w in balls[c_in]:
            if w in balls_set[c_out]:
                continue
            if counts_arr[w] == 0:
                delta -= 1 # Becomes covered
        
        return delta

    def do_swap(c_out, c_in, counts_arr, code_set_arr, code_list):
        code_list.remove(c_out)
        code_list.append(c_in)
        code_set_arr.discard(c_out)
        code_set_arr.add(c_in)
        
        for w in balls[c_out]:
            counts_arr[w] -= 1
        for w in balls[c_in]:
            counts_arr[w] += 1

    def do_remove(c, counts_arr, code_set_arr, code_list):
        code_list.remove(c)
        code_set_arr.discard(c)
        for w in balls[c]:
            counts_arr[w] -= 1

    # Phase 1: Greedy Remove to get a good starting point
    while True:
        removed_any = False
        random.shuffle(current_code)
        for c in current_code[:]:
            # Check if c is redundant
            is_redundant = True
            for w in balls[c]:
                if current_counts[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                do_remove(c, current_counts, code_set, current_code)
                removed_any = True
                # Recalculate uncovered count? 
                # Since we only remove if redundant, uncovered count doesn't change (stays 0 or increases if we were invalid, but we start valid).
                # Actually, if we start valid, uncovered_count is 0.
                # Removing a redundant word keeps it valid, so uncovered_count remains 0.
        if not removed_any:
            break
            
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        
    # Phase 2: Simulated Annealing with Swaps
    # We maintain valid code if possible, but allow invalid states to escape minima.
    # Objective: Minimize |code| + penalty * uncovered_count
    # Since |code| is constant during swaps, we minimize penalty * uncovered_count.
    # We also try to remove words if possible.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # 1. Try to remove a redundant word
        # Randomly pick a word to check? Or iterate?
        # Iterating over all is O(N_code * Ball_Size). N_code ~ 80, Ball_Size ~ 13. ~1000 ops.
        # Let's try a few random candidates for removal to save time?
        # No, full scan is fast enough.
        removed_any = False
        # Shuffle to break symmetry
        random.shuffle(current_code)
        for c in current_code:
            is_redundant = True
            for w in balls[c]:
                if current_counts[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                do_remove(c, current_counts, code_set, current_code)
                removed_any = True
                break # Restart scan? Or continue? Continue is fine, but state changed.
                # Actually, if we remove one, the redundancy of others might change.
                # Breaking and restarting the outer loop is safer for greedy.
                # But we are in a while True loop inside SA? No, this is inside the main SA loop.
                # Let's just do one removal per SA step if found.
                break
        
        if removed_any:
            # Size decreased!
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            # Reset temperature? Or just continue?
            # Usually, finding a better solution is a good sign.
            # We can boost temperature slightly to explore more?
            # Or just keep cooling.
            pass
            
        else:
            # No removal possible. Try a swap.
            # Pick random c_out from current_code
            if not current_code:
                break
            c_out = random.choice(current_code)
            
            # Pick random c_in not in code_set
            # To avoid infinite loops picking same, we can try a few times
            c_in = None
            for _ in range(10):
                rand_idx = random.randint(0, total_words - 1)
                if rand_idx not in code_set:
                    c_in = rand_idx
                    break
            if c_in is None:
                # Code is full? Unlikely.
                continue
                
            delta_unc = calc_delta_uncovered(c_out, c_in, current_counts)
            delta_cost = penalty * delta_unc
            
            # Acceptance criterion
            if delta_cost < 0:
                # Improvement (or neutral if 0)
                do_swap(c_out, c_in, current_counts, code_set, current_code)
            else:
                # Deterioration
                # P = exp(-delta_cost / T)
                if T > 0:
                    prob = math.exp(-delta_cost / T)
                    if random.random() < prob:
                        do_swap(c_out, c_in, current_counts, code_set, current_code)
            
            # Update uncovered count if we accepted?
            # We can recalculate it periodically or track it.
            # For now, we don't explicitly track uncovered_count in the acceptance logic 
            # except via delta_cost. 
            # But we need to know if we are valid to prefer removals?
            # The removal check `counts[w] == 1` implicitly handles validity.
            # If the code is invalid (some counts[w] == 0), removal is never safe 
            # for words in the uncovered set? 
            # Actually, if counts[w] == 0, it's not covered by anyone. 
            # Removing a codeword won't make it worse (it's already 0). 
            # But we can't remove a codeword if it's the ONLY one covering some other word.
            # The check `counts[w] == 1` for all w in B(c) ensures we don't break coverage 
            # of words that were covered. 
            # If the code is invalid, there are words with count 0. 
            # The removal check still works: it prevents making covered words uncovered.
            # It doesn't prevent adding new uncovered words (which can't happen by removal).
            # So removal is always "safe" in the sense of not increasing uncovered count.
            
            # Cool down
            T *= cooling_rate
            if T < min_T:
                T = min_T

    # Final Polish: Greedy remove again
    while True:
        removed_any = False
        random.shuffle(current_code)
        for c in current_code[:]:
            is_redundant = True
            for w in balls[c]:
                if current_counts[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                do_remove(c, current_counts, code_set, current_code)
                removed_any = True
        if not removed_any:
            break

    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)

    # Convert best_code indices back to lists of ints
    return [to_word(idx) for idx in best_code]
