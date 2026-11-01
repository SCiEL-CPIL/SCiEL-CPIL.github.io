import itertools
import time
import random
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Optimized for q=3, n=7, radius=1.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (unlikely to be tested given the task contract)
        # Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        def get_ball(cw):
            mask = 0
            idx = word_to_idx[tuple(cw)]
            mask |= (1 << idx)
            for i in range(n):
                for val in range(q):
                    if val != cw[i]:
                        new_word = list(cw)
                        new_word[i] = val
                        new_idx = word_to_idx[tuple(new_word)]
                        mask |= (1 << new_idx)
            return mask

        N = len(all_words)
        full_mask = (1 << N) - 1
        balls = [get_ball(w) for w in all_words]
        
        covered_mask = 0
        codewords = []
        while covered_mask != full_mask:
            best_cw = -1
            best_gain = -1
            for i in range(N):
                gain = (balls[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cw = i
            if best_cw == -1:
                break
            codewords.append(best_cw)
            covered_mask |= balls[best_cw]
            
        return [list(all_words[cw]) for cw in codewords]

    # Specific optimization for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)  # 2187
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls as bitsets
    balls = [0] * N
    for w in all_words:
        idx = word_to_idx[w]
        mask = 1 << idx
        # Radius 1 ball in ternary length 7: 1 + 7*2 = 15 words
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    new_idx = word_to_idx[tuple(new_word)]
                    mask |= (1 << new_idx)
        balls[idx] = mask
        
    full_mask = (1 << N) - 1

    # 1. Initial Greedy Construction
    def greedy_cover():
        covered_mask = 0
        codewords = []
        # To speed up, we can maintain a list of candidates and their gains,
        # but recalculating gain is O(N) per pick. 2187 * 200 picks is ~400k ops, which is fine.
        while covered_mask != full_mask:
            best_cw = -1
            best_gain = -1
            # Optimization: Randomize tie-breaking or just pick first max
            for i in range(N):
                gain = (balls[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cw = i
            if best_cw == -1:
                break
            codewords.append(best_cw)
            covered_mask |= balls[best_cw]
        return codewords

    best_code = greedy_cover()
    best_size = len(best_code)
    
    # Local Search Helpers
    def is_valid(code_list):
        union_mask = 0
        for cw in code_list:
            union_mask |= balls[cw]
        return union_mask == full_mask

    def remove_redundant(code_list):
        """
        Removes redundant codewords. Returns new list and count of removals.
        A codeword is redundant if its ball is covered by the union of other balls.
        """
        removed_count = 0
        changed = True
        # We need to be careful with indices if we remove. 
        # Easier: create a new list by checking each one.
        # But checking each one against the rest is O(K^2). K~180, so 32k ops.
        # Doing this multiple times is fine.
        
        while changed:
            changed = False
            new_code = []
            # To make removal order independent, we can just check against the current list
            # and keep it if not redundant.
            # However, removing one might make another redundant.
            # So we iterate and check.
            
            # Optimization: Pre-compute union of all balls?
            # If we remove cw[i], we need to check if ball[i] is subset of union(others).
            
            # Let's just do a pass.
            for i in range(len(code_list)):
                cw = code_list[i]
                b = balls[cw]
                
                # Check if b is covered by others
                # Union of others = Union(all) ^ b? No, if b is subset of others, Union(all) == Union(others).
                # So: is (Union(all) & ~b) == (Union(others))?
                # Actually, simpler:
                # others_mask = 0
                # for j, c in enumerate(code_list):
                #   if j != i: others_mask |= balls[c]
                # if (b & ~others_mask) == 0: redundant
                
                # To avoid O(K^2) inside the while loop, we can use a smarter check.
                # But K is small. Let's stick to simple but correct.
                
                others_mask = 0
                for j in range(len(code_list)):
                    if j != i:
                        others_mask |= balls[code_list[j]]
                        
                if (b & ~others_mask) == 0:
                    # Redundant, do not add to new_code
                    removed_count += 1
                    changed = True
                else:
                    new_code.append(cw)
                    
            code_list = new_code
            
        return code_list, removed_count

    def kick_and_repair(code_list, kick_size):
        """
        Randomly remove 'kick_size' codewords, then use greedy to add back minimum needed.
        Returns new code list.
        """
        # Copy
        current = list(code_list)
        
        # Randomly remove
        k = min(kick_size, len(current))
        if k == 0:
            return current
        
        # Shuffle and remove last k
        random.shuffle(current)
        removed = current[-k:]
        current = current[:-k]
        
        # Calculate current coverage
        covered_mask = 0
        for cw in current:
            covered_mask |= balls[cw]
            
        # Greedily add back until full coverage
        # We want to add words that cover the most uncovered words.
        # To speed up, we can prioritize words that were in the original set?
        # No, just global greedy on uncovered.
        
        while covered_mask != full_mask:
            best_cw = -1
            best_gain = -1
            # Optimization: We can restrict search to words that cover uncovered areas.
            # But checking all N is fast enough.
            for i in range(N):
                gain = (balls[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_cw = i
            
            if best_cw == -1:
                # Should not happen if we start from a valid cover and just removed some
                break
                
            current.append(best_cw)
            covered_mask |= balls[best_cw]
            
        return current

    # Main Search Loop
    # Strategy: 
    # 1. Start with greedy solution.
    # 2. Loop:
    #    a. Try to remove redundant.
    #    b. If size improves, update best.
    #    c. If no removal, try a kick (remove random subset, repair).
    #    d. If kick improves size, update best and continue local search on it.
    #    e. Use simulated annealing acceptance for kicks that don't immediately improve size 
    #       but might lead to better states? 
    #       Actually, for covering codes, "kick and repair" often lands in a different local minimum.
    #       We can accept a kick if it results in a size <= current_size + tolerance, 
    #       then try to remove redundants from that new state.
    
    current_code = list(best_code)
    current_size = len(current_code)
    
    # Precompute a list of all words for random selection in kicks if needed, 
    # but we are removing from current_code, so no need for random additions unless repair fails? 
    # Repair uses global greedy, so it's fine.
    
    # Time management
    # We will alternate between "polish" (removals) and "kick" (perturbation).
    
    kick_size = 5
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        # 1. Polish: Remove redundants
        new_code, removed = remove_redundant(current_code)
        if removed > 0:
            current_code = new_code
            current_size = len(current_code)
            if current_size < best_size:
                best_size = current_size
                best_code = list(current_code)
                last_improvement_time = time.time()
                # If we improved, try to remove more immediately
                continue
        
        # 2. If stuck, try a kick
        # Adaptive kick size? Start small, increase if no progress?
        # Or just fixed small kicks.
        
        # We can also try "swap" moves: replace one word with another.
        # But kick+repair is more powerful.
        
        # Let's try a kick
        # To make it more effective, we can bias the removal towards "less useful" codewords?
        # Hard to define. Random is okay.
        
        kicked_code = kick_and_repair(current_code, kick_size)
        
        # Check if the kicked code is better
        # First, remove redundants from the kicked code
        polished_kick, _ = remove_redundant(kicked_code)
        kick_size_final = len(polished_kick)
        
        if kick_size_final < current_size:
            # Improvement!
            current_code = polished_kick
            current_size = kick_size_final
            if current_size < best_size:
                best_size = current_size
                best_code = list(current_code)
                last_improvement_time = time.time()
            # Keep searching from this new better state
            continue
            
        elif kick_size_final == current_size:
            # Same size, but different structure. 
            # It might allow further removals. 
            # Accept it with some probability or always?
            # If we always accept, we might wander. 
            # Let's accept it if it's different and we are stuck for a while.
            if time.time() - last_improvement_time > 5.0:
                 # Stuck for 5 seconds, accept neutral move to explore
                 current_code = polished_kick
                 # Don't update best_size, but change state
                 continue
                 
        else:
            # Worse. 
            # Simulated Annealing?
            # If we are stuck, we might accept a worse state to escape a local minimum.
            # But "worse" in covering code means larger size. 
            # If we accept a larger size, we then have to shrink it. 
            # This is risky if time is short.
            # However, kick+repair usually produces a valid cover of size close to original.
            # If it's significantly worse, skip.
            pass

        # If we are running out of time, do a final polish
        if time.time() > deadline - 10:
             break

    # Final Polish on the best found code
    final_code, _ = remove_redundant(best_code)
    
    return [list(all_words[cw]) for cw in final_code]

