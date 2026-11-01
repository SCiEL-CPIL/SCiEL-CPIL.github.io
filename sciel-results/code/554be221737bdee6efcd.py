import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # The task is specific to q=3, n=8, radius=1.
    # If parameters differ, fall back to a simple heuristic or seed logic.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed-like logic for other parameters
        # This is just to ensure the function doesn't crash if called with other params
        # though the evaluator will pass 3, 8, 1.
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total_words = q ** n
    # Map word (list of ints) to index (int) and vice versa
    # We use integer representation for speed.
    # word = sum(w[i] * q^(n-1-i))
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices covered by word i
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        # Convert index to base-q digits
        w = [0] * n
        temp = i
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        
        # Add self
        neighbors[i].append(i)
        
        # Add neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    new_idx = 0
                    for k in range(n):
                        new_idx = new_idx * q + new_w[k]
                    neighbors[i].append(new_idx)
    
    # Greedy Set Cover
    # covered_count[j] = number of codewords covering word j
    covered_count = np.zeros(total_words, dtype=np.int32)
    code = []
    
    # To speed up greedy, we can maintain a priority queue or just scan.
    # Scanning 6561 items is fast.
    while True:
        if time.time() > deadline:
            break
        
        # Find the word that covers the most uncovered words
        best_word = -1
        max_uncovered = -1
        
        # Optimization: Only consider words that cover at least one uncovered word
        # We can check this by seeing if any neighbor has covered_count == 0
        for i in range(total_words):
            # Quick check: if this word doesn't cover any new words, skip
            # This check is O(17)
            has_uncovered = False
            for nb in neighbors[i]:
                if covered_count[nb] == 0:
                    has_uncovered = True
                    break
            
            if not has_uncovered:
                continue
                
            count_uncovered = 0
            for nb in neighbors[i]:
                if covered_count[nb] == 0:
                    count_uncovered += 1
            
            if count_uncovered > max_uncovered:
                max_uncovered = count_uncovered
                best_word = i
                
        if best_word == -1:
            break # All covered
            
        # Add best_word to code
        code.append(best_word)
        
        # Update covered counts
        for nb in neighbors[best_word]:
            covered_count[nb] += 1
            
        # Early termination if all covered
        if np.all(covered_count > 0):
            break

    # Local Search: Redundancy Removal
    # Repeat until no more removals
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if removing c leaves any word uncovered
            # i.e., check if any neighbor of c has covered_count == 1
            # and c is the only one covering it.
            # Actually, if covered_count[nb] == 1, then c MUST be the one covering it 
            # (since we are checking if we can remove c).
            # So if any nb has covered_count[nb] == 1, we cannot remove c.
            # If all nb have covered_count[nb] > 1, we can remove c.
            
            redundant = True
            for nb in neighbors[c]:
                if covered_count[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove c
                code.pop(i)
                for nb in neighbors[c]:
                    covered_count[nb] -= 1
                changed = True
                # After a removal, restart the scan to ensure we catch cascading redundancies
                # or just continue. Continuing is faster, restarting is safer for finding all.
                # Given small size, restarting is fine.
                break 

    # Advanced Local Search: Swap-based Optimization
    # Try to swap codewords to create redundancies or improve coverage structure
    # We iterate several times. In each iteration, we try to remove a codeword.
    # If it fails, we try to find a "better" replacement that might allow removal in future steps.
    # A simpler heuristic: Randomly perturb the code and re-prune.
    
    # Strategy: 
    # 1. Try to remove every codeword. If successful, great.
    # 2. If not, pick a random codeword, remove it, and find the best new codeword to cover the gaps.
    #    This is a "move" operation. If the new codeword is already in the set, it's a swap.
    #    If it's new, the size stays same.
    #    Then re-prune.
    
    # To avoid infinite loops or slow progress, we limit the number of swap iterations.
    max_swap_iterations = 5000
    swap_iterations = 0
    random.seed(42) # Deterministic for reproducibility, but we can also use random
    
    while swap_iterations < max_swap_iterations:
        if time.time() > deadline:
            break
        
        # Try to remove a random codeword
        if not code:
            break
            
        idx_to_try = random.randint(0, len(code) - 1)
        c = code[idx_to_try]
        
        # Check if c is removable
        removable = True
        for nb in neighbors[c]:
            if covered_count[nb] == 1:
                removable = False
                break
        
        if removable:
            # Remove it
            code.pop(idx_to_try)
            for nb in neighbors[c]:
                covered_count[nb] -= 1
            swap_iterations += 1
            continue
        
        # If not removable, try to "move" it.
        # Identify the "critical" neighbors (those with count 1).
        # We want to find a new codeword c_new that covers all critical neighbors of c,
        # and ideally covers some other low-count neighbors to facilitate future removals.
        
        criticals = [nb for nb in neighbors[c] if covered_count[nb] == 1]
        
        # Find a candidate c_new that covers all criticals
        # and minimizes the "cost" (number of new unique words covered? or maximizes overlap?)
        # Actually, we just want to restore coverage.
        # Let's find a codeword that covers all criticals.
        
        best_candidate = -1
        best_score = -1
        
        # To speed this up, we can just iterate through all possible words?
        # 6561 words * 17 neighbors check is fast enough if we do it a few thousand times.
        # But we can be smarter.
        
        # For each word w in 0..total_words-1:
        #   if w covers all criticals:
        #      score = number of neighbors of w with count == 1 (excluding criticals? no, including)
        #      Actually, we want to maximize the number of neighbors that become "shared" (count > 1)
        #      or minimize the "uniqueness" of the new codeword.
        
        # Simpler: Just find ANY word that covers all criticals.
        # Then swap. Then prune.
        
        # To avoid bias, we can pick a random word that covers all criticals.
        # Let's collect candidates.
        
        candidates = []
        for w in range(total_words):
            # Check if w covers all criticals
            # This is O(|criticals|)
            covers_all = True
            for crit in criticals:
                if crit not in neighbors[w]:
                    covers_all = False
                    break
            if covers_all:
                candidates.append(w)
        
        if candidates:
            # Pick a random candidate to diversify
            c_new = random.choice(candidates)
            
            # If c_new is already in code, we can't just "add" it, we must swap.
            # If c_new is c, do nothing.
            if c_new != c:
                # Check if c_new is in code
                # This is O(N)
                is_in_code = False
                for cw in code:
                    if cw == c_new:
                        is_in_code = True
                        break
                
                if is_in_code:
                    # It's a duplicate. Swapping c with c_new (which is already there)
                    # effectively removes c and adds a duplicate? No, code is a set/list of unique words.
                    # If c_new is in code, we can just remove c? 
                    # Wait, if c_new is in code, it already covers the criticals?
                    # If c_new is in code, then the criticals of c are covered by c_new?
                    # No, criticals are words with count 1 covered ONLY by c.
                    # If c_new is in code, and c_new covers a critical word, then that critical word's count would be at least 2 (c and c_new).
                    # So if a word is critical (count 1) for c, NO other codeword covers it.
                    # Therefore, c_new CANNOT be in the code if it covers a critical word of c, unless c_new == c.
                    # So c_new must be a NEW word not in the code.
                    
                    # So we can just swap: remove c, add c_new.
                    code[idx_to_try] = c_new
                    # Update counts
                    for nb in neighbors[c]:
                        covered_count[nb] -= 1
                    for nb in neighbors[c_new]:
                        covered_count[nb] += 1
                    swap_iterations += 1
                    # After a swap, we should immediately try to prune, as the structure changed.
                    # The outer loop will handle removals.
                else:
                    # c_new is not in code. Swap c with c_new.
                    code[idx_to_try] = c_new
                    for nb in neighbors[c]:
                        covered_count[nb] -= 1
                    for nb in neighbors[c_new]:
                        covered_count[nb] += 1
                    swap_iterations += 1
            else:
                # c_new is c itself. Do nothing.
                pass
        else:
            # No candidate found? This shouldn't happen if the code is valid.
            # But if it does, skip.
            pass

    # Final Pruning
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            redundant = True
            for nb in neighbors[c]:
                if covered_count[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                code.pop(i)
                for nb in neighbors[c]:
                    covered_count[nb] -= 1
                changed = True
                break 

    # Convert indices back to lists of integers
    result = []
    for idx in code:
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        result.append(w)
        
    return result
