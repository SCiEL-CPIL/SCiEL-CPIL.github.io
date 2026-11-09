import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    # For this task, q=3, n=7, radius=1
    # We assume the evaluator passes these specific values or compatible ones.
    # The problem statement says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # But the specific problem is K_3(7,1).
    # We will implement the heuristic for the general case but tuned for q=3, n=7.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to a simple construction for other parameters if needed, 
        # though the task is specific.
        # For now, we focus on the specific case.
        pass

    total_words = q ** n
    # Map word (tuple) to index
    # We use base-q representation
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (radius 1) for each word
    # neighbors[i] is a list of indices covered by word i
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        covered = [i]
        for pos in range(n):
            original = w[pos]
            for val in range(q):
                if val == original:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = original
        neighbors.append(covered)
        
    # Convert to numpy arrays for faster operations? 
    # List of lists is probably fine for 2187 * 15 size.
    
    # Initial code: Seed's construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code_words.append(h + list(tail))
            
    # Map to indices
    initial_indices = set(word_to_idx(w) for w in initial_code_words)
    
    # State
    code_set = set(initial_indices)
    in_code = np.zeros(total_words, dtype=bool)
    for i in code_set:
        in_code[i] = True
        
    count = np.zeros(total_words, dtype=np.int32)
    for i in code_set:
        for j in neighbors[i]:
            count[j] += 1
            
    def is_valid():
        return np.all(count > 0)
        
    def greedy_removal():
        # Remove any word that is not critical
        # We iterate until no more can be removed
        changed = True
        while changed:
            changed = False
            # Randomize order to avoid bias
            items = list(code_set)
            random.shuffle(items)
            for i in items:
                if i not in code_set:
                    continue
                # Try removing i
                # Check if any neighbor has count 1
                removable = True
                for j in neighbors[i]:
                    if count[j] == 1:
                        removable = False
                        break
                if removable:
                    # Remove i
                    code_set.remove(i)
                    in_code[i] = False
                    for j in neighbors[i]:
                        count[j] -= 1
                    changed = True
                    # Note: We don't break out of the while loop immediately 
                    # because removing one might make others removable.
                    # But we do break out of the for loop? 
                    # Actually, if we remove i, counts change. 
                    # The current `items` list is static. 
                    # It's safer to restart the shuffle loop or just continue.
                    # Continuing is fine, we just check `if i not in code_set`.
                    
    # Run initial greedy removal
    greedy_removal()
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    # Local Search
    # Strategy: 
    # 1. Pick random i in code_set
    # 2. Remove i
    # 3. Find U (uncovered)
    # 4. If U empty, keep removal (size -1), run greedy_removal, continue
    # 5. Else, find j not in code_set such that U is subset of neighbors[j]
    # 6. If found, add j, run greedy_removal. 
    #    If size decreased, keep. 
    #    If size same, keep with some prob? Or revert? 
    #    Let's keep if size same to explore, but revert if we get stuck.
    #    Actually, if size is same, we might cycle. 
    #    Let's track best size found.
    
    best_size = len(code_set)
    best_code = set(code_set)
    
    # We will use a simple hill climbing with occasional restarts if stuck
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        if not code_set:
            break
            
        i = random.choice(list(code_set))
        
        # Tentatively remove i
        # Check criticality
        critical = False
        for j in neighbors[i]:
            if count[j] == 1:
                critical = True
                break
                
        if not critical:
            # Safe to remove
            code_set.remove(i)
            in_code[i] = False
            for j in neighbors[i]:
                count[j] -= 1
            greedy_removal()
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
            stuck_count = 0
            continue
            
        # Critical: Removing i causes some words to become uncovered
        # Find U
        U = []
        for j in neighbors[i]:
            if count[j] == 1:
                U.append(j)
                
        # We need to find a replacement k not in code_set
        # such that U is subset of neighbors[k]
        
        # Optimization: 
        # If U is empty, we handled it above.
        # If U is not empty, we need k to cover all of U.
        
        # Iterate through all possible k
        found_k = -1
        # We can filter candidates: k must be close to words in U?
        # Just iterate all 2187
        for k in range(total_words):
            if in_code[k]:
                continue
            # Check if k covers all U
            covers_all = True
            # We can precompute a mask for U? 
            # Or just check: is every u in U in neighbors[k]?
            # neighbors[k] is a list. 
            # For small U, checking `u in neighbors[k]` is O(len(neighbors[k])) = O(15).
            # Total O(2187 * |U| * 15). 
            # If |U| is small (usually 1-3), this is fast.
            for u in U:
                if u not in neighbors[k]:
                    covers_all = False
                    break
            if covers_all:
                found_k = k
                break
                
        if found_k != -1:
            # Swap i with found_k
            code_set.remove(i)
            in_code[i] = False
            for j in neighbors[i]:
                count[j] -= 1
                
            code_set.add(found_k)
            in_code[found_k] = True
            for j in neighbors[found_k]:
                count[j] += 1
                
            greedy_removal()
            
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                stuck_count = 0
            else:
                # Size did not improve
                # We might want to revert to avoid cycling
                # But maybe this new configuration allows future improvements?
                # Let's accept it with some probability or always accept to explore.
                # Given time limit, exploring is good.
                pass
        else:
            # No single swap works
            # Try a random restart? Or just continue.
            # To escape local minima, we can randomly add a word and then try to remove two?
            # For now, just continue.
            pass

    # Return the best code found
    final_indices = list(best_code)
    final_words = [idx_to_word(i) for i in final_indices]
    return final_words
