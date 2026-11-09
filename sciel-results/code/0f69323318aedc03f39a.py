import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Specific parameters for this task: q=3, n=8, radius=1
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        # For q=3, n >= 4, use tetracode construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Generic small case fallback
        centers = []
        covered = set()
        all_words = set(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != all_words and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers

    # --- Optimized Path for q=3, n=8, radius=1 ---
    
    total_words = 3 ** 8  # 6561
    code_size_limit = 486 # Best known, we aim to beat this
    
    # 1. Pre-compute mapping and neighbors
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    # Pre-compute neighbors for each word (including itself)
    # neighbors[i] is a list of indices of words within distance 1 of idx_to_word[i]
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
    
    # 2. Initial Code: Random Start
    # We start with a random set of size slightly larger than the target to allow room for optimization.
    # Or we can start with the Tetracode direct sum (729) and reduce.
    # Let's try a random start of size 450 to see if SA can find a valid code.
    # If it fails to find a valid code, we might need to start higher.
    # However, random 450 is likely invalid.
    # Let's start with the 729 construction (Tetracode x {0,1,2}^4) as it is guaranteed valid.
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code_indices = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=4):
            full_word = t + list(tail)
            idx = word_to_idx[tuple(full_word)]
            initial_code_indices.append(idx)
            
    code_set = set(initial_code_indices)
    
    # 3. Simulated Annealing with Vectorized Moves
    
    # counts[w] = number of centers in code_set covering w
    counts = np.zeros(total_words, dtype=np.int32)
    
    # Vectorized update helper
    # We maintain counts as a numpy array.
    # To update counts when adding/removing a center, we need to increment/decrement
    # the counts for all neighbors of that center.
    
    def add_center(idx):
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
        code_set.add(idx)
        
    def remove_center(idx):
        for n_idx in neighbors[idx]:
            counts[n_idx] -= 1
        code_set.remove(idx)

    for idx in code_set:
        add_center(idx)
        
    best_len = len(code_set)
    best_code = set(code_set)
    
    # SA Parameters
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    all_indices = list(range(total_words))
    
    # Pre-compute neighbor lists as numpy arrays for faster access?
    # neighbors is a list of lists. Accessing neighbors[idx] is fast.
    
    while time.time() < deadline - 2:
        
        # Generate a random move
        # Move types:
        # 1. Add a random center (if size < best_len or random)
        # 2. Remove a random center (if it doesn't break validity)
        # 3. Swap a random center with another
        
        move_type = random.random()
        
        # Check current validity: min(counts) > 0
        # We maintain counts, so we can check min quickly?
        # np.min(counts) is O(N). N=6561. Fast.
        
        if move_type < 0.3:
            # Try to remove a center
            # Pick a random center
            if len(code_set) == 0: continue
            c_remove = random.choice(list(code_set))
            
            # Check if removal is valid:
            # All neighbors of c_remove must have count > 1 (so after removal, count >= 1)
            # i.e., min(counts[neighbors[c_remove]]) > 1
            n_list = neighbors[c_remove]
            # Convert to numpy array for min?
            # counts is a numpy array.
            # n_list is a list of ints.
            # counts[n_list] creates a view/copy.
            min_val = np.min(counts[n_list])
            
            if min_val > 1:
                # Valid removal
                remove_center(c_remove)
                new_len = len(code_set)
                if new_len < best_len:
                    best_len = new_len
                    best_code = set(code_set)
                # Accept always (since it reduces size and maintains validity)
                # Note: In SA, we might accept worse moves, but removal is always "better" in size.
                # However, it might lead to a dead end where we can't reduce further.
                # But since we are minimizing size, and validity is preserved, this is a good move.
                
        elif move_type < 0.6:
            # Try to add a center (only if we are currently valid and want to explore larger states? 
            # Or if we are stuck? 
            # Actually, adding a center increases size. We should only do this if it allows 
            # subsequent removals. 
            # Let's only add if current size > best_len? No, that's not helpful.
            # Let's skip pure additions to keep size down. 
            # Instead, let's do swaps.
            pass
        else:
            # Swap: Remove one, Add one
            if len(code_set) == 0: continue
            c_remove = random.choice(list(code_set))
            c_add = random.choice(all_indices)
            
            if c_add in code_set:
                continue
                
            # Check validity of swap:
            # 1. Remove c_remove.
            # 2. Add c_add.
            # Result must be valid.
            
            # Words only covered by c_remove (counts[w]==1) must be covered by c_add.
            # Words covered by c_add but not c_remove don't matter for validity (they just increase count).
            
            # Let's calculate the "delta" for counts.
            # For w in N(c_remove): counts[w] - 1
            # For w in N(c_add): counts[w] + 1
            # Intersection N(c_remove) & N(c_add): net change 0.
            # Only N(c_remove) \ N(c_add) decreases.
            # Only N(c_add) \ N(c_remove) increases.
            
            # We need min(counts[w] - 1) > 0 for all w in N(c_remove) \ N(c_add).
            # i.e., counts[w] > 1 for all w in N(c_remove) that are NOT in N(c_add).
            
            # Efficient check:
            # 1. Identify words in N(c_remove) with count == 1.
            # 2. Check if any of them are NOT in N(c_add).
            
            n_remove = neighbors[c_remove]
            n_add = neighbors[c_add]
            
            # Create a set of n_add for O(1) lookup?
            # n_add has size 18.
            set_add = set(n_add)
            
            valid = True
            for w in n_remove:
                if counts[w] == 1:
                    if w not in set_add:
                        valid = False
                        break
                        
            if valid:
                # Perform swap
                remove_center(c_remove)
                add_center(c_add)
                # Size unchanged.
                # Accept based on SA?
                # Since size is unchanged, we accept if it improves "diversity" or just accept to escape local minima.
                # Let's always accept valid swaps to explore the space.
                
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min # Floor temperature to keep exploring
            
    # Return best code found
    result = [list(idx_to_word[idx]) for idx in best_code]
    return result
