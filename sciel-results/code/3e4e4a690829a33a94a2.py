"""
Optimized solver for K_3(7,1) using bitset-based greedy removal.
"""
import time
import random
import itertools

def construct(q, n, radius):
    # Parameters specific to the task: q=3, n=7, radius=1
    # Total words in space: 3^7 = 2187
    total_words = q ** n
    # Deadline: 115 seconds to leave margin for Python overhead and return
    deadline = time.time() + 115.0
    
    # Precompute mapping from word (tuple of digits) to index
    # and from index to word
    # We represent the space as integers 0..2186
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls for each word
    # ball[i] is a Python integer where bit j is 1 if word j is within distance 1 of word i
    balls = [0] * total_words
    for i, w in enumerate(all_words):
        mask = 1 << i
        # Distance 0
        # Distance 1: change one position
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                mask |= (1 << j)
        balls[i] = mask
        
    # Initial code: Direct sum of Tetracode (9 words) and All-3s (27 words) -> 243 words
    # Tetracode: [a, b, (a+b)%3, (a+2b)%3] for a,b in 0..2
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
            
    initial_code = []
    for t in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = t + tail
            initial_code.append(word_to_idx[w])
            
    # Shuffle to randomize removal order
    random.shuffle(initial_code)
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    def is_covering(code_indices):
        union_mask = 0
        for idx in code_indices:
            union_mask |= balls[idx]
        # Check if union covers all 2187 bits
        # (1 << 2187) - 1 is the full mask
        return (union_mask == (1 << total_words) - 1)
        
    def prune_code(code_indices):
        # Remove redundant codewords
        # We maintain a set of indices for O(1) lookup and removal
        # But for bitset union, we just sum them up.
        # To check redundancy of c: 
        # union_others = union_all ^ balls[c]  (since balls[c] is subset of union_all if covered)
        # Actually, union_all includes balls[c].
        # So union_others = union_all & ~balls[c]
        # If balls[c] & ~union_others == 0, then c is redundant.
        
        code_set = set(code_indices)
        # Calculate initial union
        union_mask = 0
        for idx in code_set:
            union_mask |= balls[idx]
            
        changed = True
        # We need to be careful with the order of removal. 
        # If we remove one, union changes.
        # A simple iterative pass:
        while changed:
            changed = False
            # Iterate over a copy of the list to allow modification
            # Sorting might help with determinism but random is better for escaping local minima
            # We iterate over the current set
            for idx in list(code_set):
                # Check if idx is redundant
                # union_without_idx = union_mask & ~balls[idx]
                # If balls[idx] is a subset of union_without_idx, it's redundant.
                # Equivalent to: (balls[idx] & ~union_without_idx) == 0
                # ~union_without_idx in Python integers is tricky due to infinite bits.
                # Better: (balls[idx] | union_without_idx) == union_without_idx
                # Or simply: (balls[idx] & (union_mask ^ balls[idx])) == 0 is WRONG because union_mask might not contain balls[idx] if it was already removed? No, we maintain union_mask as union of CURRENT code_set.
                
                # Let's use:
                # others_mask = union_mask & ~balls[idx]
                # if (balls[idx] & ~others_mask) == 0: redundant
                # Note: ~others_mask has 1s where others_mask has 0s.
                # balls[idx] & ~others_mask checks bits in balls[idx] that are NOT in others_mask.
                # If this is 0, balls[idx] is subset of others_mask.
                
                others_mask = union_mask & ~balls[idx]
                if (balls[idx] & ~others_mask) == 0:
                    # Redundant
                    code_set.remove(idx)
                    union_mask = others_mask # Since balls[idx] was subset, removing it doesn't change the union
                    changed = True
                    if time.time() > deadline:
                        return list(code_set)
                        
        return list(code_set)
    
    # Main Search Loop
    current_code = initial_code[:]
    
    # First, prune the initial code
    current_code = prune_code(current_code)
    if len(current_code) < best_size:
        best_code = current_code[:]
        best_size = len(best_code)
        
    # Iterative improvement: Random restarts / kicks
    # Since 243 is the start, and 186 is target, we need to remove ~57 words.
    # The pruning process finds a minimal code. Different orders might yield different minimal codes.
    
    iterations = 0
    while time.time() < deadline:
        # Try a random permutation of the current best code (or a subset?)
        # Actually, starting from the best code found so far is good.
        # Let's shuffle the best code and prune again.
        shuffled = best_code[:]
        random.shuffle(shuffled)
        
        pruned = prune_code(shuffled)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned[:]
            # If we found a new best, we might want to continue from here.
            # The loop naturally does this by using best_code in the next iteration.
            
        # Optional: "Kick" strategy
        # Add a random word, then prune. This can help escape local minima of the "minimal" set.
        # However, adding a word increases size, so we only do this if we have time and are stuck.
        # For now, simple shuffling and pruning is the baseline.
        
        # If we are stuck at a size, maybe try adding a random word and pruning?
        # Let's check if we've been stuck for a while.
        # For simplicity and speed, we just keep shuffling and pruning.
        # The bitset operations are very fast, so we can do many iterations.
        
        iterations += 1
        
        # If we hit the deadline, break
        if time.time() > deadline - 2:
            break
            
    # Convert indices back to words
    # Ensure distinctness (set guarantees this)
    return [all_words[i] for i in best_code]
