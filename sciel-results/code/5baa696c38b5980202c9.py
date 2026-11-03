import time
import random
import itertools

def construct(q, n, radius):
    # Parameters for the specific task
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters, though the task specifies this one
        # Just return a trivial covering (all words) if parameters differ significantly
        # But the contract says evaluator passes this task's q, n, radius.
        # We assume q=3, n=6, radius=1.
        pass

    # Precomputation
    # Total words
    total_words = q ** n
    # Convert word (tuple of ints) to index
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

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors for each word index
    # Neighbors are words at Hamming distance <= 1
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = all_words[i]
        neighbors[i].append(i) # distance 0
        for pos in range(n):
            for val in range(q):
                if w[pos] == val:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neighbors[i].append(word_to_idx(tuple(new_w)))
        
    # Initial Code: Direct Sum of Tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    # Extend to length 6 by appending all combinations of length 2
    init_code_idx = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            init_code_idx.append(word_to_idx(tuple(word)))
    
    # Ensure unique
    init_code_idx = list(set(init_code_idx))
    current_code = init_code_idx.copy()
    
    # Coverage Count
    cover_count = [0] * total_words
    for c in current_code:
        for nb in neighbors[c]:
            cover_count[nb] += 1

    def is_removable(c, cover_count):
        for nb in neighbors[c]:
            if cover_count[nb] <= 1:
                return False
        return True

    def remove_code(c, cover_count):
        for nb in neighbors[c]:
            cover_count[nb] -= 1

    def add_code(c, cover_count):
        for nb in neighbors[c]:
            cover_count[nb] += 1

    best_size = len(current_code)
    best_code = current_code.copy()
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Local Search with Swaps
    # We will perform a series of iterations.
    # In each iteration, we try to remove redundant words.
    # If no word can be removed, we perform a random swap to change the landscape.
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        changed = True
        # Greedy Removal Phase
        while changed:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(current_code)
            for c in current_code:
                if is_removable(c, cover_count):
                    remove_code(c, cover_count)
                    current_code.remove(c)
                    changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code.copy()
        
        # If we reached 73, we can stop early or continue to ensure validity
        if best_size == 73:
            break
            
        # Perturbation Phase: Swap
        if len(current_code) > 0:
            # Pick a random codeword to replace
            c_old = random.choice(current_code)
            # Pick a random candidate from the whole space
            # To make it more effective, we could pick candidates that cover "sparse" areas
            # but random is a good baseline.
            c_new = random.randint(0, total_words - 1)
            
            if c_new != c_old and c_new not in current_code:
                # Perform swap
                remove_code(c_old, cover_count)
                add_code(c_new, cover_count)
                current_code.remove(c_old)
                current_code.append(c_new)
                
                # Try to remove again after swap
                changed = True
                while changed:
                    changed = False
                    random.shuffle(current_code)
                    for c in current_code:
                        if is_removable(c, cover_count):
                            remove_code(c, cover_count)
                            current_code.remove(c)
                            changed = True
                            if len(current_code) < best_size:
                                best_size = len(current_code)
                                best_code = current_code.copy()
                            if best_size == 73:
                                break
                    if best_size == 73:
                        break

    # Convert best_code indices back to words
    result = [all_words[i] for i in best_code]
    return result
