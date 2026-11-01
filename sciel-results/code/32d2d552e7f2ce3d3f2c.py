import itertools
import random
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But the function signature is general. We handle the specific case efficiently.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a simple construction for other cases
        # Direct sum with tetracode if possible
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: all words (trivially valid, but large)
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Specific case: q=3, n=6, radius=1
    # Total words: 3^6 = 729
    # Index mapping: word (w0, w1, w2, w3, w4, w5) -> index = w0 + 3*w1 + 9*w2 + 27*w3 + 81*w4 + 243*w5
    
    # Precompute masks
    masks = [0] * 729
    for w in itertools.product(range(3), repeat=6):
        idx = w[0] + 3*w[1] + 9*w[2] + 27*w[3] + 81*w[4] + 243*w[5]
        mask = 1 << idx
        for i in range(6):
            for val in range(3):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                new_idx = new_w[0] + 3*new_w[1] + 9*new_w[2] + 27*new_w[3] + 81*new_w[4] + 243*new_w[5]
                mask |= (1 << new_idx)
        masks[idx] = mask
    
    all_mask = (1 << 729) - 1
    
    best_solution = None
    best_size = 999
    
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # Greedy with random tie-breaking
    for restart in range(10000):
        if time.time() > deadline:
            break
        
        uncovered = all_mask
        chosen = set()
        solution = []
        
        while uncovered:
            max_gain = -1
            candidates = []
            for i in range(729):
                if i in chosen:
                    continue
                gain = (masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain:
                    candidates.append(i)
            
            if max_gain == 0:
                break # Should not happen
            
            # Pick randomly from candidates with max gain
            c = random.choice(candidates)
            chosen.add(c)
            solution.append(c)
            uncovered &= ~masks[c]
            
        # Reduction step: try to remove any codeword
        improved = True
        while improved:
            improved = False
            for i in range(len(solution)-1, -1, -1):
                # Check if solution without solution[i] still covers all
                # Reconstruct mask
                mask_sum = 0
                for j in range(len(solution)):
                    if j != i:
                        mask_sum |= masks[solution[j]]
                if mask_sum == all_mask:
                    solution.pop(i)
                    improved = True
                    break
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
            # If we reach the known best, we might stop early or keep looking for better
            if best_size == 73:
                # We found a solution of size 73. Can we find 72 or 71?
                # Lower bound is 71. Let's keep searching for a bit.
                pass
        
        # Convert indices to words
        # We need to return a list of lists of integers
        # idx -> word
        def idx_to_word(idx):
            w = [0]*6
            for i in range(6):
                w[i] = (idx // (3**i)) % 3
            return w
        
        # Store best_solution as indices, convert at the end
        # Actually, let's store the indices and convert once.
        
    if best_solution is None:
        # Fallback
        return [idx_to_word(i) for i in range(729)]
        
    return [idx_to_word(i) for i in best_solution]
