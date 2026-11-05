import time
import random
import itertools

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, or at least not crash.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we should assume q=3, n=6, radius=1 but write it generally enough or check.
    # Given the complexity, we might restrict to the specific case or use a general fallback.
    # The seed program used a specific construction for n=4+...
    # If n is different, the greedy approach still works.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback to a simple construction if space is too large for this specific solver
        # But for q=3, n=6, total is 729.
        # For q=3, n=7, total is 2187.
        # For q=3, n=8, total is 6561.
        # Greedy might be too slow for n=8 with Python loops if not optimized.
        # But the task is specifically K_3(6,1).
        pass

    # Generate all words
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors
    # neighbors[i] is a bitmask of words within distance 'radius' of words[i]
    # Since radius is likely 1, we can compute efficiently.
    # For general radius, it's harder.
    # Let's assume radius is small (1 or 2).
    
    masks = [0] * total_words
    
    # Helper to convert word to index
    def w_to_idx(w):
        return word_to_idx[w]
        
    # Compute masks
    # Optimization: only compute for radius 1?
    # The task is radius 1.
    # If radius != 1, we might need a different approach.
    # Let's assume radius=1 for the main logic.
    
    if radius == 1:
        for i, w in enumerate(words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = w_to_idx(tuple(new_w))
                    mask |= (1 << idx)
            masks[i] = mask
    else:
        # Fallback: simple greedy without precomputed masks?
        # Or compute masks on the fly?
        # For now, assume radius=1 is the main case.
        # If not, use a naive distance check.
        for i, w in enumerate(words):
            mask = 0
            for j, v in enumerate(words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            masks[i] = mask

    best_code = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Run multiple randomized greedy iterations
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            # Find best candidate
            best_score = -1
            best_candidates = []
            
            # Optimization: if uncovered is small, we can check all.
            # If uncovered is large, checking all 729 is fine.
            for i in range(total_words):
                # If i is already in code, skip? 
                # Actually, we don't need to skip, but adding same word twice is useless.
                # We can maintain a set of indices in code.
                # But checking popcount is cheap.
                score = (masks[i] & uncovered).bit_count()
                if score > best_score:
                    best_score = score
                    best_candidates = [i]
                elif score == best_score and score > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if uncovered is not 0
            
            # Randomly pick from best candidates to add diversity
            # If we want deterministic, pick best_candidates[0]
            # Let's mix: 50% random, 50% first
            if random.random() < 0.5:
                chosen = random.choice(best_candidates)
            else:
                chosen = best_candidates[0]
            
            code.append(chosen)
            uncovered &= ~masks[chosen]
            
        # Prune phase
        changed = True
        while changed:
            changed = False
            for idx in range(len(code)):
                w_idx = code[idx]
                # Check if w_idx is redundant
                # Union of all other masks
                union_mask = 0
                for jdx in range(len(code)):
                    if jdx == idx:
                        continue
                    union_mask |= masks[code[jdx]]
                
                # If w_idx covers nothing unique, remove it
                if (masks[w_idx] & ~union_mask) == 0:
                    code.pop(idx)
                    changed = True
                    break # Restart loop as indices changed
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code[:]
            
        # If we reached a very good solution, we can stop early?
        # The theoretical min is 73. If we hit 73, we can't do much better.
        if best_len == 73:
            break

    if best_code is None:
        # Fallback to the seed program's logic if something went wrong
        # Or just return an empty list? No, must be valid.
        # Use the direct sum construction from seed as fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, just return all words?
            return [list(w) for w in words]
            
    return [list(words[i]) for i in best_code]
