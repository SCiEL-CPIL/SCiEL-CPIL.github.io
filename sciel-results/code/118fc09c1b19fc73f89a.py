import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute covers - for each word, find all words within radius
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = {idx}
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.add(word_to_idx[tuple(nw)])
        covers.append(list(cover_set))
        
    covers_np = [np.array(c, dtype=np.int32) for c in covers]
    
    # Build Incidence Matrix A
    # A[i, j] = 1 if j is in covers[i]
    A = np.zeros((N, N), dtype=np.int8)
    for i, cover_set in enumerate(covers):
        A[i, cover_set] = 1
        
    covered = np.zeros(N, dtype=bool)
    selected_indices = []
    
    # Greedy Set Cover with priority queue optimization
    while not covered.all():
        if time.time() > deadline - 10.0:
            break
            
        uncovered = ~covered
        # scores[i] = number of uncovered words covered by codeword i
        scores = A @ uncovered
        # Mask out already selected
        if selected_indices:
            scores[selected_indices] = 0
            
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        covered[covers_np[best_idx]] = True
        
    # If we didn't finish greedy, continue with safety margin
    if not covered.all():
        while not covered.all() and time.time() < deadline - 2.0:
            uncovered = ~covered
            scores = A @ uncovered
            if selected_indices:
                scores[selected_indices] = 0
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            selected_indices.append(best_idx)
            covered[covers_np[best_idx]] = True
            
    # Local Search Phase - Remove redundant codewords
    if time.time() < deadline - 5.0:
        is_selected = np.zeros(N, dtype=bool)
        for idx in selected_indices:
            is_selected[idx] = True
            
        # Calculate initial counts
        counts = A[:, is_selected].sum(axis=1).astype(np.int32)
        
        def try_remove(idx):
            if not is_selected[idx]: 
                return False
            if np.any(counts[covers_np[idx]] == 1):
                return False
            is_selected[idx] = False
            counts[covers_np[idx]] -= 1
            return True

        # Greedy Remove Pass - multiple iterations
        for _ in range(3):
            changed = True
            while changed:
                changed = False
                for idx in np.where(is_selected)[0]:
                    if try_remove(idx):
                        changed = True
        
        # Local Search Swaps - try to replace codewords to enable more removals
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
                
            critical_words = np.where(counts == 1)[0]
            if len(critical_words) == 0:
                break
            
            w_idx = random.choice(critical_words)
            
            covering_c = -1
            for c in covers[w_idx]:
                if is_selected[c]:
                    covering_c = c
                    break
            
            if covering_c == -1:
                continue
                
            replacements = [c for c in covers[w_idx] if not is_selected[c]]
            if not replacements:
                continue
                
            new_c = random.choice(replacements)
            
            counts_copy = counts.copy()
            counts_copy[covers_np[covering_c]] -= 1
            counts_copy[covers_np[new_c]] += 1
            
            if np.any(counts_copy == 0):
                continue
            else:
                is_selected[covering_c] = False
                is_selected[new_c] = True
                counts = counts_copy
                
                changed = True
                while changed:
                    changed = False
                    for idx in np.where(is_selected)[0]:
                        if try_remove(idx):
                            changed = True

    # Final validation
    if time.time() < deadline - 1.0:
        if not (counts > 0).all():
            uncovered = np.where(counts == 0)[0]
            for w_idx in uncovered:
                for c in covers[w_idx]:
                    if not is_selected[c]:
                        is_selected[c] = True
                        counts[covers_np[c]] += 1
                        break

    selected_indices = np.where(is_selected)[0].tolist()
    return [all_words[idx].tolist() for idx in selected_indices]
