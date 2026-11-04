import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Handle the specific high-interest case
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Fallback for other parameters
    if q == 3:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Simple fallback: return all words (valid, just not minimal)
    return [list(w) for w in itertools.product(range(q), repeat=n)]

def _construct_3_6_1():
    q = 3
    n = 6
    N = q ** n
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Generate all words as a numpy array
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Build neighbors
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    num_neighbors = 1 + n * (q - 1)
    neighbors = np.empty((N, num_neighbors), dtype=np.int32)
    
    for i in range(N):
        c = words[i]
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            for new_val in range(q):
                if new_val == c[pos]:
                    continue
                new_w = list(c)
                new_w[pos] = new_val
                neighbors[i, idx] = word_to_idx[tuple(new_w)]
                idx += 1
                
    # Helper: Run local search on a given selected set
    def local_search(selected, cov, deadline):
        # 1. Reduction phase
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(selected)):
                j = selected[idx]
                # Check if redundant
                if np.all(cov[neighbors[j, :]] > 1):
                    cov[neighbors[j, :]] -= 1
                    selected.pop(idx)
                    changed = True
                    break # Restart loop as indices changed
        
        # 2. Swap phase (Hill Climbing)
        # We try to swap one word for another to allow further reductions
        # Since N is small, we can try random swaps
        for _ in range(1000): # Limit iterations for this phase
            if time.time() > deadline:
                break
            if len(selected) == 0:
                break
                
            # Try to remove a word and see if we can add a different one to cover the gaps
            # Actually, standard swap: remove i, add j, then reduce.
            # To be faster, we can just try random swaps and accept if size decreases after reduction.
            
            i_idx = random.randint(0, len(selected) - 1)
            i_word = selected[i_idx]
            
            # Pick random j
            j_word = random.randint(0, N - 1)
            if j_word in selected:
                continue
                
            # Temporarily remove i
            cov[neighbors[i_word, :]] -= 1
            
            # Check if valid
            if np.any(cov < 1):
                cov[neighbors[i_word, :]] += 1
                continue
                
            # Add j
            cov[neighbors[j_word, :]] += 1
            
            # Try to reduce
            temp_selected = list(selected)
            temp_selected[i_idx] = j_word
            temp_cov = cov.copy()
            
            # Reduce temp_selected
            changed = True
            while changed:
                changed = False
                for idx in range(len(temp_selected)):
                    j = temp_selected[idx]
                    if np.all(temp_cov[neighbors[j, :]] > 1):
                        temp_cov[neighbors[j, :]] -= 1
                        temp_selected.pop(idx)
                        changed = True
                        break
                        
            if len(temp_selected) < len(selected):
                # Accept
                selected = temp_selected
                cov = temp_cov
            else:
                # Revert
                cov = cov.copy() # Wait, we modified cov in place. We need to revert.
                # This is inefficient. Let's use a different approach for swaps.
                # Instead of copying, we can just track the delta.
                # But for simplicity and given N=729, let's just revert manually.
                cov[neighbors[j_word, :]] -= 1
                cov[neighbors[i_word, :]] += 1
                
        return selected, cov

    best_selected = None
    best_cov = None
    best_size = float('inf')
    
    # Run multiple randomized greedy starts
    num_starts = 20
    for k in range(num_starts):
        if time.time() > deadline - 10: # Leave time for local search
            break
            
        # Randomized Greedy
        uncovered = np.ones(N, dtype=bool)
        selected = []
        
        while uncovered.any() and time.time() < deadline:
            counts = np.sum(uncovered[neighbors], axis=1)
            # Add noise to break ties and explore
            scores = counts.astype(float) + np.random.uniform(0, 1, N)
            best_idx = int(np.argmax(scores))
            
            if counts[best_idx] == 0:
                break
            
            selected.append(best_idx)
            uncovered[neighbors[best_idx, :]] = False
            
        # Initialize coverage
        cov = np.zeros(N, dtype=np.int32)
        for j in selected:
            cov[neighbors[j, :]] += 1
            
        # Local Search
        selected, cov = local_search(selected, cov, deadline - 5)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected
            best_cov = cov
            
        # Early stop if we hit the theoretical lower bound or known optimal
        if best_size <= 73:
            break

    if best_selected is None:
        # Fallback to deterministic greedy if no random runs completed
        uncovered = np.ones(N, dtype=bool)
        selected = []
        while uncovered.any():
            counts = np.sum(uncovered[neighbors], axis=1)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] == 0: break
            selected.append(best_idx)
            uncovered[neighbors[best_idx, :]] = False
        best_selected = selected

    return [words[i].tolist() for i in best_selected]
