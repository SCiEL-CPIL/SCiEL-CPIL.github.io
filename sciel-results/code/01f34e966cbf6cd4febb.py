import time
import random
import numpy as np

_DEADLINE = 150.0

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5.0
    
    # 1. Generate all words
    words = []
    for i in range(q ** n):
        w = []
        for j in range(n):
            w.append((i // (q ** (n - 1 - j))) % q)
        words.append(tuple(w))
    
    m = len(words)
    
    # 2. Build coverage matrix C
    # C[i, j] is True if words[i] is within Hamming distance 'radius' of words[j]
    C = np.zeros((m, m), dtype=bool)
    # Optimization: Precompute neighbors or just loop
    # For m=729, double loop is fine.
    for i in range(m):
        w1 = words[i]
        for j in range(m):
            w2 = words[j]
            dist = 0
            # Unroll or use zip for speed? zip is fine.
            for a, b in zip(w1, w2):
                if a != b:
                    dist += 1
                    if dist > radius:
                        break
            if dist <= radius:
                C[i, j] = True
                
    # 3. Greedy Set Cover
    uncovered = np.ones(m, dtype=bool)
    chosen = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        # counts[i] = number of currently uncovered words covered by codeword i
        # C is (m, m), uncovered is (m,)
        # We need column sum of C * uncovered? 
        # C[i, j] * uncovered[j] -> sum over j
        # This is dot product of row i of C with uncovered
        counts = C @ uncovered.astype(np.int8)
        
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
            
        chosen.append(best_idx)
        # Update uncovered
        uncovered &= ~C[best_idx]
        
    # 4. Local Search: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen)):
            if time.time() > deadline:
                break
            # Create list without i
            # Using list comprehension is fast enough for small lists
            test_chosen = [x for k, x in enumerate(chosen) if k != i]
            
            if not test_chosen:
                continue
                
            # Check coverage
            # C[test_chosen] is a list of rows. np.any handles this.
            covered = np.any(C[test_chosen], axis=0)
            
            if np.all(covered):
                chosen = test_chosen
                improved = True
                break # Restart the for loop as list changed
                
    # 5. Local Search: Swaps (limited)
    # Try to find a swap that allows a subsequent removal
    if time.time() < deadline - 10:
        rng = random.Random(42)
        # Limit number of swap attempts to avoid timeout
        for _ in range(200): 
            if time.time() > deadline - 5:
                break
            if not chosen:
                break
            
            i = rng.randint(0, len(chosen) - 1)
            j = rng.randint(0, m - 1)
            
            if j in chosen:
                continue
                
            # Perform swap
            test_chosen = [x for k, x in enumerate(chosen) if k != i]
            test_chosen.append(j)
            
            # Check if valid
            covered = np.any(C[test_chosen], axis=0)
            if not np.all(covered):
                continue
                
            # Swap is valid. Try to remove one more
            found_removal = False
            for k in range(len(test_chosen)):
                if time.time() > deadline - 5:
                    break
                test_rem = [x for p, x in enumerate(test_chosen) if p != k]
                if not test_rem:
                    continue
                covered_rem = np.any(C[test_rem], axis=0)
                if np.all(covered_rem):
                    chosen = test_rem
                    found_removal = True
                    break
            
            if not found_removal:
                # If no removal, we can still accept the swap if we want to explore?
                # Usually we only accept if it leads to improvement (removal).
                # Otherwise we might cycle.
                pass

    result = [list(words[i]) for i in chosen]
    return result
