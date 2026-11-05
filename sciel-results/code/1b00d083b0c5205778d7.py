import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = [[] for _ in range(len(all_words))]
        for i, w in enumerate(all_words):
            neighbors[i].append(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    neighbors[i].append(word_to_idx[tuple(new_w)])
                    
        codewords = []
        counts = np.zeros(len(all_words), dtype=np.int32)
        # Greedy fill
        uncovered = set(range(len(all_words)))
        while uncovered:
            best_idx = -1
            best_count = -1
            # Sample to save time
            for _ in range(100):
                idx = random.randint(0, len(all_words)-1)
                cnt = sum(1 for nb in neighbors[idx] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx == -1: break
            codewords.append(list(all_words[best_idx]))
            for nb in neighbors[best_idx]:
                counts[nb] += 1
                if counts[nb] >= 1:
                    uncovered.discard(nb)
                    
        # Greedy remove
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(codewords)-1, -1, -1):
                idx = word_to_idx[tuple(codewords[i])]
                if np.min(counts[neighbors[idx]]) >= 1:
                    for nb in neighbors[idx]:
                        counts[nb] -= 1
                    codewords.pop(i)
                    changed = True
        return codewords

    # Specific q=3, n=7, radius=1
    all_words = list(itertools.product(range(3), repeat=7))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(7):
            for val in range(3):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                neighbors[i].append(word_to_idx[tuple(new_w)])
                
    # Initial: 243 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    
    counts = np.zeros(num_words, dtype=np.int32)
    for c in codewords:
        idx = word_to_idx[tuple(c)]
        for nb in neighbors[idx]:
            counts[nb] += 1
            
    def is_valid():
        return np.min(counts) >= 1

    while time.time() < deadline:
        # 1. Prune
        pruned = False
        # Shuffle indices to randomize order
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        for i in indices:
            idx = word_to_idx[tuple(codewords[i])]
            if np.min(counts[neighbors[idx]]) >= 1:
                for nb in neighbors[idx]:
                    counts[nb] -= 1
                codewords.pop(i)
                pruned = True
                break # Restart prune loop
        
        if not pruned:
            # 2. Swap
            if len(codewords) == 0: break
            i = random.randint(0, len(codewords) - 1)
            c_old = codewords[i]
            idx_old = word_to_idx[tuple(c_old)]
            
            # Remove
            for nb in neighbors[idx_old]:
                counts[nb] -= 1
            codewords.pop(i)
            
            if is_valid():
                continue # Successfully removed
            
            # Invalid, find replacement
            uncovered = np.where(counts == 0)[0]
            
            found = False
            if len(uncovered) > 0:
                # Pick a random uncovered word to guide search
                u = random.choice(uncovered.tolist())
                candidates = neighbors[u]
                random.shuffle(candidates)
                
                for idx_new in candidates:
                    # Check if idx_new covers all uncovered
                    ok = True
                    for uu in uncovered:
                        if uu not in neighbors[idx_new]:
                            ok = False
                            break
                    if ok:
                        # Accept swap
                        for nb in neighbors[idx_new]:
                            counts[nb] += 1
                        codewords.append(list(all_words[idx_new]))
                        found = True
                        break
            
            if not found:
                # Restore
                for nb in neighbors[idx_old]:
                    counts[nb] += 1
                codewords.insert(i, c_old)
                
                # If we are stuck, we might want to try a "2-swap" or just retry.
                # To avoid infinite loops on stuck states, we can add a small penalty or just let it retry.
                # Given the small space, random restarts of the swap process usually work.
                
    return [list(c) for c in codewords]
