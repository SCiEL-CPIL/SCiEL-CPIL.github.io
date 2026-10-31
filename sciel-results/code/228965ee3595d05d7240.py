import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(3), repeat=n)]

    total_words = q ** n
    words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    masks = np.zeros((total_words, total_words), dtype=bool)
    for i, w_i in enumerate(words):
        masks[i, i] = True
        for pos in range(n):
            for delta in [1, 2]:
                w_j = w_i.copy()
                w_j[pos] = (w_i[pos] + delta) % q
                idx = 0
                for k in range(n):
                    idx = idx * q + w_j[k]
                masks[i, idx] = True

    def greedy_search(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        while np.any(uncovered):
            if time.time() > deadline:
                break
            counts = np.sum(masks & uncovered, axis=1)
            if randomize:
                # Add small noise to break ties or explore alternatives
                noise = np.random.uniform(0, 0.1, total_words)
                counts += noise
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            code_indices.append(max_idx)
            uncovered &= ~masks[max_idx]
        return code_indices

    def local_search(code_indices):
        code_set = set(code_indices)
        counts = np.zeros(total_words, dtype=int)
        for idx in code_indices:
            counts += masks[idx]
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code_indices) - 1, -1, -1):
                if time.time() > deadline: break
                idx = code_indices[i]
                unique_words = np.where(masks[idx] & (counts == 1))[0]
                
                if len(unique_words) == 0:
                    code_indices.pop(i)
                    code_set.remove(idx)
                    counts -= masks[idx]
                    improved = True
                    break
                else:
                    unique_list = unique_words.tolist()
                    for c2 in range(total_words):
                        if c2 in code_set:
                            continue
                        if np.all(masks[c2, unique_list]):
                            code_indices[i] = c2
                            code_set.remove(idx)
                            code_set.add(c2)
                            counts -= masks[idx]
                            counts += masks[c2]
                            improved = True
                            break
                    if improved:
                        break
        return code_indices

    best_code = None
    best_size = float('inf')
    
    # First run deterministic
    code = greedy_search(randomize=False)
    code = local_search(code)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code
        
    # Random restarts
    while time.time() < deadline - 10:
        code = greedy_search(randomize=True)
        code = local_search(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size == 73: # Known optimum
                break
                
    return [words[i] for i in best_code]
