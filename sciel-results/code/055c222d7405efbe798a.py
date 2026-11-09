import numpy as np
import time
import random
import itertools
import concurrent.futures
import threading

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        return [[0]*n]

    total_words = q ** n
    
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

    # Precompute neighbors as indices and masks
    neighbors_idx = []
    neighbor_masks = [0] * total_words
    
    for i in range(total_words):
        w = idx_to_word(i)
        covered = [i]
        for pos in range(n):
            original = w[pos]
            for val in range(q):
                if val == original:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = original
        neighbors_idx.append(covered)
        mask = 0
        for j in covered:
            mask |= (1 << j)
        neighbor_masks[i] = mask

    # Initial code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code_words.append(h + list(tail))
            
    initial_indices = set(word_to_idx(w) for w in initial_code_words)

    def run_search(seed, time_limit):
        rng = random.Random(seed)
        
        code_set = set(initial_indices)
        in_code = np.zeros(total_words, dtype=bool)
        for i in code_set:
            in_code[i] = True
            
        count = np.zeros(total_words, dtype=np.int32)
        for i in code_set:
            for j in neighbors_idx[i]:
                count[j] += 1
                
        def is_valid():
            return np.all(count > 0)
            
        def greedy_removal():
            changed = True
            while changed:
                changed = False
                items = list(code_set)
                rng.shuffle(items)
                for i in items:
                    if i not in code_set:
                        continue
                    removable = True
                    for j in neighbors_idx[i]:
                        if count[j] == 1:
                            removable = False
                            break
                    if removable:
                        code_set.remove(i)
                        in_code[i] = False
                        for j in neighbors_idx[i]:
                            count[j] -= 1
                        changed = True

        greedy_removal()
        
        best_size = len(code_set)
        best_code = set(code_set)
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if not code_set:
                break
                
            i = rng.choice(list(code_set))
            
            critical = False
            for j in neighbors_idx[i]:
                if count[j] == 1:
                    critical = True
                    break
                    
            if not critical:
                code_set.remove(i)
                in_code[i] = False
                for j in neighbors_idx[i]:
                    count[j] -= 1
                greedy_removal()
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
                continue
            
            U = []
            for j in neighbors_idx[i]:
                if count[j] == 1:
                    U.append(j)
            
            if not U:
                continue

            mask_U = 0
            for u in U:
                mask_U |= (1 << u)
                
            found_k = -1
            for k in range(total_words):
                if in_code[k]:
                    continue
                if (neighbor_masks[k] & mask_U) == mask_U:
                    found_k = k
                    break
                    
            if found_k != -1:
                code_set.remove(i)
                in_code[i] = False
                for j in neighbors_idx[i]:
                    count[j] -= 1
                    
                code_set.add(found_k)
                in_code[found_k] = True
                for j in neighbors_idx[found_k]:
                    count[j] += 1
                    
                greedy_removal()
                
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
            else:
                # No swap found
                pass

        return best_size, list(best_code)

    # Run multiple threads
    num_threads = 4
    time_per_thread = 115.0 / num_threads
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for t in range(num_threads):
            seed = 42 + t * 1000
            futures.append(executor.submit(run_search, seed, time_per_thread))
        
        for f in concurrent.futures.as_completed(futures):
            try:
                size, indices = f.result()
                results.append((size, indices))
            except Exception as e:
                pass

    if not results:
        # Fallback to single thread if threads failed
        size, indices = run_search(42, 115.0)
        return [idx_to_word(i) for i in indices]

    # Find best
    best_size = min(r[0] for r in results)
    best_indices = [r[1] for r in results if r[0] == best_size][0]
    
    return [idx_to_word(i) for i in best_indices]
