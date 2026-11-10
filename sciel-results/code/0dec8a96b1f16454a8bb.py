import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_words = q ** n  # 2187

    # Precompute all words
    all_words = []
    for i in range(total_words):
        w = [0] * n
        idx = i
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        all_words.append(w)

    # Precompute spheres as lists of indices
    spheres = []
    for i in range(total_words):
        w = all_words[i]
        s = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    nw = w.copy()
                    nw[pos] = new_val
                    j = 0
                    for d in nw:
                        j = j * q + d
                    s.append(j)
        spheres.append(s)

    # Convert spheres to numpy arrays for fast operations
    sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]
    
    # Precompute sphere bitmasks for greedy
    sphere_bits = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in spheres[i]:
            mask |= (1 << j)
        sphere_bits[i] = mask

    # Precompute coverers: coverers[j] = list of i such that j in spheres[i]
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        for j in spheres[i]:
            coverers[j].append(i)

    best_code = None
    best_size = float('inf')
    
    num_runs = 200 # Increase runs as local search is faster with numpy

    def local_search(code_set, deadline):
        current_code = set(code_set)
        count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            count[sphere_arrays[c]] += 1
            
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            # 1. Prune: Remove any redundant codeword
            for c in list(current_code):
                if np.min(count[sphere_arrays[c]]) > 1:
                    current_code.remove(c)
                    count[sphere_arrays[c]] -= 1
                    changed = True
                    # Restart loop to check for new redundancies
                    break
            
            if not changed and time.time() < deadline:
                # 2. Swap: Try to replace one codeword with another to enable future prunes
                # Find a codeword whose removal creates uncovered words
                candidates_remove = list(current_code)
                random.shuffle(candidates_remove)
                
                for c_remove in candidates_remove:
                    # Simulate removal
                    temp_count = count.copy()
                    temp_count[sphere_arrays[c_remove]] -= 1
                    
                    # Find words that become uncovered (count == 0)
                    uncovered_mask = (temp_count == 0)
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_indices) == 0:
                        # This shouldn't happen if we pruned, but just in case
                        current_code.remove(c_remove)
                        count = temp_count
                        changed = True
                        break
                        
                    if len(uncovered_indices) > 3:
                        # Unlikely to find a single codeword covering >3 specific words
                        # unless they are close, but let's skip for speed
                        continue
                        
                    # Find a codeword c_add that covers all uncovered_indices
                    # and is not in current_code
                    # We check candidates from coverers of the first uncovered word
                    target = uncovered_indices[0]
                    cands = coverers[target]
                    random.shuffle(cands)
                    
                    found_swap = False
                    for c_add in cands:
                        if c_add in current_code:
                            continue
                        
                        # Check if c_add covers all uncovered_indices
                        add_sphere_set = set(spheres[c_add])
                        covers_all = all(j in add_sphere_set for j in uncovered_indices)
                        
                        if covers_all:
                            # Perform swap
                            current_code.remove(c_remove)
                            count[sphere_arrays[c_remove]] -= 1
                            current_code.add(c_add)
                            count[sphere_arrays[c_add]] += 1
                            changed = True
                            found_swap = True
                            break
                            
                    if found_swap:
                        break

            if not changed and time.time() < deadline:
                # 3. 1-for-2 Move: Add one, remove two
                # Try adding a random codeword that has high overlap with current code
                # but allows removing two redundant ones.
                
                # Heuristic: Pick a random codeword not in code
                # Or pick one that covers many words with count == 1
                # Let's try a few random additions
                
                attempts = 0
                while attempts < 5 and time.time() < deadline:
                    attempts += 1
                    c_add = random.randint(0, total_words - 1)
                    if c_add in current_code:
                        continue
                    
                    temp_count = count.copy()
                    temp_count[sphere_arrays[c_add]] += 1
                    temp_code = set(current_code)
                    temp_code.add(c_add)
                    
                    removed_count = 0
                    # Try to remove redundant codewords
                    for c_remove in list(temp_code):
                        if np.min(temp_count[sphere_arrays[c_remove]]) > 1:
                            temp_code.remove(c_remove)
                            temp_count[sphere_arrays[c_remove]] -= 1
                            removed_count += 1
                            if removed_count == 2:
                                break
                                
                    if removed_count == 2:
                        # Net change: +1 - 2 = -1
                        current_code = temp_code
                        count = temp_count
                        changed = True
                        break
                    # Else revert (do nothing)

        return current_code

    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        # Greedy
        while uncovered:
            best_gain = -1
            best_c = -1
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                gain = bin(sphere_bits[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
                    
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_bits[best_c]
            
        # Local Search
        code_set = local_search(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break

    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = [all_words[i] for i in best_code]
    return result
