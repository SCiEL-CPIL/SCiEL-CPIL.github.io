import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]
    
    # Precompute neighbors as numpy array (13 neighbors: self + 12 at dist 1)
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1
    
    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]
    
    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Initial Solution: Greedy Set Cover
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 60:
            break
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0
        
    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            codewords.append(c)
            uncovered[neighbors[c]] = 0
            
    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
        
    # Track which codeword indices are in the set
    in_set = set(codewords)
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    random.seed(42)
    np.random.seed(42)
    
    # SA loop with genetic-style perturbations
    T = 1.0
    T_min = 0.005
    cooling = 0.9997
    
    steps = 0
    last_improve_step = 0
    restart_count = 0
    
    while time.time() < deadline - 3:
        steps += 1
        
        # Periodically restart with new random seed if stuck
        if steps - last_improve_step > 50000 and restart_count < 10:
            restart_count += 1
            T = 1.0
            # Perturb current solution
            for _ in range(random.randint(3, 7)):
                if codewords:
                    idx_to_remove = random.randint(0, len(codewords) - 1)
                    c_to_remove = codewords[idx_to_remove]
                    # Find a replacement
                    new_c = random.randint(0, total - 1)
                    codewords[idx_to_remove] = new_c
                    coverage_counts[neighbors[c_to_remove]] -= 1
                    coverage_counts[neighbors[new_c]] += 1
                    in_set.discard(c_to_remove)
                    in_set.add(new_c)
            last_improve_step = steps
            
        # Find redundant words efficiently
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)
                
        if redundant_list:
            # Remove a redundant word - prefer one with highest "slack"
            if len(redundant_list) > 1:
                slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                max_slack = max(slacks)
                best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                idx_to_remove = random.choice(best_candidates)
            else:
                idx_to_remove = redundant_list[0]
            
            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)
            
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                last_improve_step = steps
            
            T = max(T * cooling, T_min)
            continue
            
        else:
            # No redundant words. Try swaps.
            found_swap = False
            num_tries = 5
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]
                
                # Find critical words: neighbors of c_to_remove with coverage exactly 1
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]
                
                if len(critical_words) == 0:
                    continue
                
                # Candidates must be in intersection of neighbors of all critical words
                w0 = int(critical_words[0])
                candidates = neighbors[w0]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))
                
                if not valid_candidates:
                    continue
                
                # Score candidates by how much they improve coverage balance
                # Prefer candidates that cover words with low coverage
                best_cand = None
                best_score = -1
                for cand in valid_candidates:
                    # Score: sum of coverage_counts for neighbors of cand
                    # Lower sum means we're adding coverage to less-covered areas
                    score = -coverage_counts[neighbors[cand]].sum()
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                c_to_add = best_cand if best_cand is not None else random.choice(valid_candidates)
                
                # Perform swap
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)
                
                found_swap = True
                break
            
            T = max(T * cooling, T_min)

    # Final cleanup: iteratively remove redundant words
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    # Use the best found
    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    return [idx_to_list(c) for c in codewords]
