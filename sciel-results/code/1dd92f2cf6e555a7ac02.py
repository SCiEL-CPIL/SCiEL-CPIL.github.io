import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
        
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate all words and neighbor structure
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Map from word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Build neighbors list for each word
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = [None] * total
    for i in range(total):
        w = all_words[i]
        nbrs = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    nbrs.append(word_to_idx[tuple(w_new)])
        neighbors[i] = nbrs
    
    # Convert neighbors to numpy arrays for faster indexing
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    
    # Precompute adjacency matrix M (6561x6561) for vectorized greedy
    # M[i, j] = 1 if j is a neighbor of i
    M = np.zeros((total, total), dtype=np.float32)
    for i in range(total):
        M[i, neighbors[i]] = 1.0

    # 2. Greedy Construction
    uncovered = np.ones(total, dtype=np.float32)
    code_indices = []
    
    while np.any(uncovered) and time.time() < deadline - 60.0:
        # scores[i] = number of uncovered words covered by word i
        scores = M @ uncovered
        best_idx = int(np.argmax(scores))
        
        if scores[best_idx] == 0:
            break
            
        code_indices.append(best_idx)
        # Mark neighbors of best_idx as covered
        uncovered[neighbors_arr[best_idx]] = 0.0

    # Ensure full coverage (if greedy didn't finish due to time, finish it)
    if np.any(uncovered):
        while np.any(uncovered):
            best_idx = -1
            best_cnt = -1
            for i in range(total):
                if i in code_indices:
                    continue
                cnt = np.sum(uncovered[neighbors_arr[i]])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_indices.append(best_idx)
            uncovered[neighbors_arr[best_idx]] = 0.0

    # 3. Local Search
    code_set = set(code_indices)
    code_list = list(code_indices)
    
    # cover_count[j] = number of codewords that cover word j
    cover_count = np.zeros(total, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_arr[c]] += 1

    best_size = len(code_list)
    best_code = list(code_list)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # --- Phase 1: Removal ---
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
            c = code_list[i]
            # Check if c is redundant: all its neighbors are covered by at least one other codeword
            if np.all(cover_count[neighbors_arr[c]] > 1):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_arr[c]] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)

        if improved:
            no_improvement_count = 0
            continue
        
        # --- Phase 2: Swap ---
        if code_list:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            nbrs_c = neighbors_arr[c]
            cover_count[nbrs_c] -= 1
            
            # Find uncovered words among neighbors of c
            uncovered_mask = cover_count == 0
            uncovered_words = nbrs_c[uncovered_mask[nbrs_c]]
            
            if len(uncovered_words) == 0:
                # c was redundant
                code_list.pop(idx_c)
                code_set.discard(c)
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
            else:
                # Find replacement: word that covers ALL uncovered_words
                # Use matrix multiplication approach: for each candidate, check if it covers all uncovered
                # M[:, uncovered_words] gives for each row i, how many of the uncovered words it covers
                # We need rows where this count == len(uncovered_words)
                coverage_counts = M[:, uncovered_words].sum(axis=1)
                valid_candidates = np.where(coverage_counts == len(uncovered_words))[0]
                
                # Exclude existing codewords
                valid_candidates = [cand for cand in valid_candidates if cand not in code_set]
                
                if valid_candidates:
                    # Heuristic: pick candidate that maximizes coverage of words with low cover_count
                    # (i.e., words that are barely covered)
                    # Score: sum of (1 / cover_count[j]) for j in neighbors of candidate, excluding uncovered
                    best_cand = None
                    best_score = -1
                    for cand in valid_candidates[:20]:  # Limit to top 20 for speed
                        nbrs_cand = neighbors_arr[cand]
                        # Only consider words that are currently covered (cover_count > 0)
                        covered_mask = cover_count[nbrs_cand] > 0
                        if covered_mask.any():
                            # Score based on how "critical" the coverage is
                            score = np.sum(1.0 / cover_count[nbrs_cand[covered_mask]])
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                    
                    if best_cand is not None:
                        code_list[idx_c] = best_cand
                        code_set.add(best_cand)
                        cover_count[neighbors_arr[best_cand]] += 1
                        improved = True
                    else:
                        # Revert
                        cover_count[nbrs_c] += 1
                        no_improvement_count += 1
                else:
                    # Revert
                    cover_count[nbrs_c] += 1
                    no_improvement_count += 1

        if not improved:
            no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            no_improvement_count = 0
            random.shuffle(code_list)
            if time.time() < deadline - 30.0 and random.random() < 0.15:
                # Aggressive restart
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, num_remove)
                for c in to_remove:
                    code_set.discard(c)
                    cover_count[neighbors_arr[c]] -= 1
                    code_list.remove(c)
                
                # Re-greedy to fix coverage
                current_uncovered = (cover_count == 0).astype(np.float32)
                while np.any(current_uncovered) and time.time() < deadline - 15.0:
                    scores = M @ current_uncovered
                    mask = np.ones(total, dtype=bool)
                    for c in code_set:
                        mask[c] = False
                    scores_masked = np.where(mask, scores, -1)
                    best_idx = int(np.argmax(scores_masked))
                    
                    if scores_masked[best_idx] <= 0:
                        break
                        
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    cover_count[neighbors_arr[best_idx]] += 1
                    current_uncovered[neighbors_arr[best_idx]] = 0.0

    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
