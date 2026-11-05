<<<<<<< SEARCH
import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1)
    ball_size = 1 + n * (q - 1)
    neighbors = np.zeros((N, ball_size), dtype=np.int32)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        w_list = list(w)
        pos = 1
        for d in range(n):
            orig_val = w_list[d]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[d] = val
                neighbors[i, pos] = word_to_idx[tuple(w_list)]
                pos += 1
                w_list[d] = orig_val
    
    # Precompute neighbor sets for intersection operations
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(N)]
    
    # 3. Fast greedy using numpy vectorized gain
    def greedy_cover():
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            # Compute gain for all words at once using matrix multiply trick
            # gain[i] = number of uncovered points in neighbors[i]
            # We can use: for each i, sum of uncovered[neighbors[i]]
            # Efficient approach: for each uncovered point p, add 1 to all its neighbors' gains
            gains = np.zeros(N, dtype=np.int32)
            # For each uncovered point, increment gains of all words that cover it
            # This is the transpose operation
            uncovered_indices = np.where(uncovered)[0]
            for p in uncovered_indices:
                # Find all words that cover p: words i where p is in neighbors[i]
                # Equivalent: words i such that dist(words[i], words[p]) <= radius
                # We precompute reverse neighbors: rev[p] = list of i where p in neighbors[i]
                gains[rev_neighbors[p]] += 1
            
            # Pick the word with max gain that's not already in code
            best_idx = -1
            max_gain = 0
            for i in range(N):
                if i not in code_set and gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
            
            if max_gain == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        return code
    
    # Precompute reverse neighbors: rev_neighbors[p] = all i such that p is in neighbors[i]
    rev_neighbors = []
    for p in range(N):
        # Words i where dist(words[i], words[p]) <= radius
        # i.e., words[i] differs from words[p] in at most 'radius' positions
        wp = list(words[p])
        idxs = []
        for i in range(N):
            wi = words[i]
            dist = sum(1 for a, b in zip(wp, wi) if a != b)
            if dist <= radius:
                idxs.append(i)
        rev_neighbors.append(np.array(idxs, dtype=np.int32))
    
    # 4. Local search with deletion and swap
    def local_search(code):
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        def is_removable(c):
            for p in neighbors[c]:
                if counts[p] == 1:
                    return False
            return True
        
        def compress():
            changed = True
            while changed:
                changed = False
                for c in list(code):
                    if is_removable(c):
                        code.remove(c)
                        code_set.remove(c)
                        counts[neighbors[c]] -= 1
                        changed = True
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Deletion pass
            to_remove = [c for c in code if is_removable(c)]
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
            if to_remove:
                improved = True
            
            # Swap pass
            if time.time() > deadline - 5:
                break
            current_code_copy = list(code)
            random.shuffle(current_code_copy)
            
            for c in current_code_copy:
                if time.time() > deadline - 5:
                    break
                if c not in code_set:
                    continue
                
                # Get critical points
                crit_points = [p for p in neighbors[c] if counts[p] == 1]
                if not crit_points:
                    continue
                
                # Find candidates that cover all critical points
                candidate_set = set(neighbors[crit_points[0]].tolist())
                for p in crit_points[1:]:
                    candidate_set &= neighbor_sets[p]
                
                candidates = [x for x in candidate_set if x not in code_set]
                if not candidates:
                    continue
                
                best_candidate = -1
                best_score = -1
                neigh_c_set = neighbor_sets[c]
                
                for w in candidates:
                    score = 0
                    for p in neighbors[w]:
                        if counts[p] == 1 and p not in neigh_c_set:
                            score += 1
                    for p in neighbors[c]:
                        if counts[p] == 2 and p not in neighbor_sets[w]:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                
                if best_candidate != -1 and best_score > 0:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    code.append(best_candidate)
                    code_set.add(best_candidate)
                    counts[neighbors[best_candidate]] += 1
                    improved = True
                    # Immediate compress
                    to_remove2 = [c2 for c2 in code if is_removable(c2)]
                    for c2 in to_remove2:
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors[c2]] -= 1
        
        compress()
        return code
    
    # 5. Iterated Local Search with kicks
    best_code = None
    best_len = float('inf')
    
    # Initial greedy
    init_code = greedy_cover()
    if init_code:
        init_code = local_search(init_code)
        best_code = init_code[:]
        best_len = len(init_code)
    
    # ILS restarts
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Start from best solution and kick
        current_code = best_code[:]
        
        # Kick: remove k random codewords and re-greedy fill
        k = max(1, len(current_code) // 10)
        if k >= len(current_code):
            k = len(current_code) // 2
        
        to_remove = random.sample(current_code, k)
        for c in to_remove:
            current_code.remove(c)
        
        # Re-greedy to fill gaps
        code_set = set(current_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in current_code:
            counts[neighbors[idx]] += 1
        
        # Find uncovered points
        uncovered_mask = counts == 0
        while np.any(uncovered_mask):
            if time.time() > deadline - 10:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            gains = np.zeros(N, dtype=np.int32)
            for p in uncovered_indices:
                gains[rev_neighbors[p]] += 1
            
            best_idx = -1
            max_gain = 0
            for i in range(N):
                if i not in code_set and gains[i] > max_gain:
                    max_gain = gains[i]
                    best_idx = i
            
            if max_gain == 0:
                break
            
            current_code.append(best_idx)
            code_set.add(best_idx)
            counts[neighbors[best_idx]] += 1
            uncovered_mask = counts == 0
        
        # Local search
        current_code = local_search(current_code)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
    
    return [list(words[i]) for i in best_code]
=======
import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all words and index them
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1) as a numpy array
    ball_size = 1 + n * (q - 1)
    neighbors = np.zeros((N, ball_size), dtype=np.int32)
    for i, w in enumerate(words):
        neighbors[i, 0] = i
        w_list = list(w)
        pos = 1
        for d in range(n):
            orig_val = w_list[d]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[d] = val
                neighbors[i, pos] = word_to_idx[tuple(w_list)]
                pos += 1
                w_list[d] = orig_val
    
    # Precompute neighbor sets for intersection operations
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(N)]
    
    # 3. Precompute reverse neighbors using vectorized distance computation
    words_arr = np.array(words, dtype=np.int32)
    
    # Compute pairwise Hamming distances in chunks to save memory
    rev_neighbors = []
    chunk_size = 256
    for p_start in range(0, N, chunk_size):
        p_end = min(p_start + chunk_size, N)
        chunk_words = words_arr[p_start:p_end]
        sub_chunk = 128
        for sc_start in range(0, p_end - p_start, sub_chunk):
            sc_end = min(sc_start + sub_chunk, p_end - p_start)
            sub_words = chunk_words[sc_start:sc_end]
            diff = np.abs(words_arr[:, np.newaxis, :] - sub_words[np.newaxis, :, :])
            dist = np.sum(diff != 0, axis=2)
            for j in range(sc_end - sc_start):
                p = p_start + sc_start + j
                idxs = np.where(dist[:, j] <= radius)[0]
                rev_neighbors.append(idxs)
    
    # 4. Fast greedy using vectorized gain computation
    def greedy_cover(existing_code=None):
        if existing_code is None:
            existing_code = []
        
        uncovered = np.ones(N, dtype=bool)
        code = list(existing_code)
        code_set = set(code)
        
        # Initialize counts from existing code
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        uncovered = counts == 0
        
        while np.any(uncovered):
            if time.time() > deadline - 10:
                break
            
            # Vectorized gain: gains[i] = sum(uncovered[neighbors[i]])
            gains = neighbors @ uncovered.astype(np.int32)
            
            # Exclude words already in code
            for c in code_set:
                gains[c] = 0
            
            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]
            
            if max_gain == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered[neighbors[best_idx]] = False
        
        return code
    
    # 5. Local search with deletion and swap
    def local_search(code):
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        def is_removable(c):
            for p in neighbors[c]:
                if counts[p] == 1:
                    return False
            return True
        
        def compress():
            changed = True
            while changed:
                changed = False
                for c in list(code):
                    if is_removable(c):
                        code.remove(c)
                        code_set.remove(c)
                        counts[neighbors[c]] -= 1
                        changed = True
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Deletion pass
            to_remove = [c for c in code if is_removable(c)]
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
            if to_remove:
                improved = True
            
            # Swap pass
            if time.time() > deadline - 5:
                break
            current_code_copy = list(code)
            random.shuffle(current_code_copy)
            
            for c in current_code_copy:
                if time.time() > deadline - 5:
                    break
                if c not in code_set:
                    continue
                
                crit_points = [p for p in neighbors[c] if counts[p] == 1]
                if not crit_points:
                    continue
                
                candidate_set = set(neighbors[crit_points[0]].tolist())
                for p in crit_points[1:]:
                    candidate_set &= neighbor_sets[p]
                
                candidates = [x for x in candidate_set if x not in code_set]
                if not candidates:
                    continue
                
                best_candidate = -1
                best_score = -1
                neigh_c_set = neighbor_sets[c]
                
                for w in candidates:
                    score = 0
                    for p in neighbors[w]:
                        if counts[p] == 1 and p not in neigh_c_set:
                            score += 1
                    for p in neighbors[c]:
                        if counts[p] == 2 and p not in neighbor_sets[w]:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                
                if best_candidate != -1 and best_score > 0:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    code.append(best_candidate)
                    code_set.add(best_candidate)
                    counts[neighbors[best_candidate]] += 1
                    improved = True
                    to_remove2 = [c2 for c2 in code if is_removable(c2)]
                    for c2 in to_remove2:
                        code.remove(c2)
                        code_set.remove(c2)
                        counts[neighbors[c2]] -= 1
        
        compress()
        return code
    
    # 6. Final polish: aggressive deletion with multiple random orderings
    def final_polish(code):
        code_set = set(code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        def is_removable(c):
            for p in neighbors[c]:
                if counts[p] == 1:
                    return False
            return True
        
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            order = list(code)
            random.shuffle(order)
            for c in order:
                if c not in code_set:
                    continue
                if is_removable(c):
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
        
        return code
    
    # 7. Iterated Local Search with kicks
    best_code = None
    best_len = float('inf')
    
    # Initial greedy
    init_code = greedy_cover()
    if init_code:
        init_code = local_search(init_code)
        init_code = final_polish(init_code)
        best_code = init_code[:]
        best_len = len(init_code)
    
    # ILS restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        current_code = best_code[:]
        
        k = max(1, len(current_code) // 10)
        if k >= len(current_code):
            k = len(current_code) // 2
        
        to_remove = random.sample(current_code, k)
        for c in to_remove:
            current_code.remove(c)
        
        # Re-greedy to fill gaps using vectorized greedy
        current_code = greedy_cover(current_code)
        
        # Local search
        current_code = local_search(current_code)
        current_code = final_polish(current_code)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
    
    # Final polish on best
    if best_code:
        best_code = final_polish(best_code)
    
    return [list(words[i]) for i in best_code]
>>>>>>> REPLACE
