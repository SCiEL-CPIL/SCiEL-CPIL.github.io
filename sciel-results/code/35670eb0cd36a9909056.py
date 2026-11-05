import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate all words
    all_words = []
    word_to_idx = {}
    idx = 0
    for digits in np.ndindex(*([3]*n)):
        w = list(digits)
        word_to_idx[tuple(w)] = idx
        all_words.append(w)
        idx += 1
    N = len(all_words)
    
    # Precompute neighbors
    neighbors = [[] for _ in range(N)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                neighbors[i].append(j)
    
    # Convert to numpy arrays and also build a membership matrix for fast lookup
    neighbor_arrays = [np.array(nbs) for nbs in neighbors]
    
    # For pair replacement, we need to check if a word is in a neighbor set
    # Build a boolean matrix: is_neighbor[i, j] = 1 if j is in neighbors[i]
    # This is 2187x2187 = ~4.8M booleans, which is ~4.8MB, acceptable
    is_neighbor = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in neighbors[i]:
            is_neighbor[i, j] = True
    
    # Greedy construction
    uncovered = np.ones(N, dtype=bool)
    code_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 80:
            break
        best_candidate = -1
        max_covered = -1
        for c in range(N):
            if c in code_set:
                continue
            count = np.sum(uncovered[neighbor_arrays[c]])
            if count > max_covered:
                max_covered = count
                best_candidate = c
        if best_candidate == -1:
            break
        code_set.add(best_candidate)
        uncovered[neighbor_arrays[best_candidate]] = False
    
    # Local search
    cover_count = np.zeros(N, dtype=int)
    for c in code_set:
        cover_count[neighbor_arrays[c]] += 1
    
    best_size = len(code_set)
    best_code_set = set(code_set)
    
    def try_remove():
        for c in list(code_set):
            nbs = neighbor_arrays[c]
            if np.all(cover_count[nbs] > 1):
                code_set.discard(c)
                cover_count[nbs] -= 1
                return True
        return False
    
    def try_swap():
        if not code_set:
            return False
        c_out = random.choice(list(code_set))
        nbs_out = neighbor_arrays[c_out]
        cover_count[nbs_out] -= 1
        code_set.discard(c_out)
        
        best_c_in = -1
        max_cover = -1
        for c_in in range(N):
            if c_in in code_set:
                continue
            nbs_in = neighbor_arrays[c_in]
            cnt = np.sum(cover_count[nbs_in] == 0)
            if cnt > max_cover:
                max_cover = cnt
                best_c_in = c_in
        
        if best_c_in != -1 and max_cover > 0:
            nbs_in = neighbor_arrays[best_c_in]
            cover_count[nbs_in] += 1
            code_set.add(best_c_in)
            return True
        else:
            code_set.add(c_out)
            cover_count[nbs_out] += 1
            return False
    
    def try_pair_replace():
        """Remove 2 codewords, add 1 that covers all uniquely-covered words."""
        if len(code_set) < 3:
            return False
        
        code_list = list(code_set)
        random.shuffle(code_list)
        attempts = min(15, len(code_list) * (len(code_list) - 1) // 2)
        
        for _ in range(attempts):
            i = random.randrange(len(code_list))
            j = random.randrange(len(code_list) - 1)
            if j >= i:
                j += 1
            c1, c2 = code_list[i], code_list[j]
            if c1 == c2:
                continue
            
            nbs1 = neighbor_arrays[c1]
            nbs2 = neighbor_arrays[c2]
            
            # Find words uniquely covered by c1 or c2
            unique_words = np.zeros(N, dtype=bool)
            unique_words[nbs1] = (cover_count[nbs1] == 1)
            unique_words[nbs2] = (cover_count[nbs2] == 1)
            
            unique_count = np.sum(unique_words)
            if unique_count == 0:
                continue
            
            # Find a codeword that covers all unique words
            best_c_in = -1
            for c_in in range(N):
                if c_in in code_set:
                    continue
                nbs_in = neighbor_arrays[c_in]
                if np.all(unique_words[nbs_in]):
                    best_c_in = c_in
                    break
            
            if best_c_in != -1:
                code_set.discard(c1)
                code_set.discard(c2)
                cover_count[nbs1] -= 1
                cover_count[nbs2] -= 1
                nbs_in = neighbor_arrays[best_c_in]
                cover_count[nbs_in] += 1
                code_set.add(best_c_in)
                return True
        
        return False
    
    # Main local search loop
    no_improvement = 0
    max_no_improve = 30
    
    while time.time() < deadline - 15:
        improved = False
        
        # Removals
        while try_remove():
            improved = True
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code_set = set(code_set)
        
        # Pair replacements
        for _ in range(3):
            if try_pair_replace():
                improved = True
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code_set = set(code_set)
                break
        
        # Swaps
        if not improved:
            for _ in range(2):
                if try_swap():
                    improved = True
                    break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > max_no_improve:
                # Perturbation
                if code_set:
                    c_out = random.choice(list(code_set))
                    code_set.discard(c_out)
                    cover_count[neighbor_arrays[c_out]] -= 1
                    no_improvement = 0
    
    # Final cleanup
    while try_remove() and time.time() < deadline - 2:
        best_size = len(code_set)
        best_code_set = set(code_set)
    
    return [all_words[c] for c in best_code_set]


def _generic_construct(q, n, radius):
    """Simple fallback."""
    all_words = []
    for digits in np.ndindex(*([q]*n)):
        all_words.append(list(digits))
    
    N = len(all_words)
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        best = next(iter(uncovered))
        code.append(best)
        uncovered.discard(best)
    
    return [all_words[c] for c in code]
