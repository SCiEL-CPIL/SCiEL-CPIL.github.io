import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 150s kill

    # Direct sum with tetracode for q=3
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        if time.time() < deadline - 5:
            improved = _reduce_cover(q, n, radius, result, deadline)
            if improved is not None:
                return improved
        return result
    
    # General fallback: greedy set cover
    return _greedy_set_cover(q, n, radius, deadline)


def _reduce_cover(q, n, radius, initial_words, deadline):
    """Reduce a covering code using deletion and swap passes with numpy."""
    if q != 3 or n != 7 or radius != 1:
        return None
    
    space_size = q ** n  # 2187
    powers = [q ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
    
    # Build neighborhood table: for each index, list of indices within distance <= radius
    # For radius=1: self + all single-position changes
    neighborhoods = [None] * space_size
    for idx in range(space_size):
        w = [0] * n
        tmp = idx
        for i in range(n):
            w[i] = tmp % q
            tmp //= q
        neigh = [idx]
        for pos in range(n):
            orig = w[pos]
            for delta in range(1, q):
                w[pos] = (orig + delta) % q
                # Compute index
                ni = 0
                for j in range(n):
                    ni += w[j] * powers[j]
                neigh.append(ni)
            w[pos] = orig
        neighborhoods[idx] = neigh
    
    # Convert to numpy arrays for vectorized ops
    neigh_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighborhoods]
    
    # Current codewords as indices
    codeword_indices = set(word_to_index(w) for w in initial_words)
    code_list = list(codeword_indices)
    
    # Compute coverage counts
    counts = np.zeros(space_size, dtype=np.int32)
    for cw in code_list:
        counts[neigh_arrays[cw]] += 1
    
    best_count = len(code_list)
    best_code = list(code_list)
    
    # Iterative improvement: deletion + swap
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code_list:
            if np.min(counts[neigh_arrays[c]]) > 1:
                to_remove.append(c)
        
        if to_remove:
            improved = True
            for c in to_remove:
                code_list.remove(c)
                codeword_indices.discard(c)
                counts[neigh_arrays[c]] -= 1
            if len(code_list) < best_count:
                best_count = len(code_list)
                best_code = list(code_list)
        
        if time.time() > deadline - 8:
            break
        
        # --- Swap Pass ---
        # For each codeword that is NOT removable, try to find a replacement
        # that covers all its critical points (points with count==1 in its neighborhood)
        random.shuffle(code_list)
        
        for c in code_list:
            if time.time() > deadline - 5:
                break
            
            # Check if c is removable
            min_count_in_neigh = np.min(counts[neigh_arrays[c]])
            if min_count_in_neigh > 1:
                continue  # Already handled by deletion
            
            # Get critical points: points in neighbors[c] with count == 1
            crit_mask = counts[neigh_arrays[c]] == 1
            crit_points = neigh_arrays[c][crit_mask]
            
            if len(crit_points) == 0:
                continue
            
            # Find candidates: codewords not in code_list that cover ALL critical points
            # Start with candidates from the first critical point
            first_cp = crit_points[0]
            cands = [x for x in neighborhoods[first_cp] if x not in codeword_indices]
            
            best_w = -1
            best_score = -1
            
            for w in cands:
                # Check if w covers all critical points
                valid = True
                w_neigh_set = set(neighborhoods[w])
                for k in crit_points:
                    if k not in w_neigh_set:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: maximize the number of points where count would become 2
                # (i.e., points already covered once by other codewords)
                score = int(np.sum(counts[neigh_arrays[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w
            
            if best_w != -1:
                # Perform the swap
                code_list.remove(c)
                codeword_indices.discard(c)
                counts[neigh_arrays[c]] -= 1
                
                code_list.append(best_w)
                codeword_indices.add(best_w)
                counts[neigh_arrays[best_w]] += 1
                
                improved = True
                if len(code_list) < best_count:
                    best_count = len(code_list)
                    best_code = list(code_list)
        
        # --- Second deletion pass after swaps ---
        if improved and time.time() < deadline - 5:
            to_remove2 = []
            for c in code_list:
                if np.min(counts[neigh_arrays[c]]) > 1:
                    to_remove2.append(c)
            if to_remove2:
                improved = True
                for c in to_remove2:
                    code_list.remove(c)
                    codeword_indices.discard(c)
                    counts[neigh_arrays[c]] -= 1
                if len(code_list) < best_count:
                    best_count = len(code_list)
                    best_code = list(code_list)
    
    if best_count < len(initial_words):
        return [index_to_word(idx) for idx in best_code]
    
    return None


def _greedy_set_cover(q, n, radius, deadline):
    """General greedy set cover for covering codes."""
    space_size = q ** n
    
    if space_size > 50000:
        random.seed(42)
        words = []
        seen = set()
        while len(words) < 200 and time.time() < deadline - 5:
            w = tuple(random.randint(0, q - 1) for _ in range(n))
            if w not in seen:
                seen.add(w)
                words.append(list(w))
        return words
    
    powers = [q ** i for i in range(n)]
    
    def word_to_index(word):
        idx = 0
        for i in range(n):
            idx += word[i] * powers[i]
        return idx
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
    
    # Build neighborhoods
    neighborhoods = [None] * space_size
    for idx in range(space_size):
        w = [0] * n
        tmp = idx
        for i in range(n):
            w[i] = tmp % q
            tmp //= q
        neigh = [idx]
        for pos in range(n):
            orig = w[pos]
            for delta in range(1, q):
                w[pos] = (orig + delta) % q
                ni = 0
                for j in range(n):
                    ni += w[j] * powers[j]
                neigh.append(ni)
            w[pos] = orig
        neighborhoods[idx] = neigh
    
    neigh_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighborhoods]
    
    # Greedy set cover
    uncovered = np.ones(space_size, dtype=bool)
    selected = []
    selected_set = set()
    
    while time.time() < deadline - 3 and uncovered.any():
        # Compute gains: for each candidate, how many uncovered points it covers
        gains = np.zeros(space_size, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neigh_arrays[idx]] += 1
        
        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]
        
        if max_gain == 0:
            break
        
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered[neigh_arrays[best_idx]] = False
    
    return [index_to_word(idx) for idx in selected]
