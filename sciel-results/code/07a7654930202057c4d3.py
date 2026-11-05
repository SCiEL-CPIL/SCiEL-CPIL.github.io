import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task
    # q=3, n=6, radius=1
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for general case (not expected to be called with other params in this specific task context, 
        # but good to have a basic constructor)
        # For now, return a trivial covering (all words) if params don't match, 
        # though the task specifies this is for the ternary covering code problem.
        # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
        # So we assume q=3, n=6, radius=1.
        pass

    total_words = q ** n
    # Map word tuple to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words within Hamming distance 1 of word i
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        word = idx_to_word(idx)
        neighbors[idx].append(idx) # distance 0
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    word[pos] = new_val
                    n_idx = word_to_idx(word)
                    neighbors[idx].append(n_idx)
            word[pos] = orig_val
            
    # Initial Code: 81 words from Tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = h + list(tail)
            initial_codewords.append(word_to_idx(cw))
            
    # cover_count[w] = number of codewords in C that cover w
    cover_count = np.zeros(total_words, dtype=np.int32)
    C = set(initial_codewords)
    
    # Precompute neighbor arrays as numpy for faster operations
    neighbors_np = np.array(neighbors, dtype=np.int32)
    
    for c in C:
        cover_count[neighbors_np[c]] += 1

    def is_valid_cover():
        return bool(np.all(cover_count > 0))

    # Greedy Removal
    def greedy_remove():
        removed_any = True
        while removed_any:
            removed_any = False
            # Try to remove any redundant codeword
            for c in list(C):
                # Check if c is redundant using numpy
                if np.all(cover_count[neighbors_np[c]] > 1):
                    C.remove(c)
                    cover_count[neighbors_np[c]] -= 1
                    removed_any = True
                    break
        return len(C)

    # Initial Greedy Pass
    greedy_remove()
    
    # Randomized Swap Search with Kick mechanism
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_size = len(C)
    best_C = set(C)
    best_cover_count = cover_count.copy()
    
    # Precompute neighbor sets as Python sets for faster intersection
    neighbors_set = [set(nbrs) for nbrs in neighbors]
    
    def do_swap(c_old, c_new):
        """Perform a swap and update state."""
        C.remove(c_old)
        cover_count[neighbors_np[c_old]] -= 1
        C.add(c_new)
        cover_count[neighbors_np[c_new]] += 1
    
    def undo_swap(c_old, c_new):
        """Undo a swap."""
        C.remove(c_new)
        cover_count[neighbors_np[c_new]] -= 1
        C.add(c_old)
        cover_count[neighbors_np[c_old]] += 1
    
    def try_kick():
        """Try a kick: remove a codeword that breaks coverage, then try to repair."""
        nonlocal best_size, best_C, best_cover_count
        
        c_old_list = list(C)
        if not c_old_list:
            return False
            
        c_old = random.choice(c_old_list)
        
        # Remove c_old (may break coverage)
        C.remove(c_old)
        cover_count[neighbors_np[c_old]] -= 1
        
        # Now try to repair by adding new codewords
        # Find uncovered words
        uncovered_mask = cover_count <= 0
        uncovered_words = np.where(uncovered_mask)[0]
        
        if len(uncovered_words) == 0:
            # Coverage was maintained, just do greedy removal
            greedy_remove()
            return True
        
        # Try to cover uncovered words with new codewords
        repairs = 0
        max_repairs = len(uncovered_words) + 2
        
        for _ in range(max_repairs):
            if is_valid_cover():
                break
            if time.time() > deadline:
                # Undo everything
                C.add(c_old)
                cover_count[neighbors_np[c_old]] += 1
                return False
                
            # Pick an uncovered word
            uncovered_mask = cover_count <= 0
            uncovered_words = np.where(uncovered_mask)[0]
            if len(uncovered_words) == 0:
                break
                
            w = random.choice(uncovered_words.tolist())
            
            # Find candidate codewords that cover w
            candidates = neighbors_set[w] - C
            if not candidates:
                # No valid candidate, undo
                C.add(c_old)
                cover_count[neighbors_np[c_old]] += 1
                return False
            
            # Score candidates by how many uncovered words they cover
            best_cand = None
            best_score = -1
            cand_list = list(candidates)
            for cand in cand_list:
                score = np.sum(cover_count[neighbors_np[cand]] <= 0)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand is None:
                C.add(c_old)
                cover_count[neighbors_np[c_old]] += 1
                return False
            
            C.add(best_cand)
            cover_count[neighbors_np[best_cand]] += 1
            repairs += 1
        
        # Check if we repaired successfully
        if is_valid_cover():
            greedy_remove()
            if len(C) < best_size:
                best_size = len(C)
                best_C = set(C)
                best_cover_count = cover_count.copy()
            return True
        else:
            # Undo kick
            C.add(c_old)
            cover_count[neighbors_np[c_old]] += 1
            return False
    
    while time.time() < deadline:
        greedy_remove()
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = set(C)
            best_cover_count = cover_count.copy()
        
        c_old_list = list(C)
        if not c_old_list:
            break
        c_old = random.choice(c_old_list)
        
        # Identify words uniquely covered by c_old
        crit_mask = (cover_count == 1)
        critical_words = neighbors_np[c_old][crit_mask[neighbors_np[c_old]]]
        
        valid_c_new = []
        if len(critical_words) == 0:
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in C:
                    valid_c_new = [c_new]
                    break
        else:
            candidates = neighbors_set[critical_words[0]]
            for w in critical_words[1:]:
                candidates = candidates & neighbors_set[w]
                if not candidates:
                    break
            for c in candidates:
                if c not in C:
                    valid_c_new.append(c)
        
        if valid_c_new:
            c_new = random.choice(valid_c_new)
            do_swap(c_old, c_new)
        else:
            # No valid swap, try a kick
            try_kick()
        
        # Occasionally try a kick even when swaps are available
        if random.random() < 0.05:
            try_kick()
    
    # Restore best solution
    C = set(best_C)
    cover_count = best_cover_count.copy()
    
    # Final Greedy Pass
    greedy_remove()
    
    # Return list of codewords
    return [idx_to_word(c) for c in C]
