"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    total_words = q ** n
    # Precompute all words as a 2D numpy array
    all_words = np.array(list(range(total_words)), dtype=np.int32)
    
    # Convert index to word (base-q digits)
    def idx_to_word(idx):
        word = np.zeros(n, dtype=np.int32)
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx
    
    # Precompute neighbors for radius=1: for each word, the set of words within distance <= 1
    # For radius=1: word + all words differing in exactly one position
    # Number of neighbors per word: 1 + n*(q-1)
    num_neighbors = 1 + n * (q - 1)
    
    # Precompute neighbor indices for all words
    # neighbors[i] = array of word indices within distance <= radius from word i
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        idx_count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    neighbors[i, idx_count] = word_to_idx(w)
                    idx_count += 1
            w[pos] = orig_val
    
    # Greedy set cover with numpy-based coverage tracking
    def greedy_set_cover(max_time=None):
        """Greedy set cover: repeatedly pick the center covering most uncovered words."""
        if max_time is None:
            max_time = deadline - 5
        
        uncovered = np.ones(total_words, dtype=np.int32)  # 1 if uncovered
        selected = []
        
        # For each center, count how many uncovered words it covers
        # cover_count[c] = number of uncovered words in neighbors[c]
        cover_count = np.array([np.sum(neighbors[c] >= 0) for c in range(total_words)], dtype=np.int32)
        # Actually, all words are uncovered initially, so cover_count[c] = num_neighbors for all c
        # But we need to update as we select centers
        cover_count = np.full(total_words, num_neighbors, dtype=np.int32)
        
        while np.any(uncovered) and time.time() < max_time:
            # Find center with max cover_count
            best_center = np.argmax(cover_count)
            if cover_count[best_center] <= 0:
                break
            
            selected.append(best_center)
            cover_count[best_center] = 0  # Mark as selected
            
            # Update: for each word covered by best_center, decrement cover_count of all centers covering it
            covered_words = neighbors[best_center]
            for w_idx in covered_words:
                if uncovered[w_idx]:
                    uncovered[w_idx] = 0
                    # Decrement cover_count for all centers that cover w_idx
                    # A center c covers w_idx if w_idx is in neighbors[c]
                    # This is expensive to do naively. Instead, precompute reverse mapping.
                    pass
        
        return selected
    
    # More efficient approach: precompute reverse mapping
    # For each word w, which centers cover it?
    # centers_covering[w] = list of center indices c such that w is in neighbors[c]
    # For radius=1, this is the same as neighbors[w] (symmetric relation)
    
    # Actually for Hamming distance, the relation is symmetric:
    # c covers w iff w covers c iff dist(c,w) <= radius
    # So centers_covering[w] = neighbors[w]
    
    # Efficient greedy:
    def greedy_set_cover_efficient(max_time=None):
        if max_time is None:
            max_time = deadline - 5
        
        uncovered = np.ones(total_words, dtype=np.bool_)
        selected = []
        
        # cover_count[c] = number of uncovered words that center c covers
        cover_count = np.full(total_words, num_neighbors, dtype=np.int32)
        
        while uncovered.any() and time.time() < max_time:
            best_center = int(np.argmax(cover_count))
            if cover_count[best_center] <= 0:
                break
            
            selected.append(best_center)
            cover_count[best_center] = 0
            
            # For each word w covered by best_center that is still uncovered:
            #   mark w as covered
            #   for each center c that covers w (i.e., c in neighbors[w]):
            #       decrement cover_count[c]
            for w_idx in neighbors[best_center]:
                if uncovered[w_idx]:
                    uncovered[w_idx] = False
                    for c in neighbors[w_idx]:
                        if cover_count[c] > 0:
                            cover_count[c] -= 1
        
        return selected
    
    # Local search: try to remove redundant codewords
    def local_search_reduce(selected, max_time=None):
        """Try to remove codewords that don't uniquely cover any word."""
        if max_time is None:
            max_time = deadline - 5
        
        selected_set = set(selected)
        
        # For each word, count how many selected centers cover it
        cover_by_word = np.zeros(total_words, dtype=np.int32)
        for c in selected:
            for w in neighbors[c]:
                cover_by_word[w] += 1
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(selected)):
                c = selected[i]
                # Check if all words covered by c are covered by at least one other selected center
                can_remove = True
                for w in neighbors[c]:
                    if cover_by_word[w] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    for w in neighbors[c]:
                        cover_by_word[w] -= 1
                    selected.pop(i)
                    improved = True
                    break
        
        return selected
    
    # Try multiple random restarts with different initial orderings
    def randomized_greedy(seed_offset, max_time=None):
        if max_time is None:
            max_time = deadline - 5
        
        rng = random.Random(seed_offset)
        uncovered = np.ones(total_words, dtype=np.bool_)
        selected = []
        
        # cover_count[c] = number of uncovered words that center c covers
        cover_count = np.full(total_words, num_neighbors, dtype=np.int32)
        
        # Add some randomness to tie-breaking
        while uncovered.any() and time.time() < max_time:
            # Find all centers with max cover_count
            max_count = cover_count.max()
            if max_count <= 0:
                break
            
            # Get all centers with this max count
            candidates = np.where(cover_count == max_count)[0]
            if len(candidates) > 1:
                best_center = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                best_center = candidates[0]
            
            selected.append(int(best_center))
            cover_count[best_center] = 0
            
            for w_idx in neighbors[best_center]:
                if uncovered[w_idx]:
                    uncovered[w_idx] = False
                    for c in neighbors[w_idx]:
                        if cover_count[c] > 0:
                            cover_count[c] -= 1
        
        return selected
    
    # Main algorithm:
    # 1. Run deterministic greedy
    # 2. Run several randomized greedy variants
    # 3. Apply local search to each
    # 4. Keep the best (smallest) result
    
    best_selected = None
    best_len = float('inf')
    
    # Deterministic greedy
    sel = greedy_set_cover_efficient(deadline - 20)
    sel = local_search_reduce(sel, deadline - 10)
    if len(sel) < best_len:
        best_len = len(sel)
        best_selected = sel[:]
    
    # Randomized greedy with different seeds
    remaining_time = deadline - time.time()
    num_restarts = min(5, max(1, int(remaining_time / 15)))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed_offset = restart * 1000 + 42
        sel = randomized_greedy(seed_offset, deadline - 10)
        sel = local_search_reduce(sel, deadline - 5)
        if len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel[:]
    
    # Final polish: try harder local search on the best solution
    if best_selected is not None and time.time() < deadline - 5:
        best_selected = local_search_reduce(best_selected, deadline - 2)
    
    # Convert selected indices to words
    result = [idx_to_word(c) for c in best_selected]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for word in result:
        key = tuple(word)
        if key not in seen:
            seen.add(key)
            unique_result.append(word.tolist())
    
    return unique_result
# EVOLVE-BLOCK-END
