"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # Map word to index for quick lookup
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets: covers[i] is the set of indices covered by codeword i
    # A word covers itself and all words at Hamming distance 1
    covers = [set() for _ in range(total_words)]
    
    for i, w in enumerate(all_words):
        covers[i].add(i)
        w_list = list(w)
        for pos in range(n):
            original_val = w_list[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                w_list[pos] = new_val
                neighbor = tuple(w_list)
                covers[i].add(word_to_idx[neighbor])
            w_list[pos] = original_val
            
    # Greedy Set Cover using Sets
    uncovered = set(range(total_words))
    selected = []
    
    # Optimization: sort candidates by potential? No, just pick max intersection each time.
    # To speed up, we can maintain a list of candidates.
    
    while uncovered:
        if time.time() > deadline - 20.0:
            break
            
        best_idx = -1
        best_count = 0
        
        # Iterate through all possible codewords to find the one covering most uncovered words
        # We can skip already selected ones? No, they are in selected, but we can just check all.
        # Actually, we can just check all. 2187 is small.
        for i in range(total_words):
            # Count how many words in covers[i] are in uncovered
            # Using set intersection
            common = len(covers[i] & uncovered)
            if common > best_count:
                best_count = common
                best_idx = i
                
        if best_idx == -1 or best_count == 0:
            break
            
        selected.append(best_idx)
        uncovered -= covers[best_idx]
        
    # Post-processing: Remove redundant codewords
    # Maintain coverage counts
    cov_counts = [0] * total_words
    for idx in selected:
        for w in covers[idx]:
            cov_counts[w] += 1
            
    # Pass 1: Remove any codeword that is not critical for any word
    # We do this iteratively until no more can be removed
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        new_selected = []
        for i in range(len(selected)):
            idx = selected[i]
            is_redundant = True
            for w in covers[idx]:
                if cov_counts[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove it
                for w in covers[idx]:
                    cov_counts[w] -= 1
                changed = True
            else:
                new_selected.append(idx)
        selected = new_selected
        
    # Convert indices back to words
    return [list(all_words[idx]) for idx in selected]
# EVOLVE-BLOCK-END
