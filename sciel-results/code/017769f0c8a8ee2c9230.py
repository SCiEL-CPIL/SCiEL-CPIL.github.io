import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    # Total number of words
    total_words = q ** n
    
    # Precompute neighbors for each word
    # Word w is an integer 0..total_words-1
    # Convert to base-q digits to find neighbors
    # neighbors[w] is a list of integers
    
    # Optimization: Precompute the "ball" for each word
    # Since n=6, q=3, radius=1, ball size is 1 + 6*2 = 13.
    # We can represent the set of neighbors as a frozenset or a boolean mask.
    # Given total_words=729, a boolean mask of length 729 is efficient.
    
    # Let's use a list of lists for simplicity first, then optimize if needed.
    # Actually, for speed in the main loop, we want to quickly compute intersection size.
    # Using Python sets is fast enough for 729 elements.
    
    # Generate all words as tuples of digits
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for w in all_words:
        neigh = [w]
        for i in range(n):
            for val in range(q):
                if val == w[i]:
                    continue
                new_w = list(w)
                new_w[i] = val
                neigh.append(tuple(new_w))
        # Convert to indices
        neighbors.append([word_to_idx[ng] for ng in neigh])
        
    # Convert to sets for fast intersection
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_candidate = -1
        max_gain = 0
        
        # Optimization: We can skip candidates that are already in code? 
        # No, a candidate in code might cover new uncovered words if we re-evaluate, 
        # but we just removed its coverage. Actually, once a word is in code, 
        # its neighbors are removed from uncovered. So it can't cover anything new 
        # that wasn't already covered by it. So we can skip words already in code.
        
        candidates = [i for i in range(total_words) if i not in code]
        
        # To speed up, we can sample or just iterate. 729 is small.
        for c in candidates:
            # Count how many uncovered words are in the ball of c
            gain = len(neighbor_sets[c] & uncovered)
            if gain > max_gain:
                max_gain = gain
                best_candidate = c
            elif gain == max_gain and gain > 0:
                # Tie-breaking: prefer words with higher "centrality" or random?
                # Let's stick to deterministic for reproducibility first.
                # Or prefer lower index?
                pass
                
        if best_candidate == -1:
            break
            
        code.append(best_candidate)
        uncovered -= neighbor_sets[best_candidate]
        
    # Post-processing: Remove redundant codewords
    # A codeword is redundant if all its neighbors are covered by the other codewords in the code.
    # We can iterate and remove any that are redundant.
    
    def is_redundant(idx, code_list, neighbor_sets, total_words):
        # Check if the ball of idx is covered by the union of balls of others
        # This is expensive to check fully every time.
        # Simpler: Check if there is any word in ball(idx) that is ONLY covered by idx in the current code.
        # We need to know for each word, how many codewords cover it.
        pass
        
    # A faster check for redundancy:
    # Reconstruct coverage counts for the current code.
    # If any codeword has a unique uncovered contribution, it's not redundant.
    # Actually, let's just do a simple pass:
    # For each c in code, check if removing it leaves any word uncovered.
    # This is O(|code| * |code| * ball_size). 73 * 73 * 13 ~ 70k ops. Very fast.
    
    final_code = list(code)
    changed = True
    while changed:
        changed = False
        for i in range(len(final_code)):
            # Temporarily remove final_code[i]
            temp_code = final_code[:i] + final_code[i+1:]
            # Check if temp_code covers everything
            # This is expensive if done naively.
            # Better: Maintain a count of how many codewords cover each word.
            pass
            
    # Let's implement the count-based redundancy removal.
    # counts[w] = number of codewords in final_code that cover w
    
    def check_coverage(code_list):
        counts = np.zeros(total_words, dtype=int)
        for c in code_list:
            for ng in neighbor_sets[c]:
                counts[ng] += 1
        return counts
        
    counts = check_coverage(final_code)
    # If any count is 0, it's not covered (shouldn't happen from greedy)
    
    # Remove redundants
    i = 0
    while i < len(final_code):
        c = final_code[i]
        # Check if c is the ONLY cover for any word in its ball
        is_only_cover = False
        for ng in neighbor_sets[c]:
            if counts[ng] == 1:
                is_only_cover = True
                break
        if not is_only_cover:
            # Remove c
            final_code.pop(i)
            # Update counts
            for ng in neighbor_sets[c]:
                counts[ng] -= 1
            changed = True
            # Do not increment i, check the same index again
        else:
            i += 1
            
    # Convert back to list of lists
    return [list(all_words[c]) for c in final_code]
