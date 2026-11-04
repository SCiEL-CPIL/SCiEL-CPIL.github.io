"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1.
    # Total words: 3^6 = 729.
    # We use a greedy set cover algorithm with bitsets for efficiency,
    # followed by a local search to minimize the number of codewords.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (though the task is specific)
        # Simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        covered = set()
        codewords = []
        for c in all_words:
            c = list(c)
            new_covered = []
            for i in range(n):
                for v in range(q):
                    if v != c[i]:
                        w = c[:]
                        w[i] = v
                        new_covered.append(tuple(w))
            new_covered.append(tuple(c))
            
            added = 0
            for w in new_covered:
                if w not in covered:
                    added += 1
            
            if added > 0:
                covered.update(new_covered)
                codewords.append(c)
                if len(covered) == len(all_words):
                    break
        return codewords

    # Specific optimization for q=3, n=6, r=1
    num_words = 729
    
    # Map each word to an index 0..728
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    # Precompute the set of words covered by each potential codeword
    # coverage[c] is a set of indices of words covered by codeword c
    # There are 729 potential codewords.
    # Each covers 1 + 3*5 = 16 words.
    # Total operations: 729 * 16 = 11664. Very fast.
    
    coverage = {}
    for c in range(num_words):
        w = [int(d) for d in f"{c:06d}".zfill(6)] # This is wrong, base 3 representation
        # Correct base 3 conversion
        tmp = c
        w = [0]*n
        for i in range(n-1, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        
        covered_indices = set()
        # The word itself
        covered_indices.add(c)
        
        # Words at distance 1
        for i in range(n):
            for v in range(3):
                if v != w[i]:
                    w2 = w[:]
                    w2[i] = v
                    # Convert w2 to index
                    idx2 = 0
                    for j in range(n):
                        idx2 = idx2 * 3 + w2[j]
                    covered_indices.add(idx2)
        coverage[c] = covered_indices

    # Greedy Set Cover
    # We want to select a subset of codewords such that the union of their coverage is all 729 words.
    # Greedy: at each step, pick the codeword that covers the most *uncovered* words.
    
    uncovered = set(range(num_words))
    selected = []
    
    # To make it faster, we can pre-sort or just iterate.
    # 729 iterations, each checking 729 candidates * 16 words. 729*729*16 ~ 8.5M ops. Might be too slow in Python?
    # Let's optimize.
    
    # Better Greedy:
    # Keep a list of uncovered words.
    # For each candidate c, count how many words in coverage[c] are in uncovered.
    # Pick max. Remove those from uncovered.
    
    # This is O(K * N * Q) where K is number of selected, N is number of candidates, Q is coverage size.
    # K~73, N=729, Q=16. 73 * 729 * 16 = 850k ops. This is very fast.
    
    while uncovered:
        best_c = -1
        best_count = 0
        for c in range(num_words):
            # Count overlap with uncovered
            # We can use set intersection
            count = len(coverage[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_count == 0:
            break # Should not happen
        
        selected.append(best_c)
        uncovered -= coverage[best_c]
        
        # Optional: Stop if we hit a target? No, let it finish to ensure coverage.
    
    # Now we have a valid code, likely around 80-90 words.
    # Local Search: Try to remove redundant codewords.
    # Sort selected codewords? Maybe try removing in order of "least unique coverage".
    
    # Convert selected indices to actual codeword lists
    def idx_to_word(idx):
        w = [0]*n
        tmp = idx
        for i in range(n-1, -1, -1):
            w[i] = tmp % 3
            tmp //= 3
        return w
    
    codewords = [idx_to_word(c) for c in selected]
    
    # Refinement: Try to remove each codeword if possible.
    # Repeat until no more can be removed.
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords) - 1, -1, -1):
            # Temporarily remove codeword i
            # Check if all words covered by codeword i are still covered by others.
            c_idx = selected[i]
            c_words = coverage[c_idx]
            
            # Check if any word in c_words is NOT covered by any other selected codeword
            is_redundant = True
            for w_idx in c_words:
                # Check if w_idx is covered by any j != i
                found = False
                for j in range(len(selected)):
                    if j == i:
                        continue
                    if w_idx in coverage[selected[j]]:
                        found = True
                        break
                if not found:
                    is_redundant = False
                    break
            
            if is_redundant:
                codewords.pop(i)
                selected.pop(i)
                changed = True
    
    # Further Refinement: Try swapping?
    # If we have > 73, we might be stuck. 
    # The greedy usually yields ~80. Removal might get us to ~75-78.
    # To get to 73, we might need a smarter start or swap.
    
    # Let's try a few random restarts if we are above 73.
    best_code = codewords
    best_len = len(codewords)
    
    # If we are already at 73, we are optimal (upper bound).
    if best_len > 73:
        # Try randomized greedy
        for _ in range(10): # Few restarts due to time
            random.shuffle(list(range(num_words)))
            # Re-run greedy but with random tie-breaking? 
            # Actually, the greedy choice is deterministic based on count.
            # We can add random noise to the count.
            
            uncovered = set(range(num_words))
            sel = []
            while uncovered:
                best_c = -1
                best_score = -1
                # Randomize by adding small noise
                scores = {}
                for c in range(num_words):
                    count = len(coverage[c] & uncovered)
                    if count > 0:
                        score = count + random.random() * 0.5
                        scores[c] = score
                        if score > best_score:
                            best_score = score
                            best_c = c
                if best_c == -1:
                    break
                sel.append(best_c)
                uncovered -= coverage[best_c]
            
            # Remove redundant
            for i in range(len(sel) - 1, -1, -1):
                c_idx = sel[i]
                c_words = coverage[c_idx]
                is_redundant = True
                for w_idx in c_words:
                    found = False
                    for j in range(len(sel)):
                        if j == i:
                            continue
                        if w_idx in coverage[sel[j]]:
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                if is_redundant:
                    sel.pop(i)
            
            if len(sel) < best_len:
                best_len = len(sel)
                best_code = [idx_to_word(c) for c in sel]
                
                if best_len == 73:
                    break
                    
    return best_code
# EVOLVE-BLOCK-END
