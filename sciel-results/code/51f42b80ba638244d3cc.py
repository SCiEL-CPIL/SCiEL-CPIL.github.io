# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using Greedy Set Cover followed by Local Search.
    """
    total = q ** n
    
    # Precompute neighbors for all words
    # neighbors[i] is a set of integers (encoded words) within distance <= radius of i
    neighbors = [None] * total
    
    def decode(v, base=q, length=n):
        out = []
        for _ in range(length):
            v, r = divmod(v, base)
            out.append(r)
        return tuple(reversed(out))

    def encode(word, base=q):
        v = 0
        for sym in word:
            v = v * base + sym
        return v

    # Generate neighbors
    for w in range(total):
        word = decode(w)
        neigh = set([w])
        if radius >= 1:
            for i in range(n):
                orig = word[i]
                for sym in range(q):
                    if sym != orig:
                        new_word = list(word)
                        new_word[i] = sym
                        neigh.add(encode(new_word))
        if radius >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    orig_i = word[i]
                    orig_j = word[j]
                    for sym_i in range(q):
                        for sym_j in range(q):
                            if sym_i == orig_i and sym_j == orig_j:
                                continue
                            new_word = list(word)
                            new_word[i] = sym_i
                            new_word[j] = sym_j
                            neigh.add(encode(new_word))
        neighbors[w] = neigh

    # --- Greedy Phase ---
    uncovered = set(range(total))
    code = []
    
    while uncovered:
        best_cw = -1
        best_gain = -1
        
        # Iterate through all possible codewords to find the one covering most uncovered words
        # Optimization: We can break early if best_gain == len(uncovered)
        for c in range(total):
            if c in code: 
                # We shouldn't add duplicates, but greedy set cover usually handles sets.
                # Here codewords are distinct positions. If we already picked c, its neighbors are covered.
                # So gain would be 0. We can skip if we track which are in code.
                continue
            
            # Count how many words in 'uncovered' are in neighbors[c]
            # Using set intersection is efficient
            gain = len(uncovered.intersection(neighbors[c]))
            
            if gain > best_gain:
                best_gain = gain
                best_cw = c
                
                if best_gain == len(uncovered):
                    break
        
        if best_cw == -1:
            break # Should not happen if uncovered is not empty
            
        code.append(best_cw)
        uncovered -= neighbors[best_cw]
        
    # --- Local Search Phase ---
    # We want to minimize len(code).
    # Strategy: Try to remove a codeword. If it breaks coverage, try to swap it with another codeword that fixes it.
    # If we can remove without breaking, great.
    
    # Convert code to a set for fast lookup
    code_set = set(code)
    
    # We need a way to quickly check if a word is covered by the current code.
    # We can maintain a count array.
    covered_count = [0] * total
    for c in code:
        for nb in neighbors[c]:
            covered_count[nb] += 1
            
    def is_fully_covered():
        return all(c > 0 for c in covered_count)
        
    def add_to_code(c):
        code_set.add(c)
        for nb in neighbors[c]:
            covered_count[nb] += 1
            
    def remove_from_code(c):
        code_set.remove(c)
        for nb in neighbors[c]:
            covered_count[nb] -= 1

    improved = True
    max_iterations = 100 # Safety limit for local search loops
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # Try to remove each codeword
        for i in range(len(code)):
            c_to_remove = code[i]
            
            # Remove it
            remove_from_code(c_to_remove)
            
            # Check if coverage is still valid
            if is_fully_covered():
                # Success! We reduced the size.
                code.pop(i)
                improved = True
                # Restart the loop as indices changed and state changed significantly
                break
            else:
                # Coverage broken. Find the set of uncovered words
                uncovered_now = set(w for w in range(total) if covered_count[w] == 0)
                
                # Try to find a replacement codeword that covers all uncovered_now
                # Ideally, we want a codeword that covers uncovered_now AND is not in code
                best_replace = -1
                best_gain = -1
                
                # We only care about covering 'uncovered_now'
                # But we must ensure the replacement doesn't leave other things uncovered?
                # No, if we restore coverage of 'uncovered_now', and everything else was covered 
                # by the code BEFORE removal (and we only removed c_to_remove), 
                # then everything else is still covered by the remaining code.
                # Wait, removing c_to_remove might have uncovered words that were ONLY covered by c_to_remove.
                # Those are exactly 'uncovered_now'.
                # So if we add a codeword that covers 'uncovered_now', coverage is restored.
                
                for c in range(total):
                    if c in code_set:
                        continue
                    
                    # Check if neighbors[c] covers all of uncovered_now
                    # i.e., uncovered_now is subset of neighbors[c]
                    if uncovered_now.issubset(neighbors[c]):
                        # This is a valid swap. 
                        # To be greedy, we might prefer one that covers more "redundant" words?
                        # Or just take the first one?
                        # Taking the first one is fine.
                        best_replace = c
                        break
                        
                if best_replace != -1:
                    # Perform swap
                    add_to_code(best_replace)
                    # Note: code list still has c_to_remove at index i, we replaced it logically in code_set
                    # But we need to update the list `code` to reflect the swap for future iterations
                    code[i] = best_replace
                    improved = True
                    break # Restart loop
                else:
                    # No valid swap found. Revert removal.
                    add_to_code(c_to_remove)
                    
    # Return the list of codewords (decoded)
    return [list(decode(c)) for c in code]

# EVOLVE-BLOCK-END
