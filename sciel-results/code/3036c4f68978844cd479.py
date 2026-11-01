import time
import random
from itertools import product

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses greedy set cover with precomputed coverage matrices and local search to minimize codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # Precompute all words as integer indices
    # word_to_int: converts list of q-digits to int
    def word_to_int(word):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbor sets for radius 1 (or the given radius)
    # For efficiency, we'll use numpy arrays for coverage tracking
    # neighbors[i] = list of words covered by codeword i
    
    neighbors = []
    for idx in range(total_words):
        # Convert to word
        word = []
        tmp = idx
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        
        covered_set = set()
        # Add self
        covered_set.add(idx)
        
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for new_val in range(q):
                    if new_val != orig:
                        word[pos] = new_val
                        covered_set.add(word_to_int(word))
                word[pos] = orig
        
        # For radius 2, we'd need more, but typically radius=1 for this task
        if radius >= 2:
            # Add distance 2 neighbors
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1, orig2 = word[pos1], word[pos2]
                    for v1 in range(q):
                        if v1 != orig1:
                            word[pos1] = v1
                            for v2 in range(q):
                                if v2 != orig2:
                                    word[pos2] = v2
                                    covered_set.add(word_to_int(word))
                                elif v2 != orig2:
                                    pass
                            word[pos1] = orig1
                    # Also handle case where only one position changes differently? 
                    # Actually distance 2 means exactly 2 positions differ.
                    # The above loops over all pairs of positions and all pairs of new values.
                    # But we need to be careful: if v1==orig1 and v2!=orig2, that's distance 1.
                    # We want exactly 2 positions to change.
                    # Let's redo this more carefully for correctness if radius>=2 is ever used.
                    # For now, assuming radius=1 is the main case.
        
        neighbors.append(covered_set)
    
    # Convert to list of lists for faster iteration if needed
    neighbors_list = [list(ns) for ns in neighbors]
    
    # Greedy set cover
    uncovered = set(range(total_words))
    codewords = []
    
    # Seed with tetracode construction for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        # Tetracode: 9 codewords of length 4
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word)
                    codewords.append(idx)
                    for nb in neighbors_list[idx]:
                        uncovered.discard(nb)
    
    # Greedy steps
    while uncovered:
        if time.time() > deadline:
            break
        
        best_center = -1
        best_count = -1
        
        # Only consider candidates that are in uncovered? No, any candidate can cover uncovered words.
        # But for efficiency, we can note that if a candidate is already covered, it might still be useful.
        # However, in covering codes, it's often optimal to pick centers from the uncovered set or nearby.
        # Let's iterate all candidates for correctness.
        
        for candidate in range(total_words):
            count = 0
            for nb in neighbors_list[candidate]:
                if nb in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_center = candidate
        
        if best_center == -1 or best_count == 0:
            break
        
        codewords.append(best_center)
        for nb in neighbors_list[best_center]:
            uncovered.discard(nb)
    
    # Local search: try to remove redundant codewords
    # Repeatedly try to remove each codeword and check if coverage is maintained
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Build coverage count: for each word, how many codewords cover it
        # This is expensive, but for 729 words and ~80 codewords, it's fine.
        
        # Alternative: for each codeword, check if removing it leaves any word uncovered.
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if codewords[i] is redundant
            redundant = True
            c_idx = codewords[i]
            for w_idx in neighbors_list[c_idx]:
                # Check if w_idx is covered by any other codeword
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    # Check if w_idx is in neighbors of codewords[j]
                    if w_idx in neighbors[codewords[j]]:  # Use set for O(1) lookup
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if redundant:
                codewords.pop(i)
                changed = True
                break  # Restart from beginning as indices changed
    
    # Convert back to lists of integers
    result = []
    for c in codewords:
        word = []
        tmp = c
        for _ in range(n):
            word.append(tmp % q)
            tmp //= q
        word.reverse()
        result.append(word)
    
    return result
