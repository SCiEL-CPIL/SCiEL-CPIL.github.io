import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits = [[0]*n for _ in range(total_words)]
    for w in range(total_words):
        val = w
        for i in range(n):
            digits[w][i] = val % q
            val //= q
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    neighbors = [[] for _ in range(total_words)]
    powers = [q ** i for i in range(n)]
    
    for w in range(total_words):
        neighbors[w].append(w)  # Distance 0
        if radius >= 1:
            for pos in range(n):
                current_digit = digits[w][pos]
                for new_digit in range(q):
                    if new_digit != current_digit:
                        nw = w + (new_digit - current_digit) * powers[pos]
                        neighbors[w].append(nw)
    
    # Build reverse neighbors (same as neighbors since symmetric for Hamming distance)
    # For radius=1, neighbors are symmetric
    
    # Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    uncovered_count = np.array([len(neighbors[i]) for i in range(total_words)], dtype=np.int32)
    chosen_mask = np.zeros(total_words, dtype=bool)
    
    code = []
    
    while not np.all(covered):
        if time.time() > deadline:
            break
            
        # Find the word with max uncovered_count among unchosen words
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            if not chosen_mask[i] and uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
                
        if best_idx == -1:
            break
            
        code.append(best_idx)
        chosen_mask[best_idx] = True
        
        # Update covered and counts
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                for i in neighbors[j]:
                    uncovered_count[i] -= 1
    
    # Local search: try to remove redundant codewords
    if time.time() < deadline:
        code = _local_search_reduce(code, neighbors, total_words, digits, deadline)
    
    # Local search: try 2-for-1 swaps (replace 2 codewords with 1 if possible)
    if time.time() < deadline:
        code = _try_swaps(code, neighbors, total_words, digits, deadline)
    
    # Convert integers back to lists of digits
    result = []
    for w in code:
        result.append(digits[w])
        
    return result

def _local_search_reduce(code, neighbors, total_words, digits, deadline):
    """Try to remove codewords that are redundant (their coverage is fully covered by others)."""
    code_set = set(code)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i, cw in enumerate(code):
            if time.time() > deadline:
                break
            # Check if removing cw still covers all its neighbors
            # A neighbor j is still covered if some other codeword in code covers j
            redundant = True
            for j in neighbors[cw]:
                if j == cw:
                    # j is cw itself; need another codeword within distance 1 of cw
                    found = False
                    for other in code:
                        if other != cw and other in neighbors[j]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                else:
                    # j needs to be covered by some other codeword
                    found = False
                    for other in code:
                        if other != cw and other in neighbors[j]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
            if redundant:
                code.pop(i)
                code_set.discard(cw)
                changed = True
                break  # Restart after removal
    return code

def _try_swaps(code, neighbors, total_words, digits, deadline):
    """Try to replace pairs of codewords with single codewords if possible."""
    code_set = set(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(code)):
                if time.time() > deadline:
                    break
                cw1 = code[i]
                cw2 = code[j]
                # Find union of neighborhoods
                union = set(neighbors[cw1]) | set(neighbors[cw2])
                # Check if there exists a word w such that neighbors[w] covers union
                # And w is not already in code (or if it is, we still can remove cw1,cw2)
                for w in union:
                    if w in code_set and w not in (cw1, cw2):
                        # w already in code, check if it covers union
                        if set(neighbors[w]) >= union:
                            # Remove cw1 and cw2
                            code.remove(cw1)
                            code.remove(cw2)
                            code_set.discard(cw1)
                            code_set.discard(cw2)
                            improved = True
                            break
                    elif w not in code_set:
                        # Check if adding w and removing cw1,cw2 works
                        if set(neighbors[w]) >= union:
                            code[i] = w
                            code.pop(j)
                            code_set.discard(cw1)
                            code_set.discard(cw2)
                            code_set.add(w)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return code
