import time
import itertools
import random

def int_to_base3(x, n):
    """Convert integer x to a list of n base-3 digits."""
    digits = []
    for _ in range(n):
        digits.append(x % 3)
        x //= 3
    return digits

def base3_to_int(vec):
    """Convert a list of base-3 digits to an integer."""
    result = 0
    for i, d in enumerate(reversed(vec)):
        result += d * (3 ** i)
    return result

def neighbors(word):
    """Generate all words within Hamming distance 1 of `word`."""
    n = len(word)
    result = [word[:]]
    for i in range(n):
        for d in range(3):
            if d != word[i]:
                new_word = word[:]
                new_word[i] = d
                result.append(new_word)
    return result

def greedy_cover(q, n, radius, time_limit):
    """Greedy construction of a covering code."""
    all_words = [int_to_base3(i, n) for i in range(q ** n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    neighbors_map = {}
    for i, w in enumerate(all_words):
        neigh = neighbors(w)
        neighbors_map[i] = [word_to_idx[tuple(nw)] for nw in neigh]
    
    covered = [False] * (q ** n)
    uncovered_count = q ** n
    codewords = []
    
    start_time = time.time()
    
    while uncovered_count > 0:
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_new_covered = 0
        
        # Find the word that covers the most uncovered words
        for i in range(q ** n):
            if covered[i]:
                continue
            new_covered = 0
            for j in neighbors_map[i]:
                if not covered[j]:
                    new_covered += 1
            if new_covered > best_new_covered:
                best_new_covered = new_covered
                best_idx = i
        
        if best_idx == -1:
            break
        
        # Add this word as a codeword
        codewords.append(all_words[best_idx])
        for j in neighbors_map[best_idx]:
            if not covered[j]:
                covered[j] = True
                uncovered_count -= 1
        
        if uncovered_count == 0:
            break
    
    return codewords

def refine(codewords, q, n, radius, time_limit):
    """Refine the set of codewords using local search."""
    all_words = [int_to_base3(i, n) for i in range(q ** n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word
    neighbors_map = {}
    for i, w in enumerate(all_words):
        neigh = neighbors(w)
        neighbors_map[i] = [word_to_idx[tuple(nw)] for nw in neigh]
    
    # Convert codewords to indices
    cw_indices = [word_to_idx[tuple(cw)] for cw in codewords]
    
    covered = [False] * (q ** n)
    coverage_count = [0] * (q ** n)  # How many codewords cover each word
    
    for idx in cw_indices:
        for j in neighbors_map[idx]:
            covered[j] = True
            coverage_count[j] += 1
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try to remove a codeword if it's redundant
        for i, idx in enumerate(cw_indices):
            if time.time() - start_time > time_limit:
                break
            
            # Check if all words covered by this codeword are also covered by others
            redundant = True
            for j in neighbors_map[idx]:
                if coverage_count[j] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove this codeword
                for j in neighbors_map[idx]:
                    coverage_count[j] -= 1
                cw_indices.pop(i)
                improved = True
                break
        
        # Try to replace two codewords with one
        if not improved and len(cw_indices) > 1:
            for i in range(len(cw_indices)):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, len(cw_indices)):
                    # Check if there's a word that covers the union of the two codewords' coverage
                    # This is expensive, so we only try a few candidates
                    pass  # Skip this for now due to time constraints
    
    return [all_words[idx] for idx in cw_indices]

def construct(q, n, radius):
    """Main function to construct the covering code."""
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    # Initial greedy construction
    codewords = greedy_cover(q, n, radius, time_limit - 20)
    
    # Refine
    if time.time() - start_time < time_limit:
        codewords = refine(codewords, q, n, radius, time_limit - (time.time() - start_time))
    
    return codewords
