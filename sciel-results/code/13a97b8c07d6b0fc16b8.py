import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all possible words
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: cover[i] is the set of indices covered by word i
    # For radius=1, q=3, n=6: each word covers itself + 6*2 = 12 neighbors = 13 total
    cover = [None] * N
    for i, w in enumerate(all_words):
        covered = set()
        covered.add(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_word = list(w)
                    new_word[pos] = val
                    j = word_to_idx[tuple(new_word)]
                    covered.add(j)
        cover[i] = covered
    
    # Greedy set cover
    uncovered = set(range(N))
    code = []
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
        best_idx = None
        best_count = -1
        for i in range(N):
            if i in [word_to_idx[c] for c in code]:  # This is inefficient
                continue
            count = len(cover[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx is None or best_count == 0:
            break
        code.append(all_words[best_idx])
        uncovered -= cover[best_idx]
    
    # Convert code to set of indices for efficiency
    code_indices = set(word_to_idx[c] for c in code)
    
    # Local search: try to remove redundant codewords
    improved = True
    while improved:
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in list(code_indices):
            # Check if removing i still covers everything
            temp_indices = code_indices - {i}
            # Check coverage
            covered_set = set()
            for j in temp_indices:
                covered_set |= cover[j]
            if covered_set == set(range(N)):
                code_indices.remove(i)
                improved = True
                break  # Restart the loop
    
    # Try pairwise removal with single addition
    improved = True
    while improved:
        if time.time() > deadline - 15.0:
            break
        improved = False
        indices_list = list(code_indices)
        for a in range(len(indices_list)):
            for b in range(a+1, len(indices_list)):
                i, j = indices_list[a], indices_list[b]
                temp_indices = code_indices - {i, j}
                covered_set = set()
                for k in temp_indices:
                    covered_set |= cover[k]
                uncovered_set = set(range(N)) - covered_set
                if not uncovered_set:
                    # Can remove both
                    code_indices.remove(i)
                    code_indices.remove(j)
                    improved = True
                    break
                # Check if we can cover uncovered_set with a single word
                if len(uncovered_set) <= 13:
                    for w in range(N):
                        if cover[w] >= uncovered_set:
                            # Check if w is not already in code
                            if w not in code_indices:
                                code_indices.remove(i)
                                code_indices.remove(j)
                                code_indices.add(w)
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    
    return [all_words[i] for i in code_indices]
