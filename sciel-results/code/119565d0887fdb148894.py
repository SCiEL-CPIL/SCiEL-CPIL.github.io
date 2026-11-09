"""
Ternary covering code K_3(6,1) construction.
Uses bitset-based coverage tracking with greedy initialization and local search reduction.
"""
import itertools
import time
import random
import numpy as np

def _direct_sum_construction(q, n, radius):
    """Fallback: direct sum of ternary tetracode with all words on remaining coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (sphere of radius 1) for each word as a set of indices
    neighbors = []
    for w in all_words:
        w_list = list(w)
        nbrs = {word_to_idx[w]}
        for i in range(n):
            orig = w_list[i]
            for v in range(q):
                if v != orig:
                    nw = w_list[:]
                    nw[i] = v
                    nbrs.add(word_to_idx[tuple(nw)])
        neighbors.append(nbrs)
    
    # Strategy 1: Start with a good initial code
    # First try greedy set cover for a potentially smaller initial code
    uncovered = set(range(total_words))
    greedy_code = []
    
    # Precompute neighbor arrays for speed
    neighbor_arrays = [list(ns) for ns in neighbors]
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words
    while uncovered and time.time() < deadline - 60:
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            count = 0
            for nbr in neighbor_arrays[i]:
                if nbr in uncovered:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
            if time.time() > deadline - 65:
                break
        if best_idx < 0 or best_count == 0:
            break
        
        greedy_code.append(all_words[best_idx])
        for nbr in neighbor_arrays[best_idx]:
            uncovered.discard(nbr)
    
    # Choose between greedy and direct sum based on size
    if len(greedy_code) < 81:
        initial_code = greedy_code
    else:
        initial_code = _direct_sum_construction(q, n, radius)
    
    code_set = set(tuple(cw) for cw in initial_code)
    
    # Track which words are covered and by how many codewords
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_count[nbr] += 1
    
    # Verify initial code covers everything
    if np.any(coverage_count == 0):
        # Fallback if something went wrong
        return initial_code
    
    # Local search: try to remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for cw_tuple in list(code_set):
            if time.time() > deadline - 5:
                break
            idx = word_to_idx[cw_tuple]
            # Check if removing this codeword still covers all words
            # A word is uniquely covered if coverage_count[word] == 1 and it's in neighbors[idx]
            uniquely_covered = False
            for nbr in neighbors[idx]:
                if coverage_count[nbr] == 1:
                    uniquely_covered = True
                    break
            if not uniquely_covered:
                # Safe to remove
                for nbr in neighbors[idx]:
                    coverage_count[nbr] -= 1
                code_set.remove(cw_tuple)
                changed = True
    
    # Now try swap-based local search: replace one codeword with another to enable further removals
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        # Find words that are covered by exactly one codeword
        singly_covered = np.where(coverage_count == 1)[0]
        if len(singly_covered) == 0:
            break
        
        # For each codeword, see how many singly-covered words it covers
        best_removal = None
        best_removal_count = 0
        for cw_tuple in code_set:
            idx = word_to_idx[cw_tuple]
            count = 0
            for nbr in neighbors[idx]:
                if coverage_count[nbr] == 1:
                    count += 1
            if count > best_removal_count:
                best_removal_count = count
                best_removal = cw_tuple
        
        if best_removal is None or best_removal_count == 0:
            break
        
        # Try to find a replacement codeword that covers most of the same words
        rem_idx = word_to_idx[best_removal]
        rem_neighbors = neighbors[rem_idx]
        
        # Find a candidate replacement
        best_add = None
        best_overlap = 0
        for i in range(total_words):
            if i == rem_idx:
                continue
            # How many of rem_neighbors does candidate i cover?
            overlap = 0
            for nbr in rem_neighbors:
                if nbr in neighbors[i]:
                    overlap += 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_add = i
        
        if best_add is not None and best_overlap >= best_removal_count:
            # Try the swap
            add_idx = best_add
            add_neighbors = neighbors[add_idx]
            
            # Check if after removing best_removal and adding best_add, all words are still covered
            # Words that were only covered by best_removal must be covered by best_add
            can_swap = True
            for nbr in rem_neighbors:
                if coverage_count[nbr] == 1:
                    if nbr not in add_neighbors:
                        can_swap = False
                        break
            
            if can_swap:
                # Perform swap
                for nbr in rem_neighbors:
                    coverage_count[nbr] -= 1
                for nbr in add_neighbors:
                    coverage_count[nbr] += 1
                code_set.remove(best_removal)
                code_set.add(tuple(all_words[add_idx]))
                
                # Now try to remove more codewords
                changed2 = True
                while changed2 and time.time() < deadline - 5:
                    changed2 = False
                    for cw_tuple2 in list(code_set):
                        idx2 = word_to_idx[cw_tuple2]
                        uniquely_covered2 = False
                        for nbr in neighbors[idx2]:
                            if coverage_count[nbr] == 1:
                                uniquely_covered2 = True
                                break
                        if not uniquely_covered2:
                            for nbr in neighbors[idx2]:
                                coverage_count[nbr] -= 1
                            code_set.remove(cw_tuple2)
                            changed2 = True
    
    # Final verification
    coverage_final = np.zeros(total_words, dtype=np.int32)
    for cw in code_set:
        idx = word_to_idx[cw]
        for nbr in neighbors[idx]:
            coverage_final[nbr] += 1
    
    if np.all(coverage_final > 0):
        return [list(cw) for cw in code_set]
    else:
        # Fallback to direct sum
        return _direct_sum_construction(q, n, radius)
