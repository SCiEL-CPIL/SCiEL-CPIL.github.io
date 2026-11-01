import itertools
import random
import time
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Specific case for the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other parameters (simple product construction)
    if n <= 0:
        return [[]]
    
    # Generic greedy set cover for other cases
    all_words = list(itertools.product(range(q), repeat=n))
    space_size = q ** n
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Greedy set cover
    covered = [False] * space_size
    code = []
    remaining = set(range(space_size))
    
    while remaining and time.time() < deadline - 5.0:
        # Find uncovered word with most potential
        best_idx = -1
        best_count = -1
        for idx in list(remaining):
            if not covered[idx]:
                count = sum(1 for w in neighbors[idx] if not covered[w])
                if count > best_count:
                    best_count = count
                    best_idx = idx
        if best_idx == -1:
            break
        code.append(all_words[best_idx])
        for w in neighbors[best_idx]:
            covered[w] = True
        remaining.discard(best_idx)
    
    return code

def _solve_q3_n6_r1(deadline):
    n = 6
    q = 3
    space_size = 729
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1) for each word
    # Using numpy for faster operations if possible, but lists are fine for 729
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Initial code: Tetracode product (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
    initial_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    best_code = initial_code
    best_size = len(best_code)
    
    rng = random.Random(42)
    
    # Helper to check if a set of indices covers the space
    def is_valid_cover(indices):
        cover_count = [0] * space_size
        for idx in indices:
            for w_idx in neighbors[idx]:
                cover_count[w_idx] += 1
        # Check if any is 0
        for c in cover_count:
            if c == 0:
                return False
        return True

    # Helper to get coverage counts
    def get_coverage(indices):
        cover_count = [0] * space_size
        for idx in indices:
            for w_idx in neighbors[idx]:
                cover_count[w_idx] += 1
        return cover_count

    # Local Search with 2-for-1 swaps
    # We maintain the current set of indices and their coverage counts
    def local_search_2for1(indices, deadline):
        current_indices = set(indices)
        cover_count = get_coverage(current_indices)
        
        # Precompute unique coverage for each codeword? 
        # A word is "uniquely covered" by c if cover_count[w] == 1 and w is in neighbors[c].
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Try to remove a single codeword
            for idx in list(current_indices):
                # Check if removal is valid
                # A word w is affected if it is in neighbors[idx]
                # It becomes uncovered if cover_count[w] == 1
                removable = True
                for w in neighbors[idx]:
                    if cover_count[w] == 1:
                        removable = False
                        break
                
                if removable:
                    current_indices.remove(idx)
                    for w in neighbors[idx]:
                        cover_count[w] -= 1
                    improved = True
                    # Restart loop to check for further removals
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1 swap: remove i, j and add k
            # To be efficient, we only consider pairs (i, j) that are "bad" in some way?
            # Or just iterate. N is small (~75). N^2 is ~5600.
            # For each pair, we need to find if there exists a k such that
            # the union of neighbors[k] covers all words that were uniquely covered by i or j.
            
            # Let U_i be words uniquely covered by i.
            # Let U_j be words uniquely covered by j.
            # Let U_ij = U_i union U_j.
            # We need a k such that neighbors[k] contains U_ij.
            # Also, k must not be in current_indices (or if it is, we are just swapping, size doesn't change, so skip).
            
            # Optimization: Precompute for each word w, the list of codewords that cover it?
            # No, let's just iterate k for each pair.
            
            # To speed up, we can first check if there is any k that covers U_i and U_j.
            # We can precompute a "signature" or just check.
            
            # Let's try a few random pairs first, or all pairs if fast enough.
            # 5600 pairs * 729 words check? No, we check if neighbors[k] covers U_ij.
            # |U_ij| is small (usually < 10).
            # So check is fast.
            
            indices_list = list(current_indices)
            L = len(indices_list)
            
            # Find all valid swaps
            found_swap = False
            for a in range(L):
                i = indices_list[a]
                # Words uniquely covered by i
                u_i = [w for w in neighbors[i] if cover_count[w] == 1]
                
                for b in range(a + 1, L):
                    j = indices_list[b]
                    # Words uniquely covered by j
                    u_j = [w for w in neighbors[j] if cover_count[w] == 1]
                    
                    # Union of unique words
                    # Note: if a word is in both u_i and u_j, it's covered twice? 
                    # No, if cover_count[w] == 1, it is covered by exactly one codeword.
                    # So u_i and u_j are disjoint.
                    u_union = u_i + u_j
                    
                    if not u_union:
                        # If no unique words, we can just remove one? 
                        # But we already tried removal. 
                        # If u_i is empty, i can be removed. 
                        # So this case shouldn't happen if removal was tried first.
                        continue
                    
                    # Find a k not in current_indices such that neighbors[k] covers u_union
                    # Also, adding k should not break coverage of other words?
                    # Adding k only increases coverage, so it won't break anything.
                    # The only risk is if we remove i and j, some words become uncovered.
                    # Those are exactly u_union.
                    # So we just need k to cover u_union.
                    
                    for k in range(space_size):
                        if k in current_indices:
                            continue
                        
                        # Check if k covers u_union
                        covers = True
                        # Optimization: Check intersection of neighbors[k] and u_union
                        # If len(set(neighbors[k]) & set(u_union)) == len(u_union)
                        # Since u_union is small, just check each w in u_union
                        for w in u_union:
                            if w not in neighbors[k]:
                                covers = False
                                break
                        
                        if covers:
                            # Perform swap
                            current_indices.remove(i)
                            current_indices.remove(j)
                            current_indices.add(k)
                            
                            # Update cover_count
                            for w in neighbors[i]:
                                cover_count[w] -= 1
                            for w in neighbors[j]:
                                cover_count[w] -= 1
                            for w in neighbors[k]:
                                cover_count[w] += 1
                            
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break
                
                if found_swap:
                    break
            
            # If no 2-for-1 swap found, try 1-for-1 swap to escape local minima?
            # Or stop.
            
        return list(current_indices)

    # Run local search from initial code
    if time.time() < deadline - 5.0:
        current_indices = local_search_2for1(initial_indices, deadline)
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code = [all_words[i] for i in current_indices]
            
    # Try a few random restarts if time permits?
    # The 2-for-1 search is deterministic given the order.
    # We can shuffle the initial indices to break symmetry.
    
    for trial in range(10):
        if time.time() > deadline - 2.0:
            break
        
        shuffled_indices = list(initial_indices)
        rng.shuffle(shuffled_indices)
        
        current_indices = local_search_2for1(shuffled_indices, deadline)
        
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code = [all_words[i] for i in current_indices]
            
    return best_code
