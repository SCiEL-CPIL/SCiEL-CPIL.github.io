"""Ternary covering code K_3(8,1) via greedy set-cover + local search."""
import time
import itertools
import numpy as np
import random


def construct(q, n, radius):
    """
    Build a covering code of length n over alphabet size q with covering radius.
    For q=3, n=8, radius=1, use an optimized greedy + local search approach.
    """
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    
    # General fallback: greedy set cover
    return _greedy_general(q, n, radius)


def _solve_ternary_8_1():
    """Optimized solver for K_3(8,1)."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Precompute all words and neighbor relationships
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Map word -> index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    # For each index, precompute the set of indices it covers (within Hamming distance <= 1)
    # Each codeword covers itself + 7 neighbors (change one of 8 positions to 2 other values)
    # Total neighbors per word: 1 + 8*2 = 17
    neighbors = []
    for i in range(total):
        w = list(all_words[i])
        nbrs = set()
        nbrs.add(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    w[pos] = val
                    nbrs.add(word_to_idx[tuple(w)])
                    w[pos] = all_words[i][pos]
        neighbors.append(sorted(nbrs))
    
    # Convert to numpy arrays for speed
    neighbors_arr = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute a lookup: for each word, which codewords cover it (inverse neighbors)
    # This helps with fast redundancy checks
    # covered_by[w] = set of indices i such that w in neighbors[i]
    covered_by = [set() for _ in range(total)]
    for i in range(total):
        for j in neighbors[i]:
            covered_by[j].add(i)
    
    # Greedy set cover with incremental updates
    covered = np.zeros(total, dtype=np.uint8)
    # uncovered_count[i] = number of uncovered words that codeword i would cover
    uncovered_count = np.array([len(nb) for nb in neighbors], dtype=np.int32)
    
    codewords = []
    selected = []
    
    # For efficient incremental updates, we track which words are newly covered
    while time.time() < deadline:
        # Find candidate with max uncovered_count
        best_idx = int(np.argmax(uncovered_count))
        best_count = uncovered_count[best_idx]
        
        if best_count == 0:
            break
        
        # Add this codeword
        codewords.append(list(all_words[best_idx]))
        selected.append(best_idx)
        
        # Update covered and uncovered_count incrementally
        newly_covered = []
        for j in neighbors_arr[best_idx]:
            if not covered[j]:
                covered[j] = 1
                newly_covered.append(j)
        
        # Update uncovered_count for all candidates that share neighbors with newly_covered words
        # For each newly covered word w, all codewords that cover w lose 1 from their uncovered_count
        for w in newly_covered:
            for i in covered_by[w]:
                uncovered_count[i] -= 1
        
        if covered.all():
            break
    
    if not covered.all():
        # Should not happen for this problem, but just in case
        pass
    
    # Local search: try to remove redundant codewords and swap to reduce size
    # Represent the code as a set of indices
    code_set = set(selected)
    
    # Simpler approach: maintain a coverage count array
    # coverage_count[w] = number of codewords that cover word w
    coverage_count = np.zeros(total, dtype=np.int32)
    for cw in code_set:
        for w in neighbors_arr[cw]:
            coverage_count[w] += 1
    
    # Now redundancy check is: a codeword is redundant if for all w in neighbors[cw], coverage_count[w] > 1
    def is_redundant_fast(cw_idx):
        return np.all(coverage_count[neighbors_arr[cw_idx]] > 1)
    
    # Remove redundant codewords
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for cw in list(code_set):
            if is_redundant_fast(cw):
                for w in neighbors_arr[cw]:
                    coverage_count[w] -= 1
                code_set.remove(cw)
                codewords.remove(list(all_words[cw]))
                removed = True
    
    # Local search: try swapping codewords to reduce size
    # Try: for each codeword, try moving it to a different position and see if it allows removing another codeword
    # Or: try adding a new codeword that makes an existing one redundant
    
    # Better approach: iteratively try to find a codeword that can be moved to a better position
    # to make another codeword redundant
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to find a pair (cw_to_remove, new_position) such that:
        # 1. cw_to_remove becomes redundant if we add a codeword at new_position
        # 2. The new codeword at new_position is not already in the code
        
        # For each codeword cw_to_remove, find words that are only covered by it
        # If we can cover those words with a new codeword, we can remove cw_to_remove
        
        for cw_to_remove in list(code_set):
            if time.time() > deadline:
                break
            
            # Find words that are exclusively covered by cw_to_remove
            exclusive_words = []
            for w in neighbors_arr[cw_to_remove]:
                if coverage_count[w] == 1:
                    exclusive_words.append(w)
            
            if not exclusive_words:
                continue
            
            # Try to find a position that covers all exclusive_words
            # A position p covers a word w if w is in neighbors[p]
            # We need p such that all exclusive_words are in neighbors[p]
            # This means p must be within distance 1 of all exclusive_words
            
            # For each exclusive word, the set of positions that cover it is neighbors[exclusive_word]
            # We need the intersection of all these sets
            if len(exclusive_words) > 1:
                # Intersection of neighbors
                candidate_positions = set(neighbors_arr[exclusive_words[0]])
                for ew in exclusive_words[1:]:
                    candidate_positions &= set(neighbors_arr[ew])
            else:
                candidate_positions = set(neighbors_arr[exclusive_words[0]])
            
            # Remove positions already in the code
            candidate_positions -= code_set
            
            if not candidate_positions:
                continue
            
            # Try each candidate position
            for new_pos in candidate_positions:
                # Check if adding new_pos and removing cw_to_remove maintains coverage
                # A word w loses coverage if:
                # - w is in neighbors[cw_to_remove]
                # - w is not in neighbors[new_pos]
                # - coverage_count[w] == 1 (only covered by cw_to_remove)
                
                # Use a set for faster membership testing
                new_pos_neighbors = set(neighbors_arr[new_pos])
                
                valid = True
                for w in neighbors_arr[cw_to_remove]:
                    if coverage_count[w] == 1 and w not in new_pos_neighbors:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Perform the swap
                for w in neighbors_arr[cw_to_remove]:
                    coverage_count[w] -= 1
                for w in neighbors_arr[new_pos]:
                    coverage_count[w] += 1
                
                code_set.remove(cw_to_remove)
                code_set.add(new_pos)
                codewords.remove(list(all_words[cw_to_remove]))
                codewords.append(list(all_words[new_pos]))
                
                improved = True
                break  # Restart the loop after a change
            
            if improved:
                break
        
        # After a round of swaps, try to remove redundant codewords again
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for cw in list(code_set):
                if is_redundant_fast(cw):
                    for w in neighbors_arr[cw]:
                        coverage_count[w] -= 1
                    code_set.remove(cw)
                    codewords.remove(list(all_words[cw]))
                    removed = True
    
    # Final cleanup: sort codewords for consistency
    codewords.sort()
    
    return codewords


def _greedy_general(q, n, radius):
    """General greedy set cover for other parameters."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    def get_neighbors(w):
        yield tuple(w)
        if radius >= 1:
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        new = list(w)
                        new[i] = v
                        yield tuple(new)
    
    covered = [False] * total
    codewords = []
    
    neighbors_list = []
    for i in range(total):
        nbrs = set()
        for cand in get_neighbors(all_words[i]):
            nbrs.add(word_to_idx(cand))
        neighbors_list.append(nbrs)
    
    uncovered_count = [len(neighbors_list[i]) for i in range(total)]
    
    while time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
            if best_count == 0:
                break
        
        if best_idx == -1 or best_count == 0:
            break
        
        codewords.append(list(all_words[best_idx]))
        
        for j in neighbors_list[best_idx]:
            if not covered[j]:
                covered[j] = True
        
        new_uncovered_count = [0] * total
        for i in range(total):
            cnt = 0
            for j in neighbors_list[i]:
                if not covered[j]:
                    cnt += 1
            new_uncovered_count[i] = cnt
        uncovered_count = new_uncovered_count
        
        if all(covered):
            break
    
    return codewords
