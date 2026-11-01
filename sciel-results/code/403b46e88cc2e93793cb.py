"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from collections import deque

def construct(q, n, radius):
    """
    Constructs a covering code using greedy initialization followed by
    simulated annealing with bitmask-based coverage tracking and
    vectorized neighborhood evaluation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger spaces
        pass

    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Map from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
        
    # Precompute coverage indices for each word
    ball_size = 1 + n * (q - 1)  # For radius 1
    cover_indices = []
    for i in range(total_words):
        w = all_words[i]
        inds = [i]
        for j in range(n):
            for v in range(q):
                if v == w[j]:
                    continue
                new_w = w.copy()
                new_w[j] = v
                inds.append(word_to_idx[tuple(new_w)])
        cover_indices.append(inds)
        
    cover_matrix = np.array(cover_indices, dtype=np.int32)
    
    # Greedy initialization
    covered = np.zeros(total_words, dtype=bool)
    active = np.ones(total_words, dtype=bool)
    selected = []
    
    while True:
        if time.time() > deadline - 100.0:  # Reserve time for SA
            break
        if not active.any():
            break
            
        cov_status = covered[cover_matrix]
        new_counts = (~cov_status).sum(axis=1)
        new_counts[~active] = -1
        
        max_idx = np.argmax(new_counts)
        if new_counts[max_idx] <= 0:
            break
            
        selected.append(max_idx)
        covered[cover_matrix[max_idx]] = True
        active[max_idx] = False
        
        if covered.all():
            break
    
    # Simulated annealing
    # State: current code (list of indices), covered mask, uncovered count
    current_code = list(selected)
    current_covered = covered.copy()
    current_size = len(current_code)
    
    # Track which words are covered by which codewords (for incremental updates)
    # coverage_count[i] = number of codewords covering word i
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for idx in current_code:
        coverage_count[cover_matrix[idx]] += 1
    
    best_code = list(current_code)
    best_size = current_size
    best_covered = current_covered.copy()
    best_coverage_count = coverage_count.copy()
    
    # Precompute incompatibility: two codewords are incompatible if distance <= 2
    # For radius 1 covering, we can remove a codeword if all words it uniquely covers
    # are also covered by other codewords.
    
    # Temperature schedule
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Try moves: remove, add, swap
        
        # 1. Try removal: pick a random codeword, check if all its covered words
        #    are still covered by others
        if rng.random() < 0.4 and len(current_code) > 1:
            idx_pos = rng.randint(0, len(current_code) - 1)
            c_idx = current_code[idx_pos]
            
            # Check if any word is uniquely covered by this codeword
            cov_words = cover_matrix[c_idx]
            uniquely_covered = (coverage_count[cov_words] == 1)
            
            if not uniquely_covered.any():
                # Safe to remove
                new_code = list(current_code)
                new_code.pop(idx_pos)
                
                # Update coverage count
                coverage_count[cov_words] -= 1
                current_covered[cov_words[coverage_count[cov_words] == 0]] = False
                
                new_size = len(new_code)
                
                # Check if still valid
                if current_covered.all():
                    if new_size < best_size:
                        best_size = new_size
                        best_code = list(new_code)
                        best_covered = current_covered.copy()
                        best_coverage_count = coverage_count.copy()
                    
                    current_code = new_code
                    current_size = new_size
                else:
                    # Revert
                    current_covered[cov_words] = True
                    coverage_count[cov_words] += 1
        
        # 2. Try addition: pick a random uncovered word, add it
        elif rng.random() < 0.3:
            # Find uncovered words
            uncovered = np.where(~current_covered)[0]
            if len(uncovered) > 0:
                c_idx = uncovered[rng.randint(0, len(uncovered) - 1)]
                
                # Add this codeword
                current_code.append(c_idx)
                coverage_count[cover_matrix[c_idx]] += 1
                new_covered = current_covered.copy()
                new_covered[cover_matrix[c_idx]] = True
                
                new_size = len(current_code)
                
                if new_covered.all():
                    if new_size < best_size:
                        best_size = new_size
                        best_code = list(current_code)
                        best_covered = new_covered.copy()
                        best_coverage_count = coverage_count.copy()
                    
                    current_covered = new_covered
                    current_size = new_size
                else:
                    # Revert
                    coverage_count[cover_matrix[c_idx]] -= 1
                    current_code.pop()
        
        # 3. Try swap: remove one, add another
        else:
            if len(current_code) > 1:
                # Pick a random codeword to remove
                idx_pos = rng.randint(0, len(current_code) - 1)
                c_idx = current_code[idx_pos]
                cov_words = cover_matrix[c_idx]
                
                # Find words that would become uncovered if we remove c_idx
                would_uncover = cov_words[coverage_count[cov_words] == 1]
                
                if len(would_uncover) > 0:
                    # Pick a random word that would be uncovered, find a codeword to cover it
                    target_word = would_uncover[rng.randint(0, len(would_uncover) - 1)]
                    
                    # Find codewords that cover target_word but are not already in the code
                    # This is expensive, so use a simpler heuristic:
                    # Pick a random word within distance 1 of target_word
                    new_idx = target_word
                    new_cov_words = cover_matrix[new_idx]
                    
                    # Check if adding new_idx and removing c_idx keeps coverage valid
                    # Words covered by c_idx: cov_words
                    # Words covered by new_idx: new_cov_words
                    # Words uniquely covered by c_idx: would_uncover
                    # Words that new_idx covers that are not covered by others:
                    #   new_cov_words where coverage_count (without c_idx) == 0
                    
                    # Temporarily remove c_idx
                    coverage_count[cov_words] -= 1
                    
                    # Check if new_idx covers all would_uncover words
                    new_idx_set = set(new_cov_words.tolist())
                    all_covered = all(w in new_idx_set for w in would_uncover.tolist())
                    
                    if all_covered:
                        # Check if adding new_idx doesn't break anything
                        # It only adds coverage, so it's safe
                        current_code[idx_pos] = new_idx
                        coverage_count[new_cov_words] += 1
                        
                        new_size = len(current_code)
                        # Coverage is still valid since we covered would_uncover
                        if new_size < best_size:
                            best_size = new_size
                            best_code = list(current_code)
                            best_covered = current_covered.copy()
                            best_coverage_count = coverage_count.copy()
                    else:
                        # Revert
                        coverage_count[cov_words] += 1
        
        T *= cooling_rate
    
    # Return best code found
    result = [all_words[i].tolist() for i in best_code]
    return result

# EVOLVE-BLOCK-END
