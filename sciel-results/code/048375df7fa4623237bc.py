"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import minimize
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    For the specific case q=3, n=6, radius=1, we use a highly optimized local search
    with efficient coverage counting and aggressive reduction strategies.
    
    Strategy:
    1. Start with a valid covering code (tetracode-based: 81 words).
    2. Use efficient local search with vectorized coverage counting.
    3. Try removals first, then swaps, with multiple restarts if needed.
    4. Use numpy for fast set operations via bitmasks.
    """
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1()
    
    # Generic fallback for other parameters
    return _generic_greedy(q, n, radius)

def _construct_3_6_1():
    """Optimized construction for q=3, n=6, radius=1."""
    total_words = 729
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    
    # Precompute neighbors as index lists
    neighbors_idx = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        nb_set = {i}
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                nb_set.add(word_to_idx[new_w])
        neighbors_idx[i] = list(nb_set)
    
    # Convert to numpy arrays for fast operations
    neighbors_array = np.zeros((total_words, total_words), dtype=np.uint8)
    for i, nbs in enumerate(neighbors_idx):
        neighbors_array[i, nbs] = 1
    
    # Start with tetracode-based construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = tuple(h + list(tail))
            initial_indices.append(word_to_idx[word])
    
    # Use numpy for coverage tracking
    code_mask = np.zeros(total_words, dtype=np.uint8)
    for idx in initial_indices:
        code_mask[idx] = 1
    
    # Coverage: for each word, count how many codewords cover it
    coverage_count = neighbors_array @ code_mask  # shape (729,)
    
    def is_valid_cover():
        return np.all(coverage_count > 0)
    
    # Local search: try to remove codewords
    # Sort codewords by "criticality" - words that are only covered by few codewords
    # Try removing codewords that are least critical first
    
    code_indices = set(initial_indices)
    
    # Phase 1: Remove codewords if possible
    improved = True
    while improved:
        improved = False
        # Get list of current codewords
        cw_list = list(code_indices)
        # Sort by coverage contribution (fewer unique words covered = better candidate for removal)
        # A word is "removable" if all words it covers are also covered by other codewords
        for idx in cw_list:
            # Check if removing this codeword maintains coverage
            # Words uniquely covered by this codeword: coverage_count[w] == 1 and w in neighbors[idx]
            # If no such words exist, we can remove
            nbs = neighbors_idx[idx]
            unique = 0
            for w in nbs:
                if coverage_count[w] == 1:
                    unique += 1
                    break
            if unique == 0:
                # Remove this codeword
                code_indices.remove(idx)
                code_mask[idx] = 0
                # Update coverage counts
                for w in nbs:
                    coverage_count[w] -= 1
                improved = True
                break  # Restart since coverage changed
    
    # Phase 2: Try swaps - replace one codeword with another
    # This is more expensive but can find better solutions
    # We'll do limited swaps to stay within time budget
    
    start_time = time.time()
    time_budget = 100  # seconds
    
    # Try multiple swap rounds
    for swap_round in range(50):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        cw_list = list(code_indices)
        
        for idx_remove in cw_list:
            if time.time() - start_time > time_budget:
                break
            
            # Find words that would be uncovered if we remove idx_remove
            # These are words where coverage_count[w] == 1 and w in neighbors[idx_remove]
            critical_words = []
            for w in neighbors_idx[idx_remove]:
                if coverage_count[w] == 1:
                    critical_words.append(w)
            
            if not critical_words:
                # Can just remove
                code_indices.remove(idx_remove)
                code_mask[idx_remove] = 0
                for w in neighbors_idx[idx_remove]:
                    coverage_count[w] -= 1
                improved = True
                break
            
            # Need to find a replacement that covers all critical_words
            # Try candidate words
            best_candidate = None
            best_score = -1
            
            # Only consider words that cover at least one critical word
            for idx_add in range(total_words):
                if idx_add in code_indices:
                    continue
                
                # Check if idx_add covers all critical words
                covers_all = all(w in neighbors_idx[idx_add] for w in critical_words)
                if not covers_all:
                    continue
                
                # Score: how many additional words does idx_add cover that are currently well-covered?
                # We want to minimize disruption
                # Count words where coverage_count[w] > 1 that idx_add covers
                extra_cover = 0
                for w in neighbors_idx[idx_add]:
                    if w not in critical_words and coverage_count[w] > 1:
                        extra_cover += 1
                
                # Prefer candidates with less extra cover (less disruption)
                # But also prefer candidates that are not already in code
                if extra_cover < best_score or best_candidate is None:
                    best_score = extra_cover
                    best_candidate = idx_add
            
            if best_candidate is not None:
                # Perform the swap
                code_indices.remove(idx_remove)
                code_indices.add(best_candidate)
                code_mask[idx_remove] = 0
                code_mask[best_candidate] = 1
                
                # Update coverage
                for w in neighbors_idx[idx_remove]:
                    coverage_count[w] -= 1
                for w in neighbors_idx[best_candidate]:
                    coverage_count[w] += 1
                
                improved = True
                break
        
        if not improved:
            # Try a more aggressive approach: random restarts or 2-swap
            # For now, try 2-swap: remove two, add two
            two_swap_improved = False
            cw_list = list(code_indices)
            for i in range(len(cw_list)):
                if time.time() - start_time > time_budget:
                    break
                for j in range(i + 1, len(cw_list)):
                    if time.time() - start_time > time_budget:
                        break
                    
                    idx1, idx2 = cw_list[i], cw_list[j]
                    
                    # Find critical words when removing both
                    temp_cov = coverage_count.copy()
                    for w in neighbors_idx[idx1]:
                        temp_cov[w] -= 1
                    for w in neighbors_idx[idx2]:
                        temp_cov[w] -= 1
                    
                    critical_words = [w for w in range(total_words) if temp_cov[w] == 0]
                    
                    if not critical_words:
                        # Can just remove both
                        code_indices.discard(idx1)
                        code_indices.discard(idx2)
                        code_mask[idx1] = 0
                        code_mask[idx2] = 0
                        for w in neighbors_idx[idx1]:
                            coverage_count[w] -= 1
                        for w in neighbors_idx[idx2]:
                            coverage_count[w] -= 1
                        two_swap_improved = True
                        break
                    
                    # Need to find two replacements
                    # This is expensive, so skip for now in most cases
                    if len(critical_words) <= 4:
                        # Try to find two words that cover all critical words
                        found_pair = False
                        for idx_add1 in range(total_words):
                            if idx_add1 in code_indices or idx_add1 == idx1 or idx_add1 == idx2:
                                continue
                            # Check if idx_add1 covers some critical words
                            covers1 = [w for w in critical_words if w in neighbors_idx[idx_add1]]
                            if not covers1:
                                continue
                            remaining = [w for w in critical_words if w not in covers1]
                            if not remaining:
                                # Just need one more word
                                for idx_add2 in range(total_words):
                                    if idx_add2 in code_indices or idx_add2 == idx1 or idx_add2 == idx2 or idx_add2 == idx_add1:
                                        continue
                                    covers2 = all(w in neighbors_idx[idx_add2] for w in remaining)
                                    if covers2:
                                        # Perform 2-swap
                                        code_indices.discard(idx1)
                                        code_indices.discard(idx2)
                                        code_indices.add(idx_add1)
                                        code_indices.add(idx_add2)
                                        code_mask[idx1] = 0
                                        code_mask[idx2] = 0
                                        code_mask[idx_add1] = 1
                                        code_mask[idx_add2] = 1
                                        
                                        for w in neighbors_idx[idx1]:
                                            coverage_count[w] -= 1
                                        for w in neighbors_idx[idx2]:
                                            coverage_count[w] -= 1
                                        for w in neighbors_idx[idx_add1]:
                                            coverage_count[w] += 1
                                        for w in neighbors_idx[idx_add2]:
                                            coverage_count[w] += 1
                                        
                                        two_swap_improved = True
                                        found_pair = True
                                        break
                            else:
                                # Need idx_add2 to cover remaining
                                for idx_add2 in range(total_words):
                                    if idx_add2 in code_indices or idx_add2 == idx1 or idx_add2 == idx2 or idx_add2 == idx_add1:
                                        continue
                                    covers2 = all(w in neighbors_idx[idx_add2] for w in remaining)
                                    if covers2:
                                        # Perform 2-swap
                                        code_indices.discard(idx1)
                                        code_indices.discard(idx2)
                                        code_indices.add(idx_add1)
                                        code_indices.add(idx_add2)
                                        code_mask[idx1] = 0
                                        code_mask[idx2] = 0
                                        code_mask[idx_add1] = 1
                                        code_mask[idx_add2] = 1
                                        
                                        for w in neighbors_idx[idx1]:
                                            coverage_count[w] -= 1
                                        for w in neighbors_idx[idx2]:
                                            coverage_count[w] -= 1
                                        for w in neighbors_idx[idx_add1]:
                                            coverage_count[w] += 1
                                        for w in neighbors_idx[idx_add2]:
                                            coverage_count[w] += 1
                                        
                                        two_swap_improved = True
                                        found_pair = True
                                        break
                            if found_pair:
                                break
                        if found_pair:
                            break
                    
                    if two_swap_improved:
                        break
                if two_swap_improved:
                    break
            
            if two_swap_improved:
                continue
            else:
                break
    
    # Final validation
    if not is_valid_cover():
        # Fallback: return a known valid solution
        # Reconstruct from initial
        code_indices = set(initial_indices)
        for idx in initial_indices:
            code_mask[idx] = 1
        coverage_count = neighbors_array @ code_mask
    
    result = [list(idx_to_word[idx]) for idx in code_indices]
    return result

def _generic_greedy(q, n, radius):
    """Simple greedy algorithm for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
