import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)
    
    # For the specific case q=3, n=7, radius=1, we use the full space as candidates
    candidates = all_words
    num_candidates = len(candidates)
    
    # Precompute coverage matrix
    # coverage[i, j] = True if candidates[i] covers all_words[j] (dist <= radius)
    coverage = np.zeros((num_candidates, total_words), dtype=bool)
    
    # Chunked computation
    chunk_size = 256
    for i_start in range(0, num_candidates, chunk_size):
        i_end = min(i_start + chunk_size, num_candidates)
        diff = candidates[i_start:i_end, None, :].astype(np.int8) != all_words[None, :, :].astype(np.int8)
        dists = np.sum(diff, axis=2, dtype=np.int16)
        coverage[i_start:i_end, :] = dists <= radius
        
    # Precompute cover_counts for each candidate (how many words it covers)
    # This is just the sum of each row in coverage
    candidate_cover_counts = np.sum(coverage, axis=1, dtype=np.int32)

    # Precompute for each word, which candidates cover it (transpose of coverage)
    # This is useful for local search
    # cover_of_word[w] = list of candidates i such that coverage[i, w]
    # Since coverage is boolean, we can use np.where on columns, but it's slow.
    # Better to keep coverage as is and use matrix operations.

    def greedy_set_cover(coverage, candidate_cover_counts, deadline):
        remaining = np.ones(coverage.shape[1], dtype=np.int8)
        remaining_count = coverage.shape[1]
        selected_indices = []
        
        while remaining_count > 0:
            if time.time() > deadline - 20.0:
                break
            # counts[i] = sum of coverage[i, j] where remaining[j] == 1
            counts = coverage.dot(remaining)
            best_idx = np.argmax(counts)
            best_count = counts[best_idx]
            if best_count == 0:
                break
            selected_indices.append(best_idx)
            remaining[coverage[best_idx]] = 0
            remaining_count = np.count_nonzero(remaining)
            
        return selected_indices

    def local_search(selected_indices, coverage, candidate_cover_counts, deadline):
        num_candidates = coverage.shape[0]
        total_words = coverage.shape[1]
        
        in_set = np.zeros(num_candidates, dtype=bool)
        in_set[selected_indices] = True
        
        cover_counts = np.zeros(total_words, dtype=np.int16)
        for idx in selected_indices:
            cover_counts[coverage[idx]] += 1
            
        best_size = len(selected_indices)
        best_in_set = in_set.copy()
        
        # Precompute for each candidate, the list of words it covers
        # This is expensive to store as lists, so we stick to boolean arrays or bitsets.
        # For n=7, q=3, total_words=2187.
        # We can use a list of numpy arrays for the rows of coverage that are True.
        # Or just use the coverage matrix directly.
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Phase 1: Remove redundant codewords
            to_remove = []
            for i in range(num_candidates):
                if not in_set[i]:
                    continue
                # Check if i covers any word with cover_count == 1
                # Vectorized check
                if not np.any(coverage[i] & (cover_counts == 1)):
                    to_remove.append(i)
                    
            if to_remove:
                for i in to_remove:
                    in_set[i] = False
                    cover_counts[coverage[i]] -= 1
                current_size = np.sum(in_set)
                if current_size < best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                improved = True
                continue # Restart loop as cover_counts changed significantly
                
            # Phase 2: Try swaps
            # Identify critical codewords: those covering at least one word with cover_count == 1
            critical_indices = []
            for i in range(num_candidates):
                if in_set[i]:
                    if np.any(coverage[i] & (cover_counts == 1)):
                        critical_indices.append(i)
                        
            if not critical_indices:
                break # Should not happen if Phase 1 didn't remove anything
                
            # Shuffle to avoid bias
            random.shuffle(critical_indices)
            
            swap_found = False
            for i in critical_indices:
                if time.time() > deadline - 5.0:
                    break
                    
                # Find essential words for i
                essential_words = np.where(coverage[i] & (cover_counts == 1))[0]
                
                if len(essential_words) == 0:
                    continue
                    
                # Find candidates j not in set that cover ALL essential words
                # Vectorized:
                # mask_j = coverage[:, essential_words].all(axis=1) & ~in_set
                potential_j = np.where(coverage[:, essential_words].all(axis=1) & ~in_set)[0]
                
                if len(potential_j) == 0:
                    continue
                    
                # Evaluate potential swaps
                # We want a swap that minimizes the number of new "essential" words created,
                # or maximizes the number of redundant codewords created.
                best_j = -1
                best_score = -np.inf
                
                # Sort potential_j by coverage count (descending) to prioritize good candidates
                cov_counts_potential = candidate_cover_counts[potential_j]
                sorted_idx = np.argsort(-cov_counts_potential)
                
                # Only evaluate top 50 candidates to save time
                for idx in sorted_idx[:50]:
                    j = potential_j[idx]
                    
                    # Simulate swap: remove i, add j
                    # new_cover_counts = cover_counts - coverage[i] + coverage[j]
                    new_cover_counts = cover_counts.copy()
                    new_cover_counts[coverage[i]] -= 1
                    new_cover_counts[coverage[j]] += 1
                    
                    # Check validity: all words must have new_cover_counts >= 1
                    # We already know j covers essential words of i, so those are safe.
                    # But we need to ensure no word drops to 0.
                    # Since j covers essential words, and other words covered by i had count >= 2,
                    # removing i makes them count >= 1. Adding j doesn't hurt.
                    # So the swap is valid if j covers essential words.
                    
                    # Heuristic: Score = - (number of words with new_cover_counts == 1)
                    # Fewer essential words is better.
                    num_essentials = np.sum(new_cover_counts == 1)
                    
                    # Also prefer j with high total coverage
                    score = -num_essentials + 0.1 * candidate_cover_counts[j]
                    
                    if score > best_score:
                        best_score = score
                        best_j = j
                        
                if best_j != -1:
                    # Perform swap
                    in_set[i] = False
                    in_set[best_j] = True
                    cover_counts[coverage[i]] -= 1
                    cover_counts[coverage[best_j]] += 1
                    current_size = np.sum(in_set)
                    if current_size < best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    improved = True
                    swap_found = True
                    break # Restart Phase 1 with new configuration
                    
            if not swap_found:
                # No simple swap found.
                # Try a more aggressive move: remove a critical codeword and add TWO new ones?
                # Or just accept the local minimum.
                break
                
        return best_in_set, best_size

    # Run greedy
    initial_solution = greedy_set_cover(coverage, candidate_cover_counts, deadline)
    
    # Run local search
    best_in_set, best_size = local_search(initial_solution, coverage, candidate_cover_counts, deadline)
    
    # Extract final codewords
    final_indices = np.where(best_in_set)[0]
    selected_codewords = [candidates[i].tolist() for i in final_indices]
    
    return selected_codewords
