"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Helper to convert indices to codewords
        def indices_to_codewords(indices):
            return [[(idx // powers[p]) % 3 for p in range(n)] for idx in indices]

        # Local Search: Remove redundant codewords and perform swaps
        def local_search_remove_and_swap(indices):
            if not indices:
                return indices
            
            indices_set = set(indices)
            
            # Calculate initial coverage counts
            coverage_counts = np.zeros(total, dtype=np.int32)
            for idx in indices:
                coverage_counts[neighbors[idx]] += 1
            
            improved = True
            while improved:
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # 1. Try to remove redundant codewords
                for i, idx in enumerate(indices):
                    if np.min(coverage_counts[neighbors[idx]]) >= 2:
                        # Remove it
                        indices.pop(i)
                        indices_set.remove(idx)
                        coverage_counts[neighbors[idx]] -= 1
                        improved = True
                        break # Restart loop with new set
                
                if improved:
                    continue

                # 2. Try to swap a codeword with a non-codeword
                # Iterate through codewords and try a few random replacements
                for i in range(len(indices)):
                    if time.time() > deadline - 2:
                        break
                    
                    c = indices[i]
                    # Try a few random non-codewords
                    for _ in range(10): 
                        w = rng.randint(0, total - 1)
                        if w in indices_set:
                            continue
                        
                        # Check if swap c -> w is valid
                        # Words that lose coverage: N(c) \ N(w)
                        # We need coverage_counts[v] >= 2 for all such v
                        
                        set_w = set(neighbors[w])
                        valid = True
                        for v in neighbors[c]:
                            if v not in set_w:
                                if coverage_counts[v] < 2:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        # Perform Swap
                        # Update coverage
                        # Losing: N(c) \ N(w)
                        # Gaining: N(w) \ N(c)
                        
                        set_c = set(neighbors[c])
                        for v in neighbors[c]:
                            if v not in set_w:
                                coverage_counts[v] -= 1
                        for v in neighbors[w]:
                            if v not in set_c:
                                coverage_counts[v] += 1
                        
                        # Update indices
                        indices[i] = w
                        indices_set.remove(c)
                        indices_set.add(w)
                        
                        improved = True
                        break # Restart removal/swap loop
                    if improved:
                        break
            
            return indices

        # Greedy Set Cover with random tie-breaking
        def greedy_cover(rng):
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []
            
            while uncovered.sum() > 0:
                if time.time() > deadline - 5:
                    break
                
                # Find candidate that covers most uncovered words
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                
                if max_count == 0:
                    break
                
                # Find all candidates with max_count
                candidates = np.where(counts == max_count)[0]
                
                # Random choice among candidates
                best_candidate = rng.choice(candidates)
                
                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0
            
            return codewords

        # Add-then-remove heuristic: start from a superset, then remove redundancies
        def add_then_remove_heuristic(rng):
            # Start with a random subset of size ~80-90
            # Or start with direct sum + random extras?
            # Better: start with a slightly larger random set and prune
            # Or: start with greedy result, add some random, then prune
            # Let's try: take greedy result, add 5-10 random words, then prune
            base = greedy_cover(rng)
            if not base:
                return base
            
            # Add some random words
            current = list(base)
            current_set = set(current)
            
            num_add = rng.randint(3, 8)
            for _ in range(num_add):
                w = rng.randint(0, total - 1)
                if w not in current_set:
                    current.append(w)
                    current_set.add(w)
            
            # Now prune: remove redundant words
            # This is similar to local_search but we start from a larger set
            # and aggressively remove
            indices = current
            indices_set = set(indices)
            
            coverage_counts = np.zeros(total, dtype=np.int32)
            for idx in indices:
                coverage_counts[neighbors[idx]] += 1
            
            improved = True
            while improved:
                if time.time() > deadline - 2:
                    break
                improved = False
                
                # Try to remove redundant codewords
                for i, idx in enumerate(indices):
                    if np.min(coverage_counts[neighbors[idx]]) >= 2:
                        indices.pop(i)
                        indices_set.remove(idx)
                        coverage_counts[neighbors[idx]] -= 1
                        improved = True
                        break
                
                if improved:
                    continue
                
                # If no more removals, try swaps to enable further removals
                for i in range(len(indices)):
                    if time.time() > deadline - 2:
                        break
                    
                    c = indices[i]
                    for _ in range(15):
                        w = rng.randint(0, total - 1)
                        if w in indices_set:
                            continue
                        
                        set_w = set(neighbors[w])
                        valid = True
                        for v in neighbors[c]:
                            if v not in set_w:
                                if coverage_counts[v] < 2:
                                    valid = False
                                    break
                        
                        if not valid:
                            continue
                        
                        set_c = set(neighbors[c])
                        for v in neighbors[c]:
                            if v not in set_w:
                                coverage_counts[v] -= 1
                        for v in neighbors[w]:
                            if v not in set_c:
                                coverage_counts[v] += 1
                        
                        indices[i] = w
                        indices_set.remove(c)
                        indices_set.add(w)
                        
                        improved = True
                        break
                    if improved:
                        break
            
            return indices

        # Initial best is Direct Sum
        ds_indices = []
        ds_codewords = get_direct_sum()
        # Convert DS codewords to indices
        def codeword_to_index(cw):
            idx = 0
            for p in range(n):
                idx += cw[p] * powers[p]
            return idx

        best_indices = [codeword_to_index(cw) for cw in ds_codewords]
        best_size = len(best_indices)
        
        # Run multiple restarts with different strategies
        rng = random.Random(42)
        
        max_restarts = 1000
        for restart_num in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Alternate between pure greedy and add-then-remove
            if restart_num % 3 == 0:
                cand_indices = add_then_remove_heuristic(rng)
            else:
                cand_indices = greedy_cover(rng)
            
            if len(cand_indices) >= best_size:
                if len(cand_indices) < best_size + 10:
                    cand_indices = local_search_remove_and_swap(cand_indices)
                    if len(cand_indices) < best_size:
                        best_size = len(cand_indices)
                        best_indices = cand_indices
            else:
                cand_indices = local_search_remove_and_swap(cand_indices)
                if len(cand_indices) < best_size:
                    best_size = len(cand_indices)
                    best_indices = cand_indices
            
            # Early exit if optimal found
            if best_size == 73:
                break

        # Convert best indices to codewords
        final_codewords = indices_to_codewords(best_indices)
        
        return final_codewords

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
