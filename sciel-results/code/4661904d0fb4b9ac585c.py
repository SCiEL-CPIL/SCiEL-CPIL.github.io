import random
import time
import sys

def construct(q, n, radius):
    # Parameters
    total_words = q ** n
    # Map words to indices for faster array access
    # word -> index, index -> word
    # For q=3, n=7, total_words=2187.
    
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Pre-compute neighbors (ball of radius 1) for all words
    # neighbors[idx] = list of indices within distance <= 1
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                # Calculate new index
                # idx = ... + w[i]*q^(n-1-i) + ...
                # delta_idx = (new_val - w[i]) * q^(n-1-i)
                pow_q = q ** (n - 1 - i)
                new_idx = idx + (new_val - w[i]) * pow_q
                neigh.append(new_idx)
        neighbors[idx] = neigh

    # 1. Greedy Construction to get an initial valid cover
    uncovered = [1] * total_words # 1 means uncovered
    code_indices = []
    
    # To speed up greedy, we can just iterate.
    # Optimization: Keep a list of uncovered words? 
    # Simpler: Just iterate all words, pick best.
    # Since 2187 is small, this is fast.
    
    remaining = set(range(total_words))
    
    while remaining:
        best_idx = -1
        max_cov = 0
        # Sample or iterate all? Iterating all 2187 is fine.
        # But we only need to consider words that are still in 'remaining'? 
        # Actually, adding a word that is already covered might cover others.
        # Greedy usually picks from the whole space or uncovered.
        # Picking from uncovered is a standard heuristic.
        # Let's pick from all words to be safe, but prioritize uncovered.
        
        # Optimization: Only consider words in 'remaining' for candidates?
        # No, a covered word might cover many uncovered ones.
        # However, standard greedy for set cover picks the set with max marginal gain.
        
        # To make it faster, we can just iterate all indices.
        for idx in range(total_words):
            cov = 0
            for nb in neighbors[idx]:
                if uncovered[nb]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_idx = idx
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        
        # Update uncovered
        for nb in neighbors[best_idx]:
            uncovered[nb] = 0
            if uncovered[nb] == 0:
                remaining.discard(nb)
                
    # 2. Local Search (Greedy Removal and Swap)
    # Maintain coverage counts
    coverage = [0] * total_words
    for idx in code_indices:
        for nb in neighbors[idx]:
            coverage[nb] += 1
            
    # Check validity: min(coverage) >= 1
    # We know it's valid from construction.
    
    improved = True
    # Limit iterations to avoid timeout, but 120s is generous.
    # 2187 words, ~200 codewords.
    # Removal check: O(n * q * |code|) roughly? No, O(|neighbors|) per codeword.
    # |neighbors| = 1 + n*(q-1) = 1 + 7*2 = 15.
    # So removing one codeword takes ~15 ops to update counts and check min.
    # Checking min over 2187 words takes 2187 ops.
    # Total per removal attempt: ~2200 ops.
    # If we try all |code| ~ 200 codewords, that's 440,000 ops.
    # This is very fast.
    
    while improved:
        improved = False
        
        # Phase 1: Greedy Removal
        # Try to remove any codeword that doesn't break coverage.
        # We iterate through the current code list.
        # We must be careful with indices if we remove.
        # Let's collect removable indices first.
        removable = []
        
        # To make it deterministic or random? Random helps escape.
        # Let's shuffle the code order to avoid bias.
        random.shuffle(code_indices)
        
        for i in range(len(code_indices)):
            idx = code_indices[i]
            # Tentatively remove
            for nb in neighbors[idx]:
                coverage[nb] -= 1
            
            # Check if min coverage is still >= 1
            # We can track a count of zeros to avoid full scan?
            # Or just scan. 2187 is small.
            min_cov = min(coverage)
            
            if min_cov >= 1:
                # Removal is valid
                # Do not restore
                del code_indices[i]
                improved = True
                # Since we deleted an element, indices shift.
                # It's safer to break and restart the loop or handle carefully.
                # Given the small size, breaking and restarting the while loop is clean.
                break
            else:
                # Restore
                for nb in neighbors[idx]:
                    coverage[nb] += 1
        
        # Phase 2: Swaps
        # If no removal was possible, try swaps.
        if not improved:
            # Try to swap a codeword c with another word w
            # such that the new set is valid.
            # We want to explore the space.
            # Heuristic: Try to swap a codeword that has "low" uniqueness?
            # Or just random swaps.
            
            # To be efficient, we can try swapping each codeword with a random subset of words.
            # Or iterate all words? 200 * 2187 = 437k swaps.
            # Each swap check: update counts (30 ops), check min (2187 ops).
            # Total ~ 100 million ops. Might be too slow for Python in 120s if done many times.
            
            # Optimization: Only try swaps if we can find a w that covers some "critical" words better?
            # Simpler: Random restarts or limited swaps.
            
            # Let's try a limited number of random swaps per iteration.
            # Or iterate through codewords and pick a random candidate w.
            
            for i in range(len(code_indices)):
                c = code_indices[i]
                # Pick a few random candidates
                for _ in range(10): # 10 random swaps per codeword per pass
                    w = random.randint(0, total_words - 1)
                    if w == c:
                        continue
                    
                    # Tentative swap
                    # Remove c
                    for nb in neighbors[c]:
                        coverage[nb] -= 1
                    # Add w
                    for nb in neighbors[w]:
                        coverage[nb] += 1
                    
                    if min(coverage) >= 1:
                        # Valid swap
                        code_indices[i] = w
                        improved = True
                        # Restart removal phase?
                        # For now, mark improved and break out to restart while loop
                        # This allows us to try removals again with the new configuration.
                        break
                    else:
                        # Restore
                        for nb in neighbors[w]:
                            coverage[nb] -= 1
                        for nb in neighbors[c]:
                            coverage[nb] += 1
                
                if improved:
                    break

    # Convert indices back to list of lists
    result = []
    for idx in code_indices:
        result.append(idx_to_word(idx))
        
    return result
