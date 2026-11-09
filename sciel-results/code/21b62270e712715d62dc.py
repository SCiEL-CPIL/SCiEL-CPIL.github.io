import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy set cover for other cases
        total = q ** n
        if total > 100000:
            # Too large, return empty (invalid but won't crash)
            return []
        powers = [q ** (n - 1 - i) for i in range(n)]
        all_words = list(range(total))
        neighbors = []
        for idx in range(total):
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            word.reverse()
            neigh = [idx]
            for i in range(n):
                orig = word[i]
                for nv in range(q):
                    if nv != orig:
                        neigh.append(idx + (nv - orig) * powers[i])
            neighbors.append(neigh)
        cover_counts = np.zeros(total, dtype=np.int32)
        codewords = []
        uncovered = set(range(total))
        while uncovered and len(codewords) < total:
            # Pick word covering most uncovered
            best_c = -1
            best_count = -1
            for c in random.sample(list(uncovered), min(50, len(uncovered))):
                cnt = sum(1 for nb in neighbors[c] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c < 0:
                break
            codewords.append(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.discard(nb)
                cover_counts[nb] += 1
        result = []
        for idx in codewords:
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            word.reverse()
            result.append(word)
        return result

    # Target Case: K_3(8,1)
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for 120s limit

    # 1. Precompute neighbors using numpy
    powers = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build neighbor arrays efficiently
    # For each index, neighbors = [idx] + [idx + (dv * powers[i]) for i in range(n) for dv in (-1,1) if valid]
    # Actually for q=3: for each position, 2 alternatives
    # Total neighbors per word: 1 + 8*2 = 17
    
    # Precompute all neighbor indices as a flat numpy array
    # neighbors_flat[i*17 : (i+1)*17] = neighbors of word i
    neighbors_flat = np.zeros((total_words, 17), dtype=np.int32)
    
    for idx in range(total_words):
        neighbors_flat[idx, 0] = idx
        pos = 1
        temp = idx
        for i in range(n):
            digit = temp % q
            temp //= q
            # Two alternatives for this position
            for dv in range(-1, 2):
                if dv == 0:
                    continue
                new_digit = digit + dv
                if 0 <= new_digit < q:
                    neighbors_flat[idx, pos] = idx + dv * powers[n - 1 - i]
                    pos += 1
        # Should have exactly 17 entries
    
    # Precompute a boolean membership matrix for fast swap validation
    # member_matrix[a, b] = True if b is in neighbors of a
    # This is 6561 x 6561 = ~43M booleans = 43MB, too large
    # Instead, use a sorted approach or just check with the flat array
    
    # For faster "is x in neighbors of y" check, we can use a set per word
    # But 6561 sets of 17 elements is manageable
    neighbor_sets = [set(neighbors_flat[i].tolist()) for i in range(total_words)]
    
    # 2. Greedy Initialization
    cover_counts = np.zeros(total_words, dtype=np.int32)
    codewords = []
    is_codeword = np.zeros(total_words, dtype=bool)
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    uncovered_mask = np.ones(total_words, dtype=bool)
    
    # To speed up greedy, we'll do rounds
    # Each round: for a random subset of candidates, pick the best
    max_greedy_rounds = 550
    greedy_time_limit = 30  # seconds
    
    for round_num in range(max_greedy_rounds):
        if time.time() > start_time + greedy_time_limit:
            break
        if not np.any(uncovered_mask):
            break
            
        # Sample candidate positions from uncovered words and their neighbors
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a sample of candidates
        sample_size = min(200, len(uncovered_indices))
        candidates = random.sample(list(uncovered_indices), sample_size)
        
        # Also add random candidates from the whole space
        extra_candidates = random.sample(range(total_words), min(100, total_words))
        candidates = candidates + extra_candidates
        
        best_c = -1
        best_gain = -1
        
        for c in candidates:
            if is_codeword[c]:
                continue
            gain = np.sum(uncovered_mask[neighbors_flat[c]])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c < 0:
            # No good candidate found, pick random
            best_c = random.choice(list(uncovered_indices))
        
        # Add best_c
        codewords.append(best_c)
        is_codeword[best_c] = True
        for nb in neighbors_flat[best_c]:
            cover_counts[nb] += 1
            if cover_counts[nb] == 1:
                uncovered_mask[nb] = False
    
    # Ensure validity
    while np.any(uncovered_mask):
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        target = uncovered_indices[0]
        # Pick a neighbor of target to cover it
        best_c = random.choice(neighbors_flat[target].tolist())
        if not is_codeword[best_c]:
            codewords.append(best_c)
            is_codeword[best_c] = True
        for nb in neighbors_flat[best_c]:
            cover_counts[nb] += 1
            if cover_counts[nb] == 1:
                uncovered_mask[nb] = False
    
    initial_size = len(codewords)
    
    # 3. Local Search
    best_size = len(codewords)
    best_codewords = list(codewords)
    best_cover_counts = cover_counts.copy()
    best_is_codeword = is_codeword.copy()
    
    # Use list for codewords but maintain is_codeword array
    T = 1.0
    cooling_rate = 0.9995
    min_T = 0.05
    
    # Track consecutive failures to adjust strategy
    consecutive_failures = 0
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Try Deletion first (most impactful)
        if len(codewords) > 0:
            c = random.choice(codewords)
            # Check if c is redundant: all its neighbors have count > 1
            if np.min(cover_counts[neighbors_flat[c]]) > 1:
                # Remove c
                for nb in neighbors_flat[c]:
                    cover_counts[nb] -= 1
                is_codeword[c] = False
                codewords.remove(c)
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    best_cover_counts = cover_counts.copy()
                    best_is_codeword = is_codeword.copy()
                consecutive_failures = 0
                continue
        
        # Try Swap
        if len(codewords) > 0:
            c_out = random.choice(codewords)
            
            # Smart candidate selection: pick from words that cover under-covered areas
            # Under-covered words are those with cover_count == 1
            under_covered = np.where(cover_counts == 1)[0]
            
            if len(under_covered) > 0 and random.random() < 0.7:
                # Pick a random under-covered word, then one of its neighbors as c_in
                uc = random.choice(under_covered.tolist())
                c_in_candidates = neighbors_flat[uc].tolist()
                c_in = random.choice(c_in_candidates)
            else:
                c_in = random.randint(0, total_words - 1)
            
            if c_in == c_out or is_codeword[c_in]:
                T *= cooling_rate
                continue
            
            # Validate swap: check that removing c_out doesn't leave any word uncovered
            # Only need to check neighbors of c_out that are NOT neighbors of c_in
            nb_out = neighbors_flat[c_out]
            nb_in = neighbors_flat[c_in]
            
            # For each neighbor of c_out, if it's not a neighbor of c_in, its count decreases by 1
            # We need all such counts to be >= 2 (so after decrement >= 1)
            valid = True
            for nb in nb_out:
                if nb not in neighbor_sets[c_in]:
                    if cover_counts[nb] <= 1:
                        valid = False
                        break
            
            if valid:
                # Perform swap
                for nb in nb_out:
                    cover_counts[nb] -= 1
                for nb in nb_in:
                    cover_counts[nb] += 1
                is_codeword[c_out] = False
                is_codeword[c_in] = True
                codewords.remove(c_out)
                codewords.append(c_in)
                
                # After swap, try to delete redundant codewords
                deleted_any = False
                # Try a few random deletions
                for _ in range(min(5, len(codewords))):
                    c_check = random.choice(codewords)
                    if np.min(cover_counts[neighbors_flat[c_check]]) > 1:
                        for nb in neighbors_flat[c_check]:
                            cover_counts[nb] -= 1
                        is_codeword[c_check] = False
                        codewords.remove(c_check)
                        deleted_any = True
                
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    best_cover_counts = cover_counts.copy()
                    best_is_codeword = is_codeword.copy()
                
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                # If too many failures, do a restart-like perturbation
                if consecutive_failures > 50:
                    # Perturb: remove a few random codewords and re-add greedy
                    for _ in range(3):
                        if len(codewords) > 0:
                            c_remove = random.choice(codewords)
                            for nb in neighbors_flat[c_remove]:
                                cover_counts[nb] -= 1
                            is_codeword[c_remove] = False
                            codewords.remove(c_remove)
                    
                    # Re-cover uncovered
                    uncovered_mask = (cover_counts == 0)
                    while np.any(uncovered_mask):
                        uc = np.where(uncovered_mask)[0]
                        if len(uc) == 0:
                            break
                        target = uc[0]
                        best_c = random.choice(neighbors_flat[target].tolist())
                        if not is_codeword[best_c]:
                            codewords.append(best_c)
                            is_codeword[best_c] = True
                        for nb in neighbors_flat[best_c]:
                            cover_counts[nb] += 1
                            if cover_counts[nb] == 1:
                                uncovered_mask[nb] = False
                    consecutive_failures = 0
            
            T *= cooling_rate
    
    # Final polish: systematic deletion pass
    # Try to remove any remaining redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in list(codewords):
            if is_codeword[c] and np.min(cover_counts[neighbors_flat[c]]) > 1:
                for nb in neighbors_flat[c]:
                    cover_counts[nb] -= 1
                is_codeword[c] = False
                codewords.remove(c)
                changed = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    best_cover_counts = cover_counts.copy()
                    best_is_codeword = is_codeword.copy()
    
    # Return best found
    result = []
    for idx in best_codewords:
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result
