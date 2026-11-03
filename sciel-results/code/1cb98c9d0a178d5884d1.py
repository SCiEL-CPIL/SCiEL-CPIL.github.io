import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 40s buffer for safety
    
    N = q ** n
    
    # Precompute powers for fast index conversion
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i-1] * q
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx += word[i] * q_powers[n-1-i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbor lists using arithmetic instead of word_to_idx
    neighbor_lists = [[] for _ in range(N)]
    neighbor_masks = [0] * N
    
    if radius == 1:
        for c_idx in range(N):
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            # For each position, compute the delta to add/subtract for each new value
            for i in range(n):
                pow_q = q_powers[n-1-i]
                orig_digit = (c_idx // pow_q) % q
                for new_val in range(q):
                    if new_val != orig_digit:
                        w_idx = c_idx + (new_val - orig_digit) * pow_q
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask
    else:
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            neighbor_lists[c_idx].append(c_idx)
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
            neighbor_masks[c_idx] = mask
    
    # Build reverse mapping using hash-based approach (O(N) instead of O(N*degree))
    # For radius-1, the codewords covering word w are: w itself and words at distance 1
    # We can compute this on-the-fly using the same arithmetic as neighbor generation
    def get_coders_for_word(w_idx):
        """Return list of codeword indices that cover w_idx (i.e., within radius of w_idx)"""
        coders = [w_idx]
        if radius == 1:
            for i in range(n):
                pow_q = q_powers[n-1-i]
                orig_digit = (w_idx // pow_q) % q
                for new_val in range(q):
                    if new_val != orig_digit:
                        c_idx = w_idx + (new_val - orig_digit) * pow_q
                        coders.append(c_idx)
        else:
            w_word = all_words[w_idx]
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == w_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = w_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        c_idx = word_to_idx(new_word)
                        coders.append(c_idx)
        return coders
    
    def greedy_cover_criticality(rng):
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # For radius-1 with q=3, n=8: criticality is always q*n+1 = 25 (constant)
        # So we just need to pick any uncovered word and find best codeword
        # Use a simple approach: track uncovered as a set for faster iteration
        uncovered_set = set(range(N))
        
        while uncovered_set:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered word (randomization helps escape bad choices)
            target_word = next(iter(uncovered_set))
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            coders = get_coders_for_word(target_word)
            for c_idx in coders:
                if c_idx in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                for c_idx in coders:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            # Update uncovered
            newly_covered = uncovered & neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
            # Remove newly covered words from the set
            temp = newly_covered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_set.discard(w_idx)
                temp ^= lsb
        
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords using vectorized coverage"""
        if not code:
            return code
            
        # Build coverage matrix: cover_matrix[c][w] = 1 if c covers w
        # Use neighbor_lists for sparse representation
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Check if any word is uniquely covered by c
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break
                    
        return code
    
    def two_opt_local_search(code, rng):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Shuffle order for randomization
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                c_remove = code[i]
                
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if not newly_uncovered:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break
                
                # Collect candidate codewords that cover all newly_uncovered words
                if len(newly_uncovered) > 3:
                    continue  # Too many to cover efficiently
                
                candidate_codewords = set()
                for w in newly_uncovered:
                    coders = get_coders_for_word(w)
                    for c in coders:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                # Limit candidates to avoid excessive computation
                if len(candidate_codewords) > 80:
                    candidates_list = list(candidate_codewords)
                    rng.shuffle(candidates_list)
                    candidate_codewords = set(candidates_list[:80])
                
                best_add = -1
                best_gain = -1
                
                for c_add in candidate_codewords:
                    # Check if c_add covers all newly_uncovered using bitmask
                    mask_add = neighbor_masks[c_add]
                    all_covered = True
                    for w in newly_uncovered:
                        if (mask_add >> w) & 1 == 0:
                            all_covered = False
                            break
                    if not all_covered:
                        continue
                    
                    # Count new coverage
                    new_coverage = 0
                    for w in neighbor_lists[c_add]:
                        if cover_counts[w] == 0:
                            new_coverage += 1
                    
                    if new_coverage > best_gain:
                        best_gain = new_coverage
                        best_add = c_add
                
                if best_add != -1:
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    
                    code.append(best_add)
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1
                    code_set.add(best_add)
                    
                    improved = True
                    break
        
        return code
    
    def perturb(code, rng, strength=0.05):
        """Randomly swap some codewords to escape local minima, targeting weakly-covered areas"""
        code = list(code)
        if not code:
            return code
        
        code_set = set(code)
        n_swaps = max(1, int(len(code) * strength))
        
        # Compute coverage to identify weakly-covered words
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        for _ in range(n_swaps):
            if len(code) < 2:
                break
            i = rng.randint(0, len(code) - 1)
            old = code[i]
            # Remove old codeword and pick a new one that covers weakly-covered words
            code_set.discard(old)
            
            # Find words that were covered by old and have low coverage
            weak_words = []
            for w in neighbor_lists[old]:
                if cover_counts[w] <= 2:
                    weak_words.append(w)
            
            if weak_words:
                # Pick a random weak word and find a codeword that covers it
                target_w = rng.choice(weak_words)
                candidates = [c for c in get_coders_for_word(target_w) if c not in code_set]
                if candidates:
                    new_c = rng.choice(candidates)
                else:
                    new_c = rng.randint(0, N - 1)
                    while new_c in code_set:
                        new_c = rng.randint(0, N - 1)
            else:
                new_c = rng.randint(0, N - 1)
                while new_c in code_set:
                    new_c = rng.randint(0, N - 1)
            
            code[i] = new_c
            code_set.add(new_c)
            # Update cover counts
            for w in neighbor_lists[old]:
                cover_counts[w] -= 1
            for w in neighbor_lists[new_c]:
                cover_counts[w] += 1
        
        return code
    
    def two_swap_local_search(code, rng):
        """2-swap: try to swap one codeword to enable removal of another"""
        code = list(code)
        code_set = set(code)
        
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                c1 = code[i]
                
                # Find words uniquely covered by c1
                newly_uncovered = []
                for w in neighbor_lists[c1]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)
                
                if len(newly_uncovered) > 3:
                    continue
                
                # Try to find a replacement c2 such that removing c1 and adding c2 covers all newly_uncovered
                # and also makes some other codeword redundant
                candidates = set()
                for w in newly_uncovered:
                    coders = get_coders_for_word(w)
                    for c in coders:
                        if c not in code_set and c != c1:
                            candidates.add(c)
                
                if not candidates:
                    continue
                
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for c2 in cand_list[:40]:
                    mask2 = neighbor_masks[c2]
                    all_covered = all((mask2 >> w) & 1 for w in newly_uncovered)
                    if not all_covered:
                        continue
                    
                    # Check if adding c2 makes another codeword redundant
                    best_remove = -1
                    for j in range(len(code)):
                        if j == i:
                            continue
                        cj = code[j]
                        can_remove = True
                        for w in neighbor_lists[cj]:
                            if cover_counts[w] == 1 and w not in newly_uncovered:
                                if not ((mask2 >> w) & 1):
                                    can_remove = False
                                    break
                        if can_remove:
                            best_remove = j
                            break
                    
                    if best_remove != -1:
                        # Perform the swap and removal
                        code.pop(i)
                        for w in neighbor_lists[c1]:
                            cover_counts[w] -= 1
                        code_set.discard(c1)
                        
                        cj = code[best_remove]
                        code.pop(best_remove)
                        for w in neighbor_lists[cj]:
                            cover_counts[w] -= 1
                        code_set.discard(cj)
                        
                        code.append(c2)
                        for w in neighbor_lists[c2]:
                            cover_counts[w] += 1
                        code_set.add(c2)
                        
                        improved = True
                        break
                
                if improved:
                    break
        
        return code
    
    def full_pipeline(seed):
        """Run the full optimization pipeline with a given seed"""
        rng = random.Random(seed)
        code = greedy_cover_criticality(rng)
        code = local_search_remove(code)
        code = two_opt_local_search(code, rng)
        code = two_swap_local_search(code, rng)
        return code
    
    # Initial solution with multiple seeds
    best_code = None
    best_size = float('inf')
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 20:
            break
        candidate = full_pipeline(seed)
        if len(candidate) < best_size:
            best_code = candidate
            best_size = len(candidate)
    
    # ILS: Iterated Local Search with perturbation
    rng_ils = random.Random(12345)
    current_code = list(best_code)
    current_size = best_size
    
    iteration = 0
    while time.time() < deadline - 8:
        iteration += 1
        
        # Vary perturbation strength
        strength = 0.02 + 0.04 * (iteration % 4) / 4
        perturbed = perturb(current_code, rng_ils, strength=strength)
        
        # Run local search on perturbed solution
        improved = local_search_remove(perturbed)
        improved = two_opt_local_search(improved, rng_ils)
        improved = two_swap_local_search(improved, rng_ils)
        
        # Accept if better or equal (to allow exploration)
        if len(improved) < current_size:
            current_code = improved
            current_size = len(improved)
            if current_size < best_size:
                best_code = list(current_code)
                best_size = current_size
        elif len(improved) == current_size and rng_ils.random() < 0.4:
            # Sometimes accept equal solutions to escape plateaus
            current_code = improved
        
        # Occasionally restart from scratch with different seeds
        if iteration % 6 == 0 and time.time() < deadline - 18:
            fresh = full_pipeline(rng_ils.randint(1, 10000))
            if len(fresh) < current_size:
                current_code = fresh
                current_size = len(fresh)
                if current_size < best_size:
                    best_code = list(current_code)
                    best_size = current_size
    
    # Final verification and fix if needed
    covered_mask = 0
    for c in best_code:
        covered_mask |= neighbor_masks[c]
        
    if covered_mask != (1 << N) - 1:
        uncovered = ((1 << N) - 1) & ~covered_mask
        best_code_set = set(best_code)
        while uncovered and time.time() < deadline - 2:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code_set:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            best_code_set.add(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
