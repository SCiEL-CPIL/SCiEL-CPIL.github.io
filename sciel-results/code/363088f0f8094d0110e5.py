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
    
    # Build reverse mapping: for each word, which codewords cover it
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)
    
    def greedy_cover_criticality(rng):
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        # Precompute criticality for each word
        criticality = [len(word_to_coders[w]) for w in range(N)]
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find word with minimum criticality among uncovered (sample-based for speed)
            min_crit = N + 1
            target_word = -1
            # Sample a subset of uncovered words to find critical one
            sample_size = min(100, 1 << (uncovered.bit_length() - 1))
            temp = uncovered
            sampled = 0
            while temp and sampled < sample_size:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                crit = criticality[w_idx]
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:
                    break
                temp ^= lsb
                sampled += 1
            
            # If no good sample found, do full scan
            if target_word == -1:
                temp = uncovered
                while temp:
                    lsb = temp & (-temp)
                    w_idx = lsb.bit_length() - 1
                    crit = criticality[w_idx]
                    if crit < min_crit:
                        min_crit = crit
                        target_word = w_idx
                    if min_crit == 1:
                        break
                    temp ^= lsb
            
            if target_word == -1:
                break
            
            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1
            
            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()
                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx
            
            if best_c == -1 or best_new_count == 0:
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]
        
        return code
    
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code
            
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1
                
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
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
                
                best_add = -1
                best_gain = -1
                
                # Collect candidate codewords that cover all newly_uncovered words
                candidate_codewords = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_codewords.add(c)
                
                # Limit candidates to avoid excessive computation
                if len(candidate_codewords) > 200:
                    candidates_list = list(candidate_codewords)
                    rng.shuffle(candidates_list)
                    candidate_codewords = set(candidates_list[:200])
                
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
                    
                    # Count new coverage (words with cover_count == 0 that c_add covers)
                    new_coverage = (mask_add & ~((1 << N) - 1)).bit_count()  # placeholder
                    # Better: count words in neighbor_lists[c_add] with cover_counts == 0
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
        """Randomly swap some codewords to escape local minima"""
        code = list(code)
        if not code:
            return code
        
        code_set = set(code)
        n_swaps = max(1, int(len(code) * strength))
        for _ in range(n_swaps):
            if len(code) < 2:
                break
            i = rng.randint(0, len(code) - 1)
            old = code[i]
            # Remove old codeword and pick a random new one
            code_set.discard(old)
            new_c = rng.randint(0, N - 1)
            while new_c in code_set:
                new_c = rng.randint(0, N - 1)
            code[i] = new_c
            code_set.add(new_c)
        
        return code
    
    def full_pipeline(seed):
        """Run the full optimization pipeline with a given seed"""
        rng = random.Random(seed)
        code = greedy_cover_criticality(rng)
        code = local_search_remove(code)
        code = two_opt_local_search(code, rng)
        return code
    
    # Initial solution
    best_code = full_pipeline(42)
    best_size = len(best_code)
    
    # ILS: Iterated Local Search with perturbation
    rng_ils = random.Random(12345)
    current_code = list(best_code)
    current_size = best_size
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Perturb current solution
        perturbed = perturb(current_code, rng_ils, strength=0.03 + 0.02 * (iteration % 5) / 5)
        
        # Run local search on perturbed solution
        improved = local_search_remove(perturbed)
        improved = two_opt_local_search(improved, rng_ils)
        
        # Accept if better or equal (to allow exploration)
        if len(improved) < current_size:
            current_code = improved
            current_size = len(improved)
            if current_size < best_size:
                best_code = list(current_code)
                best_size = current_size
        elif len(improved) == current_size and rng_ils.random() < 0.3:
            # Sometimes accept equal solutions to escape plateaus
            current_code = improved
        
        # Occasionally restart from scratch with different seeds
        if iteration % 8 == 0 and time.time() < deadline - 15:
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
        while uncovered and time.time() < deadline:
            best_c = None
            best_count = -1
            for c in range(N):
                if c in best_code:
                    continue
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c is None:
                break
            best_code.append(best_c)
            covered_mask |= neighbor_masks[best_c]
            uncovered &= ~neighbor_masks[best_c]
    
    result = [all_words[i] for i in best_code]
    return result
