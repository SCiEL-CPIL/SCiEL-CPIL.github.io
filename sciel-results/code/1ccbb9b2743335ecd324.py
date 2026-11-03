import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    total_space = q ** n
    
    # Precompute word to index and index to word
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute cover masks as Python integers (bitsets)
    # mask[i] has bit j set if word j is within Hamming distance <= radius of word i
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    mask |= (1 << j)
        cover_masks[i] = mask
        
    full_mask = (1 << total_space) - 1
    
    # Precompute reverse cover: for each word, which codewords cover it
    # This is static and helps in the swap phase of local search
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Greedy Set Cover with cached gains
    def greedy_cover():
        uncovered = full_mask
        code = []
        gains = [cover_masks[i].bit_count() for i in range(total_space)]
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_space):
                if gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
            if best_idx == -1:
                break
            code.append(best_idx)
            new_uncovered = uncovered & ~cover_masks[best_idx]
            if new_uncovered == uncovered:
                # best_idx adds nothing new (shouldn't happen if gains>0)
                uncovered = new_uncovered
                break
            # Update gains only for words in the delta
            delta = uncovered ^ new_uncovered
            m = delta
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                m ^= low
                for i in reverse_cover[w_idx]:
                    gains[i] -= 1
            uncovered = new_uncovered
        return code

    # Local Search: Remove redundant codewords, perform swaps, and swap+add
    def local_search_reduce(code):
        if not code:
            return code
        
        current_code = code[:]
        current_code_set = set(current_code)
        
        # Build cover_count array: cover_count[w] = number of codewords covering word w
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)
            
            # Phase 1: Removal
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code
                
                idx = current_code[i]
                mask = cover_masks[idx]
                
                # Check if all words covered by idx are covered by at least one other codeword
                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low
                
                if removable:
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break
            
            # Phase 2: Swaps (if no removal happened)
            if not improved:
                max_unique_to_cover = 3
                
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    
                    if unique_mask == 0:
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break
                    
                    if unique_mask.bit_count() > max_unique_to_cover:
                        continue
                    
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    
                    candidates = set(reverse_cover[words_in_unique[0]])
                    
                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])
                    
                    candidates.discard(idx)
                    candidates -= current_code_set
                    
                    if candidates:
                        new_idx = random.choice(list(candidates))
                        
                        m = cover_masks[idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        
                        m = cover_masks[new_idx]
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m ^= low
                        
                        current_code[i] = new_idx
                        current_code_set.remove(idx)
                        current_code_set.add(new_idx)
                        
                        improved = True
                        break
            
            # Phase 3: Swap + Add (remove one, add two) to reduce size by 1
            if not improved and len(current_code) > 2:
                n_cw = len(current_code)
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code
                    
                    idx = current_code[i]
                    mask = cover_masks[idx]
                    
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low
                    
                    if unique_mask.bit_count() > 5:
                        continue
                    
                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low
                    
                    if not words_in_unique:
                        continue
                    
                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))
                    
                    # Find two codewords that together cover all unique words
                    found_pair = False
                    for a_pos in range(n_cw):
                        if a_pos == i:
                            continue
                        a_idx = current_code[a_pos]
                        a_mask = cover_masks[a_idx]
                        # Check if a covers at least one unique word
                        if not (a_mask & unique_mask):
                            continue
                        remaining = unique_mask & ~a_mask
                        if remaining == 0:
                            # a alone covers all unique words - just swap
                            new_idx = a_idx
                            # Remove idx, add a (a already in set, so we're swapping)
                            # Actually a is already in the code, so this is just a swap
                            m = cover_masks[idx]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                            m = cover_masks[a_idx]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m ^= low
                            current_code[i] = a_idx
                            current_code_set.remove(idx)
                            current_code_set.add(a_idx)
                            # Remove the duplicate a_idx
                            # Find and remove the old a_idx entry
                            for j in range(len(current_code)):
                                if j != i and current_code[j] == a_idx:
                                    m2 = cover_masks[a_idx]
                                    while m2:
                                        low2 = m2 & (-m2)
                                        w_idx2 = low2.bit_length() - 1
                                        cover_count[w_idx2] -= 1
                                        m2 ^= low2
                                    current_code.pop(j)
                                    current_code_set.discard(a_idx)
                                    break
                            improved = True
                            found_pair = True
                            break
                        
                        # Find b that covers remaining
                        for b_pos in range(n_cw):
                            if b_pos == i or b_pos == a_pos:
                                continue
                            b_idx = current_code[b_pos]
                            b_mask = cover_masks[b_idx]
                            if remaining & ~b_mask:
                                continue
                            # a and b together cover all unique words
                            # Remove idx, add nothing new (a and b already in code)
                            # But we need to remove one of a or b to keep size same... 
                            # Actually we want to reduce size by 1, so we remove idx
                            # and don't add anything new. But a and b are already there.
                            # This means we just remove idx if a and b cover its unique words.
                            m = cover_masks[idx]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low
                            current_code.pop(i)
                            current_code_set.remove(idx)
                            improved = True
                            found_pair = True
                            break
                        
                        if found_pair:
                            break
                    
                    if found_pair:
                        break
                        
        return current_code

    # Initial greedy construction
    best_code = greedy_cover()
    
    # Apply local search
    best_code = local_search_reduce(best_code)
    
    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    # Run randomized restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code
    
    # Genetic Algorithm with bitset-based chromosomes
    def genetic_algorithm(initial_code, population_size=10, generations=5):
        if time.time() > deadline - 5.0:
            return initial_code
        
        # Encode codeword set as a bitset over total_space
        def encode(code_list):
            bits = 0
            for idx in code_list:
                bits |= (1 << idx)
            return bits
        
        def decode(bits):
            code = []
            m = bits
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                code.append(idx)
                m ^= low
            return code
        
        def fitness(bits):
            """Lower is better: penalize uncovered words, reward minimal size"""
            # Count covered words
            covered = 0
            m = bits
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                covered |= cover_masks[idx]
                m ^= low
            
            uncovered_count = total_space - covered.bit_count()
            code_size = bits.bit_count()
            # Weight: heavily penalize uncovered, then minimize size
            return uncovered_count * 10000 + code_size
        
        def crossover(parent1, parent2):
            """Bitset crossover: combine high-performing parents"""
            # Union of parents, then randomly drop some codewords
            child = parent1 | parent2
            # Randomly remove some codewords from the union
            m = child
            while m:
                low = m & (-m)
                if random.random() < 0.3:  # 30% chance to drop each codeword
                    child ^= low
                m ^= low
            return child
        
        def mutate(bits):
            """Randomly swap or remove codewords"""
            m = bits
            code_size = m.bit_count()
            
            # Option 1: Remove a random codeword
            if random.random() < 0.4 and code_size > 1:
                m = bits
                while m:
                    low = m & (-m)
                    if random.random() < 0.5:
                        bits ^= low
                        break
                    m ^= low
                return bits
            
            # Option 2: Add a random codeword
            if random.random() < 0.4:
                new_idx = random.randint(0, total_space - 1)
                bits |= (1 << new_idx)
                return bits
            
            # Option 3: Swap a codeword
            if code_size > 1:
                m = bits
                old_idx = -1
                while m:
                    low = m & (-m)
                    if random.random() < 0.5:
                        old_idx = low.bit_length() - 1
                        break
                    m ^= low
                
                if old_idx != -1:
                    new_idx = random.randint(0, total_space - 1)
                    if new_idx != old_idx:
                        bits ^= (1 << old_idx)
                        bits |= (1 << new_idx)
            
            return bits
        
        # Initialize population
        population = [encode(initial_code)]
        for _ in range(population_size - 1):
            # Create diverse initial individuals
            cand = greedy_cover_randomized()
            population.append(encode(cand))
        
        # Sort by fitness
        population.sort(key=fitness)
        best_bits = population[0]
        best_fit = fitness(best_bits)
        
        for gen in range(generations):
            if time.time() > deadline - 5.0:
                break
            
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = max(2, population_size // 5)
            new_population.extend(population[:elite_size])
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                def select():
                    idx1 = random.randint(0, len(population) - 1)
                    idx2 = random.randint(0, len(population) - 1)
                    if fitness(population[idx1]) < fitness(population[idx2]):
                        return population[idx1]
                    else:
                        return population[idx2]
                
                parent1 = select()
                parent2 = select()
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutation
                child = mutate(child)
                
                new_population.append(child)
            
            population = new_population
            population.sort(key=fitness)
            
            if fitness(population[0]) < best_fit:
                best_fit = fitness(population[0])
                best_bits = population[0]
        
        best_code = decode(best_bits)
        # Ensure validity: if not fully covered, fall back to initial
        covered = 0
        for idx in best_code:
            covered |= cover_masks[idx]
        if covered.bit_count() < total_space:
            return initial_code
        
        return best_code
    
    # Run genetic algorithm if time permits
    if time.time() < deadline - 10.0:
        ga_code = genetic_algorithm(best_code, population_size=8, generations=4)
        ga_code = local_search_reduce(ga_code)
        if len(ga_code) < len(best_code):
            best_code = ga_code
    
    return [all_words[i] for i in best_code]

# EVOLVE-BLOCK-END
