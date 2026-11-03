import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    total_space = q ** n

    # Precompute all words and helper functions
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

    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover masks: for each word, the set of words within Hamming distance <= radius
    # Using Python integers as bitmasks
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

    # Greedy set cover
    def greedy_cover():
        uncovered = full_mask
        code = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    if gain == uncovered.bit_count():
                        break
            if best_idx == -1:
                break
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        return code

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

    # Local search: remove redundant codewords
    def local_search_reduce(code):
        if not code:
            return code

        current_code = code[:]
        code_set = set(current_code)

        # Build cover_count: cover_count[w] = number of codewords covering word w
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low

        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

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
                    # Decrement cover_count and remove
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    code_set.discard(idx)
                    improved = True
                    break
                else:
                    # Try to find a swap: replace idx with another codeword that covers its unique words
                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= low
                        m ^= low

                    if unique_mask:
                        # Find a candidate j not in current_code that covers all words in unique_mask
                        found = False
                        for j in range(total_space):
                            if j in code_set:
                                continue
                            if (cover_masks[j] & unique_mask) == unique_mask:
                                # Update cover_count
                                m = mask
                                while m:
                                    low = m & (-m)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] -= 1
                                    m ^= low

                                m_new = cover_masks[j]
                                while m_new:
                                    low = m_new & (-m_new)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] += 1
                                    m_new ^= low

                                code_set.discard(idx)
                                code_set.add(j)
                                current_code[i] = j
                                improved = True
                                found = True
                                break

                        if found:
                            break

        return current_code

    # 2-for-1 local search: replace 2 codewords with 1
    def local_search_2_for_1(code):
        if len(code) < 2:
            return code

        current_code = code[:]
        code_set = set(current_code)

        # Build cover_count
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low

        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code

                for j in range(i + 1, n_cw):
                    if time.time() > deadline:
                        return current_code

                    idx_i = current_code[i]
                    idx_j = current_code[j]
                    mask_i = cover_masks[idx_i]
                    mask_j = cover_masks[idx_j]

                    # Find words uniquely covered by the pair (i, j)
                    unique_mask = 0
                    m = mask_i | mask_j
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= low
                        m ^= low

                    if not unique_mask:
                        continue

                    # Find a candidate k not in current_code that covers all words in unique_mask
                    found = False
                    for k in range(total_space):
                        if k in code_set:
                            continue
                        if (cover_masks[k] & unique_mask) == unique_mask:
                            # Update cover_count: remove i and j, add k
                            m = mask_i
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low

                            m = mask_j
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low

                            m_new = cover_masks[k]
                            while m_new:
                                low = m_new & (-m_new)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m_new ^= low

                            code_set.discard(idx_i)
                            code_set.discard(idx_j)
                            code_set.add(k)

                            # Remove j and i from current_code
                            current_code.pop(j)
                            current_code.pop(i)
                            current_code.append(k)

                            improved = True
                            found = True
                            break

                    if found:
                        break

                if improved:
                    break

        return current_code

    # 3-for-1 local search: replace 3 codewords with 1
    def local_search_3_for_1(code):
        if len(code) < 3:
            return code

        current_code = code[:]
        code_set = set(current_code)

        # Build cover_count
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low

        improved = True
        while improved:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

            # Sample random triples to avoid O(n^3)
            if n_cw > 20:
                triples = set()
                for _ in range(min(500, n_cw * (n_cw - 1) * (n_cw - 2) // 6)):
                    a, b, c = random.sample(range(n_cw), 3)
                    triples.add((a, b, c))
            else:
                triples = [(a, b, c) for a in range(n_cw) for b in range(a + 1, n_cw) for c in range(b + 1, n_cw)]

            for i, j, k in triples:
                if time.time() > deadline:
                    return current_code

                if i >= len(current_code) or j >= len(current_code) or k >= len(current_code):
                    continue

                idx_i = current_code[i]
                idx_j = current_code[j]
                idx_k = current_code[k]
                mask_i = cover_masks[idx_i]
                mask_j = cover_masks[idx_j]
                mask_k = cover_masks[idx_k]

                # Find words uniquely covered by the triple (i, j, k)
                unique_mask = 0
                m = mask_i | mask_j | mask_k
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] == 1:
                        unique_mask |= low
                    m ^= low

                if not unique_mask:
                    continue

                # Find a candidate l not in current_code that covers all words in unique_mask
                found = False
                for l in range(total_space):
                    if l in code_set:
                        continue
                    if (cover_masks[l] & unique_mask) == unique_mask:
                        # Update cover_count: remove i, j, k and add l
                        for mask in [mask_i, mask_j, mask_k]:
                            m = mask
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low

                        m_new = cover_masks[l]
                        while m_new:
                            low = m_new & (-m_new)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] += 1
                            m_new ^= low

                        code_set.discard(idx_i)
                        code_set.discard(idx_j)
                        code_set.discard(idx_k)
                        code_set.add(l)

                        # Remove k, j, i from current_code
                        current_code.pop(k)
                        current_code.pop(j)
                        current_code.pop(i)
                        current_code.append(l)

                        improved = True
                        found = True
                        break

                if found:
                    break

        return current_code

    # Combined local search
    def full_local_search(code):
        code = local_search_reduce(code)
        code = local_search_2_for_1(code)
        code = local_search_3_for_1(code)
        while True:
            if time.time() > deadline:
                break
            old_len = len(code)
            code = local_search_reduce(code)
            code = local_search_2_for_1(code)
            code = local_search_3_for_1(code)
            if len(code) >= old_len:
                break
        return code

    # Genetic algorithm
    def genetic_algorithm(pop_size=50, generations=100):
        # Initialize population with greedy solutions
        population = []
        for _ in range(pop_size):
            if random.random() < 0.3:
                pop = greedy_cover()
            else:
                pop = greedy_cover_randomized()
            pop = local_search_reduce(pop)
            population.append(pop)

        best_code = min(population, key=len)
        best_len = len(best_code)

        for gen in range(generations):
            if time.time() > deadline - 10:
                break

            # Evaluate fitness
            fitness = [len(ind) for ind in population]

            # Selection: tournament
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                candidates = random.sample(range(len(population)), min(3, len(population)))
                winner = min(candidates, key=lambda i: fitness[i])
                parent1 = population[winner]

                if random.random() < 0.5:
                    parent2 = parent1
                else:
                    candidates2 = random.sample(range(len(population)), min(3, len(population)))
                    winner2 = min(candidates2, key=lambda i: fitness[i])
                    parent2 = population[winner2]

                # Crossover: combine parents
                if random.random() < 0.7:
                    # Uniform crossover
                    child = []
                    for idx in parent1:
                        if random.random() < 0.5:
                            child.append(idx)
                    for idx in parent2:
                        if idx not in child and random.random() < 0.5:
                            child.append(idx)

                    # Ensure coverage: add greedy completion if needed
                    if child:
                        code_set = set(child)
                        uncovered = full_mask
                        for idx in child:
                            uncovered &= ~cover_masks[idx]
                        while uncovered:
                            best_idx = -1
                            best_gain = 0
                            for i in range(total_space):
                                if i in code_set:
                                    continue
                                gain = (cover_masks[i] & uncovered).bit_count()
                                if gain > best_gain:
                                    best_gain = gain
                                    best_idx = i
                            if best_idx == -1:
                                break
                            child.append(best_idx)
                            code_set.add(best_idx)
                            uncovered &= ~cover_masks[best_idx]
                    else:
                        child = greedy_cover()
                else:
                    # Just copy parent
                    child = parent1[:]

                # Mutation
                if random.random() < 0.3:
                    # Remove a random codeword if possible
                    if len(child) > 1:
                        pos = random.randint(0, len(child) - 1)
                        idx_to_remove = child[pos]
                        # Check if removal keeps coverage
                        code_set = set(child)
                        code_set.discard(idx_to_remove)
                        uncovered = full_mask
                        for idx in code_set:
                            uncovered &= ~cover_masks[idx]
                        if uncovered == 0:
                            child.pop(pos)

                # Local search improvement
                if time.time() < deadline - 5:
                    child = local_search_reduce(child)

                new_population.append(child)

            population = new_population

            # Update best
            gen_best = min(population, key=len)
            if len(gen_best) < best_len:
                best_code = gen_best
                best_len = len(gen_best)

        return best_code

    # Initial greedy construction
    best_code = greedy_cover()
    best_code = full_local_search(best_code)

    # Run genetic algorithm
    if time.time() < deadline - 20:
        ga_code = genetic_algorithm()
        if len(ga_code) < len(best_code):
            best_code = ga_code
            best_code = full_local_search(best_code)

    # Run restarts if time permits
    if time.time() < deadline - 10.0:
        for _ in range(20):
            if time.time() > deadline - 3.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = full_local_search(cand_code)
            if len(cand_code) < len(best_code):
                best_code = cand_code

    # Convert indices back to words
    return [all_words[i] for i in best_code]
