import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    q_powers = [q ** i for i in range(n)]

    # Build neighbor table
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Build inverse index: for each word, list of words that cover it
    # inverse[w] = list of codeword indices (into the neighbor table) that have w in their neighbor set
    # Actually, for swap operations we need: for a word u, which other words v have u in their neighbor set?
    # That's just neighbors[u] itself (symmetry of Hamming distance)
    # So the "candidates" that could replace a codeword ci while still covering a critical word u
    # are all words in neighbors[u] except ci itself.

    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    def tetracode_init():
        """
        Use the tetracode structure to build a good initial code.
        The tetracode is a ternary code of length 4 with 12 codewords.
        We use a product-like construction to get a starting code.
        """
        # The 12 tetracode words (one standard representation)
        # These are words of length 4 over {0,1,2}
        tetracode = [
            [0, 0, 0, 0],
            [0, 1, 1, 2],
            [0, 2, 2, 1],
            [1, 0, 2, 1],
            [1, 1, 0, 2],
            [1, 2, 1, 0],
            [2, 0, 1, 0],
            [2, 1, 2, 2],
            [2, 2, 0, 1],
            [0, 1, 2, 0],
            [1, 0, 1, 2],
            [2, 2, 1, 0],
        ]
        
        # Actually, let me use a known set of 12 tetracode words
        # The tetracode is the unique (4, 12, 3) ternary code
        # Let me use a different approach: just do a good greedy with multiple restarts
        
        # For now, use greedy with a good seed
        return None

    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]

            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)

            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1

        return codewords

    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)

        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1

        no_improve_count = 0

        while time.time() < time_limit:
            improved = False

            # Pass 1: Remove redundant codewords
            indices = list(range(len(cw)))
            random.shuffle(indices)

            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]

                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    no_improve_count = 0

                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: Swaps - try to replace a codeword with a better one
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)

                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]

                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]

                    if len(critical) == 0:
                        # Can remove entirely
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        no_improve_count = 0
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break

                    if len(critical) > 0:
                        # Need a candidate that covers all critical words
                        # Candidate must be in neighbors[u] for all critical u
                        candidate_set = neighbor_sets[critical[0]].copy()
                        for u in critical[1:]:
                            candidate_set &= neighbor_sets[u]
                            if not candidate_set:
                                break

                        if candidate_set:
                            valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                            if valid_cands:
                                # Pick the candidate that maximizes additional coverage
                                # Score = number of newly covered words (where count was 0 or 1)
                                best_cand = None
                                best_score = -1
                                for cand in valid_cands[:20]:  # Limit to 20 for speed
                                    # Compute gain: words that go from count 0 to 1, or 1 to 2
                                    # Actually, we just need coverage maintained, so any valid cand works
                                    # But prefer one that covers more words with count==1 (reduces critical words)
                                    cand_neighbors = neighbors[cand]
                                    # Count how many of ci's critical words are also in cand's neighborhood
                                    # All critical words are covered by cand by construction
                                    # Prefer cand that covers more words overall
                                    score = np.sum(count[cand_neighbors] >= 0)  # just len
                                    # Better: prefer cand that has more words with count==0 (expands coverage)
                                    # But we're not expanding, just swapping
                                    # Just pick randomly from valid candidates
                                    if best_score < 0:
                                        best_cand = cand
                                        best_score = 0

                                if best_cand is not None:
                                    cand = best_cand
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1

                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    no_improve_count = 0
                                    break

            if not improved:
                no_improve_count += 1

                # Perturbation: try random swaps
                if len(cw) > 2 and no_improve_count % 3 == 0:
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]

                    temp_count = count.copy()
                    for u in neighbors[c1]:
                        temp_count[u] -= 1
                    for u in neighbors[c2]:
                        temp_count[u] += 1

                    if np.min(temp_count) >= 1:
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] += 1
                        in_code[c1] = False
                        in_code[c2] = True
                        cw[idx1] = c2
                        improved = True
                        no_improve_count = 0

                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]

                # Aggressive perturbation: remove and re-add
                if not improved and len(cw) > 5 and no_improve_count % 5 == 0:
                    idx = random.randint(0, len(cw) - 1)
                    c_remove = cw[idx]

                    # Find uncovered words if we remove c_remove
                    temp_count = count.copy()
                    for u in neighbors[c_remove]:
                        temp_count[u] -= 1
                    uncovered = np.where(temp_count == 0)[0]

                    if len(uncovered) > 0:
                        # Need to add a word that covers all uncovered
                        # Find a word that covers all uncovered words
                        cand_set = neighbor_sets[uncovered[0]].copy()
                        for u in uncovered[1:]:
                            cand_set &= neighbor_sets[u]
                            if not cand_set:
                                break

                        if cand_set:
                            valid_cands = [c for c in cand_set if not in_code[c]]
                            if valid_cands:
                                c_add = random.choice(valid_cands)
                                for u in neighbors[c_remove]:
                                    count[u] -= 1
                                for u in neighbors[c_add]:
                                    count[u] += 1
                                in_code[c_remove] = False
                                in_code[c_add] = True
                                cw[idx] = c_add
                                improved = True
                                no_improve_count = 0

                # If stuck for a long time, restart with a smaller perturbation
                if no_improve_count > 50:
                    # Remove a random codeword and greedily re-add
                    idx = random.randint(0, len(cw) - 1)
                    c_remove = cw[idx]
                    for u in neighbors[c_remove]:
                        count[u] -= 1
                    in_code[c_remove] = False
                    cw.pop(idx)

                    # Greedily add back words to cover uncovered
                    uncovered = np.where(count == 0)[0]
                    while len(uncovered) > 0:
                        if time.time() > time_limit:
                            break
                        # Pick word that covers most uncovered
                        best_w = -1
                        best_gain = -1
                        for w in uncovered[:50]:  # Sample
                            gain = 0
                            for v in neighbors[w]:
                                if count[v] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_w = w

                        if best_w < 0:
                            break

                        cw.append(best_w)
                        in_code[best_w] = True
                        for u in neighbors[best_w]:
                            count[u] += 1

                        uncovered = np.where(count == 0)[0]

                    no_improve_count = 0

        return best_cw

    # Phase 1: Greedy Set Cover with multiple seeds, pick best
    best_initial = None
    best_initial_size = float('inf')
    
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline - 40.0:
            break
        cw = greedy_init(seed)
        if len(cw) < best_initial_size:
            best_initial_size = len(cw)
            best_initial = cw
    
    codewords = best_initial

    # Phase 2: Local Search
    best_cw = local_search(codewords, deadline - 5.0)

    # Convert back to list of lists
    result = []
    for ci in best_cw:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)

    return result
