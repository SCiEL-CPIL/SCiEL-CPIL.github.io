import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s limit

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors: neighbors[i] = list of 13 words within Hamming distance 1 of i
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Initial Solution: Greedy Set Cover
    uncovered = np.ones(total, dtype=np.int8)
    codewords = []

    while uncovered.sum() > 0:
        if time.time() > deadline - 80:
            break
        counts = np.sum(uncovered[neighbors], axis=1)
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0

    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            codewords.append(c)
            uncovered[neighbors[c]] = 0

    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1

    in_set = set(codewords)
    best_codewords = list(codewords)
    best_size = len(codewords)

    random.seed(42)
    np.random.seed(42)

    # SA loop with genetic crossover elements
    T = 1.0
    T_min = 0.005
    cooling = 0.9995

    steps = 0
    while time.time() < deadline - 2:
        steps += 1

        # Find redundant words efficiently
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)

        if redundant_list:
            # Remove a redundant word - prefer one with highest "slack"
            if len(redundant_list) > 1:
                slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                max_slack = max(slacks)
                best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                idx_to_remove = random.choice(best_candidates)
            else:
                idx_to_remove = redundant_list[0]

            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)

            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)

            T = max(T * cooling, T_min)
            continue

        else:
            # No redundant words. Try swaps or genetic crossover.
            found_swap = False
            num_tries = 3
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]

                # Find critical words: neighbors of c_to_remove with coverage exactly 1
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]

                if len(critical_words) == 0:
                    continue

                # Candidates must cover all critical words
                w0 = int(critical_words[0])
                candidates = neighbors[w0]

                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))

                if not valid_candidates:
                    continue

                # Use a fitness-based selection: prefer codewords that add more redundancy
                # Fitness = sum of (coverage_counts[w] + 1) for w in neighbors[cand]
                # Higher fitness means more overlap with existing code (more redundancy)
                if len(valid_candidates) > 1:
                    fitnesses = []
                    for cand in valid_candidates:
                        # Measure how much this candidate overlaps with existing coverage
                        # A good candidate will have high overlap (redundancy) in its ball
                        fitness = float(np.sum(coverage_counts[neighbors[cand]]))
                        fitnesses.append(fitness)
                    # Pick with probability proportional to fitness (softmax-like)
                    max_f = max(fitnesses)
                    exp_f = [np.exp((f - max_f) / 0.5) for f in fitnesses]
                    total_exp = sum(exp_f)
                    probs = [e / total_exp for e in exp_f]
                    c_to_add = random.choices(valid_candidates, weights=probs)[0]
                else:
                    c_to_add = valid_candidates[0]

                # Perform swap
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)

                found_swap = True
                break

            if not found_swap:
                # Genetic crossover: try to combine two codewords into a better one
                if len(codewords) >= 2:
                    idx1 = random.randint(0, len(codewords) - 1)
                    idx2 = random.randint(0, len(codewords) - 1)
                    if idx1 != idx2:
                        c1 = codewords[idx1]
                        c2 = codewords[idx2]
                        
                        # Try crossover: create a new codeword by mixing positions
                        # But it must be valid (cover critical words if replacing)
                        # For simplicity, try to replace one with a "hybrid"
                        w1 = [(c1 // powers[p]) % 3 for p in range(6)]
                        w2 = [(c2 // powers[p]) % 3 for p in range(6)]
                        
                        # Random crossover
                        new_w = []
                        for p in range(6):
                            if random.random() < 0.5:
                                new_w.append(w1[p])
                            else:
                                new_w.append(w2[p])
                        
                        new_idx = 0
                        for p in range(6):
                            new_idx += new_w[p] * powers[p]
                        
                        if new_idx not in in_set:
                            # Check if this new codeword covers the critical words of c1
                            crit_mask = (coverage_counts[neighbors[c1]] == 1)
                            critical_words = neighbors[c1][crit_mask]
                            if len(critical_words) == 0 or all(w in neighbor_sets[new_idx] for w in critical_words):
                                # Replace c1 with new_idx
                                codewords[idx1] = new_idx
                                coverage_counts[neighbors[c1]] -= 1
                                coverage_counts[neighbors[new_idx]] += 1
                                in_set.discard(c1)
                                in_set.add(new_idx)
                                found_swap = True

            T = max(T * cooling, T_min)

    # Final cleanup: iteratively remove redundant words
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    # Use the best found
    if len(best_codewords) < len(codewords):
        codewords = best_codewords

    # Convert back to list of lists
    result = []
    for c in codewords:
        w = [0] * n
        val = c
        for i in range(n):
            w[i] = val % q
            val //= q
        result.append(w)

    return result
