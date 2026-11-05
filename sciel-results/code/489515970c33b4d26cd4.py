import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit, keep 32s buffer)

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]
    
    # Precompute neighbors as numpy array
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
        
    # Track which codeword indices are in the set
    in_set = set(codewords)
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    random.seed(42)
    np.random.seed(42)
    
    # SA loop
    T = 1.0
    T_min = 0.005
    cooling = 0.9995
    
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline - 2:
        steps += 1
        
        # Find redundant words efficiently
        # A codeword c is redundant if for all w in neighbors[c], coverage_counts[w] >= 2
        # i.e., min(coverage_counts[neighbors[c]]) >= 2
        
        # Vectorized check: for each codeword in the set, compute min coverage of its neighbors
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)
                
        if redundant_list:
            # Remove a redundant word - prefer one with highest "slack"
            # Slack = min coverage of neighbors (higher = safer to remove)
            if len(redundant_list) > 1:
                slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                max_slack = max(slacks)
                # Pick randomly among those with max slack
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
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            T = max(T * cooling, T_min)
            continue
            
        else:
            # No redundant words. Try swaps.
            # Strategy: pick codeword to remove, find valid replacement
            
            # Try multiple removal candidates per iteration to increase success rate
            found_swap = False
            num_tries = 5
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]
                
                # Find critical words: neighbors of c_to_remove with coverage exactly 1
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]
                
                if len(critical_words) == 0:
                    # Shouldn't happen if not redundant, but handle gracefully
                    continue
                
                # Candidates must be in intersection of neighbors of all critical words
                # Start with neighbors of first critical word
                w0 = int(critical_words[0])
                candidates = neighbors[w0]
                
                # Filter: must be in all other critical words' neighborhoods
                # and must not be c_to_remove
                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    # Check if cand covers all critical words
                    # i.e., all critical_words are in neighbors[cand]
                    # Equivalent: critical_words is subset of neighbor_sets[cand]
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))
                
                if not valid_candidates:
                    continue
                
                # Score candidates by how much extra coverage they provide
                # Prefer candidates that cover words with higher current coverage (more redundancy)
                scored_candidates = []
                for cand in valid_candidates:
                    # Extra coverage = sum of coverage_counts for neighbors of cand
                    # This measures how much "redundancy" the new codeword adds
                    extra_cov = int(np.sum(coverage_counts[neighbors[cand]]))
                    scored_candidates.append((extra_cov, cand))
                
                # Sort by extra coverage (descending) and pick from top candidates
                scored_candidates.sort(key=lambda x: -x[0])
                # Pick randomly from top 3 to maintain diversity
                top_k = min(3, len(scored_candidates))
                c_to_add = random.choice([x[1] for x in scored_candidates[:top_k]])
                
                # Perform swap
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)
                
                found_swap = True
                break
            
            if found_swap:
                # Check if this swap enables immediate removal of another word
                # (i.e., if the swap created redundancy elsewhere)
                for idx, c in enumerate(codewords):
                    if coverage_counts[neighbors[c]].min() >= 2:
                        codewords.pop(idx)
                        coverage_counts[neighbors[c]] -= 1
                        in_set.discard(c)
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        break
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            T = max(T * cooling, T_min)
            
            # If stuck for too long, restart with a perturbation
            if no_improve_count > 200:
                # Perturb: randomly swap 2-3 codewords
                num_perturb = random.randint(2, 3)
                for _ in range(num_perturb):
                    if len(codewords) < 2:
                        break
                    idx1 = random.randint(0, len(codewords) - 1)
                    idx2 = random.randint(0, len(codewords) - 1)
                    if idx1 == idx2:
                        continue
                    c1 = codewords[idx1]
                    c2 = codewords[idx2]
                    # Swap them in the list
                    codewords[idx1], codewords[idx2] = c2, c1
                    # Update coverage counts
                    coverage_counts[neighbors[c1]] -= 1
                    coverage_counts[neighbors[c2]] -= 1
                    coverage_counts[neighbors[c2]] += 1
                    coverage_counts[neighbors[c1]] += 1
                no_improve_count = 0
                T = 1.0  # Reset temperature

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
        final_indices = best_codewords
    else:
        final_indices = codewords

    return [idx_to_list(idx) for idx in final_indices]

# EVOLVE-BLOCK-END
