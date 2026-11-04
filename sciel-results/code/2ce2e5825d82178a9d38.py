import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Specific case for the task
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]
        
        # Precompute neighbors
        neighbors = []
        neighbor_sets = []
        for i in range(total):
            d = [(i // powers[p]) % q for p in range(n)]
            covered = [i]
            for p in range(n):
                for alt in range(q):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        covered.append(new_i)
            neighbors.append(covered)
            neighbor_sets.append(set(covered))
            
        # Greedy Initialization
        def greedy_init():
            uncovered = set(range(total))
            codewords = []
            while uncovered:
                best_cand = -1
                best_count = -1
                for i in range(total):
                    # Count how many uncovered words are covered by i
                    count = 0
                    for w in neighbors[i]:
                        if w in uncovered:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_cand = i
                        if count == len(uncovered): # Perfect match
                            break
                if best_cand == -1:
                    break
                codewords.append(best_cand)
                uncovered.difference_update(neighbors[best_cand])
            return codewords

        codewords = greedy_init()
        if not codewords:
             return [[0]*n]
             
        codewords_set = set(codewords)
        
        # Helper to update counts
        def update_counts(cw, delta):
            for w in neighbors[cw]:
                counts[w] += delta

        counts = np.zeros(total, dtype=np.int32)
        for cw in codewords:
            update_counts(cw, 1)

        best_len = len(codewords)
        best_codewords = codewords[:]
        
        # Local Search
        while time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            for i in range(len(codewords)):
                cw = codewords[i]
                # Check if all neighbors have count > 1
                redundant = True
                for w in neighbors[cw]:
                    if counts[w] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    update_counts(cw, -1)
                    codewords.pop(i)
                    codewords_set.remove(cw)
                    improved = True
                    if len(codewords) < best_len:
                        best_len = len(codewords)
                        best_codewords = codewords[:]
                    break # Restart loop to check for new redundancies
            
            if improved:
                continue

            # 2. Try to swap codewords to enable further reduction
            swap_done = False
            for i in range(len(codewords)):
                cw_i = codewords[i]
                # Find essential words (covered only by cw_i)
                essential = []
                for w in neighbors[cw_i]:
                    if counts[w] == 1:
                        essential.append(w)
                
                if not essential:
                    # Should have been removed in step 1
                    continue

                # Find a replacement cw_j that covers all essential words
                # and is not in codewords_set
                # To speed up, we can just iterate through all candidates
                # Optimization: We can pre-filter candidates that cover the first essential word
                first_ess = essential[0]
                candidates = [j for j in range(total) if j not in codewords_set and first_ess in neighbor_sets[j]]
                
                for cw_j in candidates:
                    # Check if cw_j covers all other essential words
                    valid = True
                    for w in essential[1:]:
                        if w not in neighbor_sets[cw_j]:
                            valid = False
                            break
                    if valid:
                        # Perform Swap
                        update_counts(cw_i, -1)
                        update_counts(cw_j, 1)
                        codewords[i] = cw_j
                        codewords_set.remove(cw_i)
                        codewords_set.add(cw_j)
                        swap_done = True
                        break
                if swap_done:
                    break
            
            if swap_done:
                # A swap might have made another codeword redundant
                # So we continue to the next iteration of the main loop
                # which will check for redundancies again
                continue

            # 3. Perturbation: Stuck, try to escape
            # Randomly remove a few codewords and repair
            if time.time() > deadline - 5:
                break
                
            num_remove = random.randint(1, 3)
            to_remove = random.sample(codewords, min(num_remove, len(codewords)))
            for cw in to_remove:
                update_counts(cw, -1)
                codewords.remove(cw)
                codewords_set.remove(cw)
                
            # Repair: Greedily add codewords to cover uncovered words
            while True:
                uncovered = [w for w in range(total) if counts[w] == 0]
                if not uncovered:
                    break
                # Pick best candidate
                best_cand = -1
                best_count = -1
                # Optimization: Only check candidates that cover at least one uncovered word
                # We can just check all, it's fast enough
                for i in range(total):
                    count = 0
                    for w in neighbors[i]:
                        if counts[w] == 0:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_cand = i
                        if count == len(uncovered):
                            break
                if best_cand == -1:
                    break
                update_counts(best_cand, 1)
                codewords.append(best_cand)
                codewords_set.add(best_cand)
                
            # After repair, loop back to check for redundancies
            # We don't set improved=True explicitly, but the loop continues
            # and step 1 will check for redundancies

        # Return best found
        if len(best_codewords) <= len(codewords):
            final_codewords = best_codewords
        else:
            final_codewords = codewords
            
        # Convert to list of lists
        result = []
        for idx in final_codewords:
            word = [(idx // powers[p]) % q for p in range(n)]
            result.append(word)
            
        # Verify
        covered = np.zeros(total, dtype=bool)
        for cw in final_codewords:
            for w in neighbors[cw]:
                covered[w] = True
        if covered.all():
            return result
        else:
            # Fallback to greedy
            greedy_result = greedy_init()
            result = []
            for idx in greedy_result:
                word = [(idx // powers[p]) % q for p in range(n)]
                result.append(word)
            return result

    # Fallback for other cases
    return [[0] * n]
