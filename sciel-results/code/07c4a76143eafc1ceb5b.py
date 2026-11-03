import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Leave a safety margin of 8 seconds for cleanup and return
    deadline = start_time + 112.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    # Precompute neighbors
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)

    q_powers = [q ** i for i in range(n)]

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

    # Precompute neighbor sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 25.0:
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

    def ca_reorganize(cw, time_limit):
        """Cellular-automaton-like global reorganization.

        Synchronously remove every in-code word whose own neighborhood
        contains a coverage deficit (deficit = max(0, 1 - count)), then
        greedily rebuild coverage from the deficit set. This performs a
        global structural reorganization that can escape single-swap
        local minima of the exchange-based local search.
        """
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        in_code = np.zeros(total_words, dtype=bool)
        for ci in cw:
            in_code[ci] = True

        for _ in range(6):
            if time.time() > time_limit - 8.0:
                break

            count = np.zeros(total_words, dtype=np.int32)
            for ci in cw:
                count[neighbors[ci]] += 1

            deficit = (1 - count) > 0  # words needing coverage
            if not deficit.any():
                break

            # Synchronous CA update: a codeword is "unstable" if any word in
            # its own neighborhood has a deficit (it is the only thing keeping
            # that word covered). Remove all unstable codewords at once.
            remove = np.zeros(total_words, dtype=bool)
            for ci in cw:
                if deficit[neighbors[ci]].any():
                    remove[ci] = True

            new_cw = [ci for ci in cw if not remove[ci]]

            # Rebuild coverage from the remaining deficit set (greedy).
            covered = np.zeros(total_words, dtype=bool)
            for ci in new_cw:
                covered[neighbors[ci]] = True
            rem_deficit = deficit & ~covered
            if rem_deficit.any():
                gain = np.full(total_words, n_neighbors, dtype=np.int32)
                for ci in new_cw:
                    for u in neighbors[ci]:
                        gain[u] -= 1
                remaining = int(rem_deficit.sum())
                while remaining > 0:
                    if time.time() > time_limit - 8.0:
                        break
                    best_word = int(np.argmax(gain))
                    if gain[best_word] == 0:
                        unc = np.where(rem_deficit)[0]
                        if len(unc) == 0:
                            break
                        best_word = int(unc[0])
                    new_cw.append(best_word)
                    for v in neighbors[best_word]:
                        if rem_deficit[v]:
                            rem_deficit[v] = False
                            remaining -= 1
                        for w in neighbors[v]:
                            gain[w] -= 1

            cw = new_cw
            if len(cw) < best_size:
                best_size = len(cw)
                best_cw = cw[:]

        return best_cw

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
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: Swaps
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
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        # Do not break, allow more removals in this pass

                    if len(critical) > 0:
                        first_crit = critical[0]

                        if len(critical) == 1:
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)

                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1

                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                # Do not break, allow more swaps in this pass
                        else:
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break

                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1

                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    # Do not break, allow more swaps in this pass

            # Pass 3: 2-for-1 Swap (Reduce size by 1 by replacing 2 codewords with 1)
            if not improved and len(cw) > 2 and random.random() < 0.5:
                # Sample a few pairs to avoid O(N^2) cost
                num_samples = 20
                for _ in range(num_samples):
                    if time.time() > time_limit:
                        break
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # Words covered only by c1 or c2 (or both, but no other)
                    # We need to find words that will become uncovered.
                    # Uncovered = (N(c1) U N(c2)) \ (N(C \ {c1, c2}))
                    # count[u] is current coverage count.
                    # If we remove c1 and c2, new_count[u] = count[u] - (1 if u in N(c1) else 0) - (1 if u in N(c2) else 0)
                    # Deficit if new_count[u] < 1 => count[u] < 1 + (u in N(c1)) + (u in N(c2))
                    
                    # Efficient check:
                    # Create a temporary mask of words that are critical for c1 or c2
                    # Actually, let's just compute the deficit set directly for the union of neighbors.
                    
                    # We only care about neighbors of c1 and c2.
                    # Let U = N(c1) U N(c2)
                    # For u in U, if count[u] == 1 and (u in N(c1) or u in N(c2)) and not (u in N(c1) and u in N(c2) and count[u]==2? No.)
                    # Simpler:
                    # new_count[u] = count[u] - (1 if u in N(c1) else 0) - (1 if u in N(c2) else 0)
                    # Deficit if new_count[u] < 1.
                    
                    # Since we only remove 2, the deficit set is small.
                    # We can iterate over neighbors of c1 and c2.
                    deficit_set = set()
                    for u in neighbors[c1]:
                        # Count how many of {c1, c2} cover u
                        covers = 0
                        if u in neighbor_sets[c2]:
                            covers += 1
                        covers += 1 # c1 always covers u
                        if count[u] - covers < 1:
                            deficit_set.add(u)
                    
                    for u in neighbors[c2]:
                        if u in deficit_set:
                            continue
                        covers = 0
                        if u in neighbor_sets[c1]:
                            covers += 1
                        covers += 1 # c2 always covers u
                        if count[u] - covers < 1:
                            deficit_set.add(u)
                    
                    if not deficit_set:
                        # We can remove both and add nothing? No, we need to add 1 to keep size -1? 
                        # If deficit is empty, we can remove both and add 0? 
                        # Wait, if deficit is empty, it means all words in N(c1) U N(c2) are covered by other codewords.
                        # So we can just remove c1 and c2. Size reduces by 2.
                        # This is a "2-for-0" which is even better.
                        # But usually we replace with 1 to maintain coverage structure or just remove.
                        # If we remove both, we must ensure validity. If deficit is empty, validity is maintained.
                        # Let's try to find a single codeword that covers deficit_set.
                        # If deficit_set is empty, we can pick any codeword? No, we want to minimize size.
                        # If deficit_set is empty, we can remove both and add nothing.
                        # But we must ensure the new codeword (if added) is not already in code.
                        # If we add nothing, size reduces by 2.
                        
                        # Let's check if we can just remove them.
                        # If we remove them, do we break anything? No, deficit is empty.
                        # So we can remove c1 and c2.
                        # However, the problem asks for a covering code. If we remove them, is it still a covering code? Yes.
                        # So we can reduce size by 2.
                        
                        # But wait, is it possible that removing them leaves some word uncovered?
                        # Deficit set calculation:
                        # For u in N(c1) U N(c2), if count[u] - covers_from_removed < 1, it's deficit.
                        # If deficit_set is empty, all u in N(c1) U N(c2) have count[u] >= covers_from_removed.
                        # So they are still covered by remaining codewords.
                        # So yes, we can remove both.
                        
                        # Let's execute 2-for-0.
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] -= 1
                        in_code[c1] = False
                        in_code[c2] = False
                        # Remove from cw list
                        # Careful with indices
                        if idx1 < idx2:
                            cw.pop(idx2)
                            cw.pop(idx1)
                        else:
                            cw.pop(idx1)
                            cw.pop(idx2)
                        improved = True
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break # Restart loop to update indices/counts? Or continue?
                        # Since we modified cw, indices are invalid. Break to restart pass.

                    # If deficit_set is not empty, try to find a single codeword covering all of it.
                    # Candidate must be in intersection of neighbor_sets[u] for u in deficit_set.
                    # And not in_code.
                    candidate_set = neighbor_sets[list(deficit_set)[0]].copy()
                    for u in list(deficit_set)[1:]:
                        candidate_set &= neighbor_sets[u]
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        valid_cands = [c for c in candidate_set if not in_code[c]]
                        if valid_cands:
                            cand = random.choice(valid_cands)
                            # Remove c1, c2
                            for u in neighbors[c1]:
                                count[u] -= 1
                            for u in neighbors[c2]:
                                count[u] -= 1
                            in_code[c1] = False
                            in_code[c2] = False
                            
                            # Add cand
                            for u in neighbors[cand]:
                                count[u] += 1
                            in_code[cand] = True
                            
                            # Update cw list
                            # Remove c1, c2, add cand
                            if idx1 < idx2:
                                cw.pop(idx2)
                                cw.pop(idx1)
                            else:
                                cw.pop(idx1)
                                cw.pop(idx2)
                            cw.append(cand)
                            
                            improved = True
                            if len(cw) < best_size:
                                best_size = len(cw)
                                best_cw = cw[:]
                            break # Restart loop

            # Perturbation: random swap of two codewords to escape local min
            if not improved and len(cw) > 2 and random.random() < 0.3:
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
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Aggressive perturbation
            if not improved and random.random() < 0.1:
                if len(cw) > 5:
                    idx = random.randint(0, len(cw) - 1)
                    c_remove = cw[idx]
                    covered_words = np.where(count > 0)[0]
                    if len(covered_words) > 0:
                        c_add = int(covered_words[random.randint(0, len(covered_words) - 1)])
                        temp_count = count.copy()
                        for u in neighbors[c_remove]:
                            temp_count[u] -= 1
                        for u in neighbors[c_add]:
                            temp_count[u] += 1
                        if np.min(temp_count) >= 1:
                            for u in neighbors[c_remove]:
                                count[u] -= 1
                            for u in neighbors[c_add]:
                                count[u] += 1
                            in_code[c_remove] = False
                            in_code[c_add] = True
                            cw[idx] = c_add
                            improved = True

        return best_cw

    def finish(cw):
        result = []
        for ci in cw:
            word = []
            val = ci
            for pos in range(n - 1, -1, -1):
                word.append(val % q)
                val //= q
            word.reverse()
            result.append(word)
        return result

    # Phase 1: Greedy Set Cover
    codewords = greedy_init(42)

    # Phase 2: Local Search
    best_cw = local_search(codewords, deadline - 30.0)

    # Phase 3: CA-like global reorganization (only if it keeps us valid & smaller)
    if time.time() < deadline - 15.0:
        ca_cw = ca_reorganize(best_cw, deadline - 12.0)
        if len(ca_cw) < len(best_cw):
            best_cw = ca_cw

    # Phase 4: Second restart with a different seed if time permits
    if time.time() < deadline - 45.0:
        codewords2 = greedy_init(1337)
        ls2 = local_search(codewords2, deadline - 20.0)
        if len(ls2) < len(best_cw):
            best_cw = ls2
        if time.time() < deadline - 12.0:
            ca2 = ca_reorganize(best_cw, deadline - 10.0)
            if len(ca2) < len(best_cw):
                best_cw = ca2

    return finish(best_cw)
