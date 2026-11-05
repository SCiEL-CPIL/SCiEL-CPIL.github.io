import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    # Fallback for general cases
    if q != 3 or n != 8 or radius != 1:
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    start_time = time.time()
    deadline = start_time + 115  # Safe margin before 120s kill (accounting for overhead)

    # Precompute neighbors as numpy arrays for fast indexing
    # neighbors[i] is a numpy array of ints (indices of words within distance 1 of i)
    neighbors_np = [None] * total_words
    # Also precompute the "difference" indices for fast updates? No, just use the array.
    
    # Helper to convert index to word (list of 8 ints)
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    # Precompute neighbors
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            orig_val = w[i]
            for val in range(3):
                if val != orig_val:
                    nw = w[:]
                    nw[i] = val
                    new_idx = 0
                    for v in nw:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors_np[idx] = np.array(nbrs, dtype=np.int32)

    # 1. Greedy Construction (Fast, deterministic-ish)
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_set = set()
    code_list = []
    
    # Greedy: pick word with max uncovered neighbors
    # Use randomization in tie-breaking or candidate selection to escape local optima?
    # For initial construction, a simple greedy is fine.
    while np.any(uncovered_mask) and time.time() < deadline - 20:
        # To speed up, we can sample candidates if the pool is large, 
        # but 6561 is small enough that scanning is okay if vectorized?
        # np.sum(uncovered_mask[neighbors_np[i]]) is O(17) per i. 6561*17 ops ~ 100k ops. Very fast.
        gains = np.array([np.sum(uncovered_mask[neighbors_np[i]]) for i in range(total_words)])
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        uncovered_mask[neighbors_np[best_idx]] = False

    # Ensure validity (in case loop broke early due to time, though unlikely to leave holes if greedy was complete)
    if np.any(uncovered_mask):
        # Force add covers for remaining holes
        holes = np.where(uncovered_mask)[0]
        for h in holes:
            # Pick a neighbor of h not in code_set? Or just h itself?
            # If h is not covered, it's not in code_set (or its neighbors don't cover it? No, h covers h).
            # Wait, if h is uncovered, no codeword in code_set is in neighbors[h].
            # So h itself is not in code_set.
            if h not in code_set:
                code_set.add(h)
                code_list.append(h)
                uncovered_mask[neighbors_np[h]] = False

    # 2. Local Search: Minimize
    # Maintain cover_count: how many times each word is covered
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_np[c]] += 1

    def is_redundant(c):
        # A codeword c is redundant if all its neighbors are covered by at least 2 words
        return np.min(cover_count[neighbors_np[c]]) >= 2

    def remove_redundant():
        changed = False
        # Iterate multiple times until stable
        for _ in range(10): # Limit iterations to prevent infinite loop if logic is flawed
            progress = False
            # Iterate in reverse to allow popping
            for i in range(len(code_list) - 1, -1, -1):
                if time.time() > deadline - 15:
                    return changed
                c = code_list[i]
                if is_redundant(c):
                    code_list.pop(i)
                    code_set.discard(c)
                    cover_count[neighbors_np[c]] -= 1
                    changed = True
                    progress = True
            if not progress:
                break
        return changed

    # Initial cleanup
    remove_redundant()

    # Local Search Loop: 1-Swap (Kick and Fill)
    last_improvement = time.time()
    
    while time.time() < deadline - 15:
        # Try to remove redundant words
        if remove_redundant():
            last_improvement = time.time()
            continue
            
        # 1. Try 1-Swap
        # Select victim: we want to remove a word that causes minimal damage.
        # Damage = number of neighbors that would become uncovered (cover_count == 1).
        # Ideally damage == 1, then we can replace with a word covering that hole.
        
        # Sample candidates for victim to save time
        if len(code_list) > 20:
            victims = random.sample(code_list, 20)
        else:
            victims = code_list[:]
            random.shuffle(victims)
        
        best_victim = -1
        best_hole = -1
        best_victim_damage = 999
        
        for c in victims:
            nbrs_c = neighbors_np[c]
            # Find neighbors with cover_count == 1
            holes_mask = cover_count[nbrs_c] == 1
            damage = np.sum(holes_mask)
            
            if damage < best_victim_damage:
                best_victim_damage = damage
                best_victim = c
                if damage == 1:
                    # Find the specific hole
                    hole_idx = np.where(holes_mask)[0]
                    if len(hole_idx) > 0:
                        best_hole = nbrs_c[hole_idx[0]]
                    # If damage == 1, we can't get better, so break? 
                    # Actually, we might want to check if the hole is "easy" to fill.
                    # But for now, damage=1 is the best structural property.
                    break
                elif damage == 0:
                    # This shouldn't happen if is_redundant check passed? 
                    # If damage 0, it's redundant and should have been removed.
                    break

        if best_victim != -1 and best_victim_damage == 1:
            # Perform Kick
            c = best_victim
            h = best_hole
            
            # Temporarily remove c
            cover_count[neighbors_np[c]] -= 1
            code_list.remove(c)
            code_set.discard(c)
            
            # Find replacement for h
            # Candidates must be in neighbors[h] and NOT in code_set
            cand_pool = [x for x in neighbors_np[h] if x not in code_set]
            
            if cand_pool:
                # We want to pick a candidate that adds "redundancy" or allows future removals.
                # Score: How many neighbors of the candidate are already covered (cover_count >= 1)?
                # This indicates overlap. High overlap is good for future removals.
                best_cand = -1
                best_score = -1
                
                # Sample if pool is large (it's max 17, so just iterate)
                for cand in cand_pool:
                    # Score: number of neighbors of cand that are covered by >= 1 word (excluding the hole h which is now 0? No, h is 0)
                    # Actually, we want to maximize the count of neighbors that are already well-covered?
                    # Or simply: if we add cand, does it make any *other* word redundant?
                    # Let's stick to: Score = sum(1 for nb in neighbors[cand] if cover_count[nb] >= 1)
                    # Note: cover_count[h] is 0.
                    score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_cand != -1:
                    # Add candidate
                    code_list.append(best_cand)
                    code_set.add(best_cand)
                    cover_count[neighbors_np[best_cand]] += 1
                    last_improvement = time.time()
                    continue # Restart to check for new redundancies
            
            # If no replacement found or failed, revert
            code_list.append(c)
            code_set.add(c)
            cover_count[neighbors_np[c]] += 1

        # 2. Try 2-Swap if stuck
        if time.time() - last_improvement > 1.0:
            # Select two random victims
            if len(code_list) >= 2:
                # Sample a few pairs
                found_improvement = False
                for _ in range(5): # Try 5 random pairs
                    if time.time() > deadline - 15:
                        break
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]
                    
                    # Temporarily remove both
                    cover_count[neighbors_np[c1]] -= 1
                    cover_count[neighbors_np[c2]] -= 1
                    
                    # Find holes (words with cover_count == 0)
                    holes_mask = cover_count == 0
                    holes = np.where(holes_mask)[0]
                    
                    if len(holes) <= 2:
                        # We need to find words to cover these holes.
                        # Candidates are neighbors of the holes that are not in code_set.
                        cand_set = set()
                        for h in holes:
                            for nb in neighbors_np[h]:
                                if nb not in code_set:
                                    cand_set.add(nb)
                        
                        cand_list = list(cand_set)
                        
                        if len(holes) == 0:
                            # We removed 2 and created 0 holes? That means we can keep them removed!
                            # This is an improvement of size 2.
                            # But wait, if holes==0, it means c1 and c2 were redundant together?
                            # is_redundant checks individual. Two words can be jointly redundant.
                            # So we just keep them out.
                            last_improvement = time.time()
                            found_improvement = True
                            break
                        elif len(holes) == 1:
                            # Need 1 word to cover the hole.
                            h = holes[0]
                            # Find best candidate from cand_list that covers h
                            best_cand = -1
                            best_score = -1
                            for cand in cand_list:
                                if h in neighbors_np[cand]: # It must, by construction
                                    score = np.sum(cover_count[neighbors_np[cand]] >= 1)
                                    if score > best_score:
                                        best_score = score
                                        best_cand = cand
                            if best_cand != -1:
                                # Add 1 word, removed 2. Improvement of 1.
                                code_list.append(best_cand)
                                code_set.add(best_cand)
                                cover_count[neighbors_np[best_cand]] += 1
                                last_improvement = time.time()
                                found_improvement = True
                                break
                        elif len(holes) == 2:
                            # Need 2 words to cover 2 holes.
                            h1, h2 = holes
                            # Find pair of candidates (cand_a, cand_b) such that cand_a covers h1/h2 and cand_b covers h1/h2?
                            # Actually, they just need to cover the set of holes.
                            # Simple heuristic: Find cand_a covering h1, cand_b covering h2.
                            # Or one candidate covering both?
                            
                            best_pair = None
                            best_pair_score = -1
                            
                            # Limit search space
                            # Candidates covering h1
                            cands_h1 = [c for c in cand_list if h1 in neighbors_np[c]]
                            # Candidates covering h2
                            cands_h2 = [c for c in cand_list if h2 in neighbors_np[c]]
                            
                            # Try combinations
                            # If len is small, try all pairs
                            if len(cands_h1) * len(cands_h2) < 100:
                                for c1_new in cands_h1:
                                    for c2_new in cands_h2:
                                        if c1_new == c2_new:
                                            # One word covers both
                                            if h1 in neighbors_np[c1_new] and h2 in neighbors_np[c1_new]:
                                                # Add 1 word, remove 2. Improvement 1.
                                                # Score it
                                                score = np.sum(cover_count[neighbors_np[c1_new]] >= 1)
                                                if score > best_pair_score:
                                                    best_pair_score = score
                                                    best_pair = (c1_new, None)
                                        else:
                                            # Check if they cover holes
                                            # c1_new covers h1 (by def). Does it cover h2?
                                            # c2_new covers h2 (by def). Does it cover h1?
                                            # We just need union to cover {h1, h2}.
                                            # c1_new covers h1. c2_new covers h2.
                                            # So yes, they cover the holes.
                                            score1 = np.sum(cover_count[neighbors_np[c1_new]] >= 1)
                                            score2 = np.sum(cover_count[neighbors_np[c2_new]] >= 1)
                                            if score1 + score2 > best_pair_score:
                                                best_pair_score = score1 + score2
                                                best_pair = (c1_new, c2_new)
                            
                            if best_pair:
                                c1_new, c2_new = best_pair
                                code_list.append(c1_new)
                                code_set.add(c1_new)
                                cover_count[neighbors_np[c1_new]] += 1
                                if c2_new is not None:
                                    code_list.append(c2_new)
                                    code_set.add(c2_new)
                                    cover_count[neighbors_np[c2_new]] += 1
                                last_improvement = time.time()
                                found_improvement = True
                                break
                    
                    if not found_improvement:
                        # Revert
                        code_list.extend([c1, c2])
                        code_set.update([c1, c2])
                        cover_count[neighbors_np[c1]] += 1
                        cover_count[neighbors_np[c2]] += 1

        # Break if no progress for a while?
        # The loop condition handles time.

    # Final cleanup
    remove_redundant()

    # Final Validity Check
    # Ensure all words are covered
    final_cover = np.zeros(total_words, dtype=bool)
    for c in code_list:
        final_cover[neighbors_np[c]] = True
        
    if not np.all(final_cover):
        # Fallback: add missing covers
        uncovered_idx = np.where(~final_cover)[0]
        for idx in uncovered_idx:
             if idx not in code_set:
                 code_list.append(idx)
                 code_set.add(idx)
             else:
                 # This implies a bug in the cover count logic or final_cover logic
                 # For safety, find any neighbor
                 for nb in neighbors_np[idx]:
                     if nb not in code_set:
                         code_list.append(nb)
                         code_set.add(nb)
                         break
                 else:
                     # All neighbors in code_set? Then add idx itself? But it is in code_set.
                     # This is a logical contradiction if final_cover is false.
                     pass

    return [idx_to_word(c) for c in code_list]

# Helper for 2-swap logic
import itertools
