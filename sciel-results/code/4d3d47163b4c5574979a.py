import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for general cases (not the specific task, but required for interface)
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
    deadline = start_time + 115  # Safe margin before 150s kill

    # Precompute neighbors as a list of lists of ints
    # This is fast enough for n=8
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        nbrs = [idx]
        # For each position, add the 2 other values
        for i in range(8):
            orig_val = w[i]
            for val in range(3):
                if val != orig_val:
                    # Calculate new index efficiently
                    # diff = val - orig_val * 3^i
                    # new_idx = idx + (val - orig_val) * (3 ** i)
                    # But careful with negative diffs and borrows? 
                    # Actually, base-3 arithmetic:
                    # Let's just use the list conversion for safety and simplicity, it's only 6561*17 ops
                    nw = w[:]
                    nw[i] = val
                    new_idx = 0
                    for v in nw:
                        new_idx = new_idx * 3 + v
                    nbrs.append(new_idx)
        neighbors[idx] = nbrs

    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return w

    # 1. Greedy Construction
    uncovered_mask = np.ones(total_words, dtype=bool)
    code_set = set()
    code_list = []
    
    # We can pre-compute gains for speed? 
    # Greedy: pick word with max uncovered neighbors
    # To speed up, we can sample candidates
    while np.any(uncovered_mask) and time.time() < deadline - 30:
        best_idx = -1
        best_gain = -1
        
        # Sample candidates if set is large
        candidates_pool = list(range(total_words))
        if len(code_set) > 100:
             candidates_pool = [i for i in range(total_words) if i not in code_set]
        
        # Sample 200 candidates for greedy step to balance speed and quality
        if len(candidates_pool) > 200:
            candidates_pool = random.sample(candidates_pool, 200)
            
        for i in candidates_pool:
            gain = np.sum(uncovered_mask[neighbors[i]])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        code_set.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered_mask[nbr] = False

    # Ensure validity
    while np.any(uncovered_mask):
         best_idx = -1
         best_gain = -1
         for i in range(total_words):
             if i in code_set: continue
             gain = np.sum(uncovered_mask[neighbors[i]])
             if gain > best_gain:
                 best_gain = gain
                 best_idx = i
         if best_idx == -1: break
         code_set.add(best_idx)
         code_list.append(best_idx)
         for nbr in neighbors[best_idx]:
             uncovered_mask[nbr] = False

    code_list = list(code_set)
    
    # 2. Local Search: Minimize
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors[c]] += 1

    def is_redundant(c):
        # A codeword c is redundant if all its neighbors are covered by at least 2 words
        # i.e., min(cover_count[neighbors[c]]) >= 2
        return np.min(cover_count[neighbors[c]]) >= 2

    def remove_redundant():
        changed = False
        # Iterate backwards to allow safe popping
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 10:
                return changed
            c = code_list[i]
            if is_redundant(c):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors[c]] -= 1
                changed = True
                # Restart scan? Or continue? Continuing is faster, but might miss chains.
                # Let's continue in this pass, and outer loop will call again if needed.
        return changed

    # Initial cleanup
    while remove_redundant():
        pass

    # Local Search Loop: 1-Swap and 2-Swap
    # Heuristic for picking victim:
    # We want to remove a word that is "most redundant".
    # Score = min(cover_count[neighbors[c]]). Higher is better.
    # If min >= 2, it's redundant (handled above).
    # If min == 1, removing it creates exactly 1 uncovered word.
    # If min == 0, it's essential (shouldn't happen in valid cover).
    
    swap_count = 0
    last_improvement = time.time()
    
    while time.time() < deadline - 10:
        # Try to remove redundant words
        if remove_redundant():
            continue
            
        # 1. Try 1-Swap (Kick and Fill)
        # Select victim with highest redundancy score (closest to being redundant)
        # Score(c) = min(cover_count[neighbors[c]])
        # We prefer c where min count is 1 (creating 1 hole) or high if >1 (but those are removed)
        # Actually, if min is 1, we have exactly 1 word to cover.
        # Let's pick the one with the fewest "unique" contributions?
        # Simple heuristic: Pick c that minimizes the number of words it uniquely covers.
        # Unique cover of c = {nbr in neighbors[c] | cover_count[nbr] == 1}
        
        # To speed up, we can sample candidates for the kick
        candidates = list(code_list)
        random.shuffle(candidates)
        # Take top 50 to evaluate
        candidates = candidates[:50]
        
        best_kick = -1
        best_kick_score = -1
        best_hole = -1
        
        for c in candidates:
            nbrs_c = neighbors[c]
            # Count how many words would be left uncovered if we remove c
            # This is count of nbrs where cover_count == 1
            holes = np.sum(cover_count[nbrs_c] == 1)
            if holes > 0:
                # We can only do a 1-swap if holes == 1 (or we find a word covering all holes, but if holes>1 it's harder)
                # Actually, if holes == 1, we need to find a word covering that specific hole.
                # If holes > 1, we could still do it if a single word covers all holes, but that's rare.
                # Let's prioritize holes == 1 for efficiency.
                if holes == 1:
                    # Score: We want the kick to be "cheap" to fix.
                    # How "central" is the hole? Or how many candidates cover it?
                    # Let's just pick the first one we find with holes==1 to save time?
                    # Better: Pick the one where the hole is covered by many other words (easy to fill)
                    hole_idx = np.where(cover_count[nbrs_c] == 1)[0]
                    if len(hole_idx) == 1:
                        h = nbrs_c[hole_idx[0]]
                        # How many words in the current code cover h? (cover_count[h] == 1, so only c does)
                        # We need a NEW word.
                        # Let's just try this kick.
                        best_kick = c
                        best_hole = h
                        break # Pick the first valid simple kick to save time?
                        # Or evaluate more? 
                        # Let's evaluate a few and pick the one where the hole h has the most potential replacements.
                        # Potential replacements for h = len(neighbors[h]) - len(code_set)
                        # This is static per h. So we want h with many neighbors.
                        # All h have same number of neighbors (17).
                        # So any h is equivalent in terms of candidate pool size.
                        # So picking the first one is fine.
                        pass
        
        if best_kick != -1:
            # Perform Kick
            c = best_kick
            h = best_hole
            
            # Temporarily remove c
            cover_count[neighbors[c]] -= 1
            code_list.remove(c)
            code_set.discard(c)
            
            # Find replacement for h
            # Candidates must be in neighbors[h] and NOT in code_set
            cand_pool = [x for x in neighbors[h] if x not in code_set]
            
            # We want to pick a candidate that allows further removals.
            # Score(candidate) = sum(1 for nbr in neighbors[cand] if cover_count[nbr] == 1)
            # i.e., how many "holes" would this candidate create if we were to remove *its* unique covers later?
            # No, we want to pick a candidate that *adds* redundancy to existing words.
            # Score(candidate) = sum(1 for nbr in neighbors[cand] if cover_count[nbr] >= 2)
            
            best_cand = -1
            best_score = -1
            
            # Sample if pool is large (it's small, max 17)
            for cand in cand_pool:
                # Check if adding cand makes any word redundant?
                # Or just maximize overlap with existing covers.
                score = np.sum(cover_count[neighbors[cand]] >= 2)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            
            if best_cand != -1:
                # Add candidate
                code_list.append(best_cand)
                code_set.add(best_cand)
                cover_count[neighbors[best_cand]] += 1
                swap_count += 1
                last_improvement = time.time()
                continue # Restart loop to try removing redundancies again
            
            # If no good candidate found, revert
            code_list.append(c)
            code_set.add(c)
            cover_count[neighbors[c]] += 1
            swap_count += 1

        # 2. Try 2-Swap (if 1-swap failed or we want to try harder)
        # Only try 2-swaps if we have been stuck for a bit
        if time.time() - last_improvement > 2.0:
             # Select two victims
             # Simplified: Pick two random codewords
             if len(code_list) >= 2:
                 i1, i2 = random.sample(range(len(code_list)), 2)
                 c1 = code_list[i1]
                 c2 = code_list[i2]
                 
                 # Temporarily remove both
                 cover_count[neighbors[c1]] -= 1
                 cover_count[neighbors[c2]] -= 1
                 
                 # Find holes
                 holes_mask = cover_count == 0
                 holes = np.where(holes_mask)[0]
                 
                 if len(holes) <= 2:
                     # We need to find 2 words that cover all holes
                     # If len(holes) == 0, we just removed 2 and added 0? No, we must add 2 to keep size?
                     # Actually, if we remove 2, we should try to add 2 to keep size, or add 1 if it covers all, or add 0 if valid (improvement!).
                     
                     # If len(holes) == 0: We can remove both! (Improvement)
                     # If len(holes) == 1: Find 1 word covering it? Or 2 words? 
                     # If len(holes) == 2: Find 2 words covering them.
                     
                     # Let's try to find 2 words c1', c2' from neighbors[holes]
                     
                     candidate_pairs = []
                     # Get unique candidates from neighbors of holes
                     cand_set = set()
                     for h in holes:
                         for nb in neighbors[h]:
                             if nb not in code_set:
                                 cand_set.add(nb)
                     
                     cand_list = list(cand_set)
                     if len(cand_list) < 2:
                         # Revert
                         code_list.extend([c1, c2])
                         code_set.update([c1, c2])
                         cover_count[neighbors[c1]] += 1
                         cover_count[neighbors[c2]] += 1
                         continue
                     
                     # Sample pairs
                     if len(cand_list) > 20:
                         cand_list = random.sample(cand_list, 20)
                         
                     best_pair = None
                     best_pair_score = -1
                     
                     # To limit O(N^2), sample pairs
                     if len(cand_list) > 10:
                         pairs = random.sample(list(itertools.combinations(cand_list, 2)), min(50, len(list(itertools.combinations(cand_list, 2)))))
                     else:
                         pairs = itertools.combinations(cand_list, 2)
                     
                     for p1, p2 in pairs:
                         if p1 == p2: continue
                         # Check if p1 and p2 cover all holes
                         covered = set()
                         for nb in neighbors[p1]:
                             if cover_count[nb] == 0 or nb in holes:
                                 covered.add(nb)
                         for nb in neighbors[p2]:
                             if cover_count[nb] == 0 or nb in holes:
                                 covered.add(nb)
                                 
                         if holes.issubset(covered):
                             # Calculate score: how many redundant words do we create?
                             # Score = sum(1 for nbr in neighbors[p1]+neighbors[p2] if cover_count[nbr] + (1 if nbr in n(p1) else 0) + (1 if nbr in n(p2) else 0) >= 2)
                             # Simpler: Just accept it?
                             # Better: Maximize redundancy.
                             score = 0
                             for nb in neighbors[p1]:
                                 if cover_count[nb] >= 1: score += 1
                             for nb in neighbors[p2]:
                                 if cover_count[nb] >= 1: score += 1
                                 
                             if score > best_pair_score:
                                 best_pair_score = score
                                 best_pair = (p1, p2)
                     
                     if best_pair:
                         p1, p2 = best_pair
                         # Apply change
                         # Remove c1, c2 from list/set
                         code_list.remove(c1)
                         code_list.remove(c2)
                         code_set.discard(c1)
                         code_set.discard(c2)
                         
                         # Add p1, p2
                         code_list.extend([p1, p2])
                         code_set.update([p1, p2])
                         
                         # Update counts
                         # Note: we already subtracted c1, c2.
                         cover_count[neighbors[p1]] += 1
                         cover_count[neighbors[p2]] += 1
                         
                         swap_count += 1
                         last_improvement = time.time()
                         continue
                     
                     # Revert
                     code_list.extend([c1, c2])
                     code_set.update([c1, c2])
                     cover_count[neighbors[c1]] += 1
                     cover_count[neighbors[c2]] += 1

        # If no improvement in a long while, might want to restart or break
        # To prevent infinite loop on bad swaps, we limit time
        if time.time() > deadline - 5:
            break

    # Final cleanup
    while remove_redundant():
        pass

    return [idx_to_word(c) for c in code_list]

# Helper for 2-swap logic
import itertools
