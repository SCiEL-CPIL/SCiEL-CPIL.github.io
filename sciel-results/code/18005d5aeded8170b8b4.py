import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        all_words.append(tuple(reversed(w)))
        
    # Precompute powers of q for index calculation
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute neighbors for distance <= 1 (for coverage check)
    # For q=3, n=8, radius=1: 17 neighbors.
    
    neighbors_r1 = [[] for _ in range(total_words)]
    
    # Generate neighbors efficiently
    for i, w in enumerate(all_words):
        neighbors_r1[i].append(i)
        
        # Dist 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig: continue
                diff = (new_val - orig) * q_powers[pos]
                j = i + diff
                neighbors_r1[i].append(j)

    # Convert to sets for faster lookup
    neighbors_r1_set = [set(lst) for lst in neighbors_r1]

    # Step 1: Greedy Set Cover Initialization
    # This provides a valid starting point, likely larger than optimal but valid.
    # We pick the word that covers the most *currently uncovered* words.
    
    C = set()
    uncovered = [True] * total_words
    uncovered_count = total_words
    
    # To speed up greedy, we can precompute degrees or just iterate.
    # With 6561 words, iterating 6561 times * 6561 checks is ~43M ops, feasible in Python if optimized?
    # Maybe too slow in pure Python loop. 
    # Let's use a faster heuristic: Randomized Greedy or just start with a random subset and repair?
    # Actually, for n=8, q=3, a good initial guess is around 500-600.
    # Let's try a simple greedy but limit time.
    
    # Alternative Initialization: 
    # Use the structure of the space. 
    # Or, start with all words and remove redundant ones? 
    # Start with C = all words. Then remove redundant ones. This is "Maximal Independent Set" on the complement?
    # If we start with all words, we can remove any word if it doesn't uniquely cover anything.
    # This will converge to a minimal covering code (irredundant).
    # Size will be around 1000+. Then local search reduces it.
    
    C = set(range(total_words))
    cover_count = [0] * total_words
    for c in C:
        for nb in neighbors_r1[c]:
            cover_count[nb] += 1
            
    # Remove redundant centers
    # A center is redundant if it doesn't uniquely cover any word.
    # i.e., for all nb in N(c), cover_count[nb] > 1.
    
    changed = True
    while changed:
        changed = False
        # Iterate in random order to break symmetries
        centers_list = list(C)
        random.shuffle(centers_list)
        for c in centers_list:
            if time.time() > 5.0: # Time limit for init phase
                break
            redundant = True
            for nb in neighbors_r1[c]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for nb in neighbors_r1[c]:
                    cover_count[nb] -= 1
                C.remove(c)
                changed = True
                if time.time() > 5.0:
                    break

    # Now C is a minimal covering code (irredundant).
    # Size is likely ~1000-1200. Optimal is 486.
    
    start_time = time.time()
    max_time = 115.0
    
    # Local Search
    # We will try:
    # 1. Single removal (if redundant appears due to previous moves)
    # 2. Single swap (replace c with w)
    # 3. Double swap (replace c1, c2 with w1, w2) - potentially reducing size if w1==w2 or one becomes redundant?
    #    Actually, a "move" that reduces size is: replace 2 centers with 1 center that covers their union of critical neighbors.
    #    Let's call this "Merge".
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # 1. Reduction Phase
        centers_list = list(C)
        random.shuffle(centers_list)
        for c_idx in centers_list:
            if time.time() - start_time > max_time:
                break
            redundant = True
            for nb in neighbors_r1[c_idx]:
                if cover_count[nb] == 1:
                    redundant = False
                    break
            if redundant:
                for nb in neighbors_r1[c_idx]:
                    cover_count[nb] -= 1
                C.remove(c_idx)
                improved = True
                break # Restart reduction phase as counts changed
                
        if time.time() - start_time > max_time:
            break
            
        if improved:
            continue
            
        # 2. Move Phase
        # Try to find a move that improves the solution (reduces size or enables reduction).
        # Since we are in a minimal code, we can't just remove.
        # We can swap c with w. This keeps size same.
        # We can try to replace {c1, c2} with {w}. This reduces size by 1.
        
        # Strategy: Randomly pick a center c.
        # Identify its "critical" neighbors: nb in N(c) with cover_count[nb] == 1.
        # If we remove c, these nb become uncovered.
        # We need to add a set of centers W such that W covers all critical nb.
        # If |W| < 1 (impossible) or if W allows further reduction, it's good.
        # Standard move: Replace c with w.
        # Condition: w covers all critical nb of c.
        # After swap, check if any other center becomes redundant.
        
        # Let's implement a more powerful move: 
        # Pick 1 or 2 centers to remove.
        # Find a set of 1 or 2 centers to add that covers the exposed critical neighbors.
        # If (num_added < num_removed) -> Improvement.
        # If (num_added == num_removed) -> Neutral move, but might enable future improvements.
        
        # To keep it simple and effective, let's try:
        # A. Replace 1 center c with 1 center w. (Neutral move, explores space)
        # B. Replace 2 centers c1, c2 with 1 center w. (Improvement move)
        
        # Try B first (Improvement)
        # Pick c1, c2 randomly.
        # Critical nb = (N(c1) \ N(C\{c1})) U (N(c2) \ N(C\{c1,c2}))
        # Actually, critical nb are those with cover_count == 1 after removing c1, c2.
        # Let's simulate removal of c1, c2.
        
        # This simulation is expensive if done for all pairs.
        # Let's pick a random c1.
        # Critical nb of c1 (with respect to C) are those with count 1.
        # If we remove c1, we need to cover these.
        # If we also remove c2, we might cover more, but we need to cover the union of critical nb of c1 and c2 that are NOT covered by other centers.
        
        # Simpler approach:
        # 1. Pick c1.
        # 2. Find critical nb of c1.
        # 3. Try to find w such that w covers critical nb.
        #    If found, swap c1 -> w.
        #    Then check for redundancy in C.
        
        # Let's enhance the swap to check if the swap leads to a reducible state.
        
        moves_attempted = 0
        max_moves_per_iter = 50
        while moves_attempted < max_moves_per_iter and (time.time() - start_time) < max_time:
            moves_attempted += 1
            
            # Try a swap
            c_idx = random.choice(list(C))
            
            # Find critical neighbors of c_idx
            critical_nb = []
            for nb in neighbors_r1[c_idx]:
                if cover_count[nb] == 1:
                    critical_nb.append(nb)
                    
            if not critical_nb:
                # Should have been caught in reduction phase
                for nb in neighbors_r1[c_idx]:
                    cover_count[nb] -= 1
                C.remove(c_idx)
                improved = True
                break
                
            # Candidates for w must cover all critical_nb.
            # Intersection of N(nb) for nb in critical_nb.
            # Pick a random critical nb to start candidate pool.
            nb0 = random.choice(critical_nb)
            candidates = list(neighbors_r1_set[nb0])
            random.shuffle(candidates)
            
            found_swap = False
            for w_idx in candidates:
                if w_idx == c_idx:
                    continue
                # Check if w_idx covers all critical_nb
                ok = True
                for nb in critical_nb:
                    if nb not in neighbors_r1_set[w_idx]:
                        ok = False
                        break
                if ok:
                    # Perform swap
                    # Remove c_idx
                    for nb in neighbors_r1[c_idx]:
                        cover_count[nb] -= 1
                    C.remove(c_idx)
                    
                    # Add w_idx
                    if w_idx not in C:
                        for nb in neighbors_r1[w_idx]:
                            cover_count[nb] += 1
                        C.add(w_idx)
                    # else w_idx was already in C? 
                    # If w_idx in C, then we just removed c_idx. 
                    # But w_idx covers critical_nb. 
                    # If w_idx in C, then critical_nb were already covered by w_idx.
                    # So cover_count[nb] for nb in critical_nb was >= 1 from w_idx.
                    # If cover_count[nb] == 1, it must have been from c_idx?
                    # No, if w_idx in C and w_idx covers nb, then cover_count[nb] >= 1 from w_idx.
                    # If cover_count[nb] == 1, then w_idx must be the one covering it?
                    # Or c_idx?
                    # If w_idx in C and w_idx != c_idx, and w_idx covers nb, then cover_count[nb] >= 1.
                    # If cover_count[nb] == 1, and c_idx covers nb, then w_idx must NOT cover nb?
                    # Contradiction.
                    # So if w_idx covers critical_nb, and cover_count[nb]==1, then w_idx cannot be in C (unless w_idx==c_idx, which we skip).
                    # So w_idx is always new to C.
                    
                    found_swap = True
                    improved = True
                    break
                    
            if found_swap:
                break
                
            # If no swap found for this c_idx, try another c_idx in next loop iteration?
            # The while loop continues.
            
        if improved:
            continue
            
        # If no simple swap found, try a "Double Swap" or "Merge"
        # Replace c1, c2 with w.
        # This is an improvement if w covers all critical nb of c1 and c2 (combined).
        
        # Pick c1, c2
        if len(C) >= 2:
            c1 = random.choice(list(C))
            c2 = random.choice([c for c in C if c != c1])
            
            # Simulate removal of c1, c2
            # Critical nb are those in N(c1) U N(c2) that have count 1 (from c1 or c2)
            # Actually, we need to find the set of nb that become uncovered.
            # nb is uncovered if cover_count[nb] == 1 and nb in N(c1) OR cover_count[nb] == 1 and nb in N(c2) AND (nb not in N(c1) or cover_count[nb] < 2?)
            # Let's just calculate the set of "exposed" neighbors.
            
            exposed = set()
            for nb in neighbors_r1[c1]:
                if cover_count[nb] == 1:
                    exposed.add(nb)
            for nb in neighbors_r1[c2]:
                if cover_count[nb] == 1:
                    exposed.add(nb)
                elif cover_count[nb] == 2:
                    # If nb is in N(c1) and N(c2), and count is 2, removing both makes it 0.
                    if nb in neighbors_r1[c1]:
                        exposed.add(nb)
                        
            if not exposed:
                # Both redundant? Should have been caught.
                # Remove c1
                for nb in neighbors_r1[c1]:
                    cover_count[nb] -= 1
                C.remove(c1)
                # Remove c2
                for nb in neighbors_r1[c2]:
                    cover_count[nb] -= 1
                C.remove(c2)
                improved = True
                break
                
            # Find w that covers all exposed nb.
            # Intersection of N(nb) for nb in exposed.
            # Pick random exposed nb.
            nb0 = random.choice(list(exposed))
            candidates = list(neighbors_r1_set[nb0])
            random.shuffle(candidates)
            
            found_merge = False
            for w_idx in candidates:
                if w_idx == c1 or w_idx == c2:
                    continue
                ok = True
                for nb in exposed:
                    if nb not in neighbors_r1_set[w_idx]:
                        ok = False
                        break
                if ok:
                    # Perform merge: Remove c1, c2, Add w_idx
                    for nb in neighbors_r1[c1]:
                        cover_count[nb] -= 1
                    C.remove(c1)
                    
                    for nb in neighbors_r1[c2]:
                        cover_count[nb] -= 1
                    C.remove(c2)
                    
                    for nb in neighbors_r1[w_idx]:
                        cover_count[nb] += 1
                    C.add(w_idx)
                    
                    found_merge = True
                    improved = True
                    break
                    
            if found_merge:
                break

    # Convert C to list of lists
    result = [list(all_words[c]) for c in C]
    return result
