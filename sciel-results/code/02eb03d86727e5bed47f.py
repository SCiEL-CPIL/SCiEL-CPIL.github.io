import random
import numpy as np
import itertools
import time
import heapq

def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n
    # Map word (list of ints) to index (int)
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    # neighbors[i] is a list of indices of words within distance 1 from word i
    neighbors = np.zeros((total_points, 2 * n + 1), dtype=np.int32)
    
    # Generate all words and their indices
    all_words = [idx_to_word(i) for i in range(total_points)]
    
    for i in range(total_points):
        w = all_words[i]
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                new_idx = word_to_idx(w)
                neighbors[i, count] = new_idx
                count += 1
            w[pos] = orig_val

    # 1. Greedy Set Cover Initialization
    covered = np.zeros(total_points, dtype=bool)
    chosen_set = set()
    
    start_time = time.time()
    time_limit = 140.0 # Keep 10s buffer for safety

    while not covered.all():
        if time.time() - start_time > time_limit:
            break
            
        avail_indices = np.where(~covered)[0]
        if len(avail_indices) == 0:
            break
            
        # Vectorized calculation of marginal gains for uncovered points
        # We want to pick a word that covers the most currently uncovered points.
        # Since we add words, the "uncovered" set shrinks.
        # For each candidate word c, gain(c) = |N(c) \ Covered|
        
        # Optimization: Only consider words that are not already chosen?
        # Actually, in standard greedy set cover, we pick from the whole universe 
        # of sets (words), but once picked, they are "used". 
        # However, since sets are identical to elements in terms of potential 
        # (any word can be picked), we just need to find the word with max 
        # coverage of the *remaining* uncovered set.
        
        # To speed this up, we can maintain a list of "active" candidates?
        # With 6561 words and 17 neighbors, iterating all is 6561*17 ~ 110k ops.
        # Doing this ~500 times is ~55M ops. Python loops are slow, NumPy is fast.
        
        # Let's use the vectorized approach from the seed, it's fast enough.
        # But we can be smarter: we only need to check words that *could* cover 
        # new points. Any word works.
        
        # Calculate gain for all words
        # covered is a boolean array.
        # neighbors shape (6561, 17)
        # is_covered = covered[neighbors] -> (6561, 17)
        # gains = (~is_covered).sum(axis=1)
        
        is_covered = covered[neighbors]
        gains = (~is_covered).sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
             break
             
        # Pick one of the max gain words. 
        # Tie-breaking: Randomize to explore different paths? 
        # Or pick the one with highest index for determinism? 
        # Let's pick randomly from the max gain candidates to help local search later.
        max_candidates = np.where(gains == max_gain)[0]
        best_global_idx = random.choice(max_candidates.tolist())
        
        chosen_set.add(best_global_idx)
        covered[neighbors[best_global_idx]] = True

    # 2. Submodular Local Search: Add-Best / Remove-Worst
    # State: chosen_set (set of indices)
    # Objective: Minimize |chosen_set| while maintaining coverage.
    # 
    # We use a priority queue for "Removals" and "Additions".
    # Removal Gain (Loss if removed): 
    #   If we remove c, we lose coverage for points in N(c) that are ONLY covered by c.
    #   Let count[p] be the number of chosen words covering p.
    #   If we remove c, the "damage" is the number of p in N(c) such that count[p] == 1.
    #   We want to remove the word with MINIMUM damage.
    #   
    # Addition Gain:
    #   If we add c, we cover points in N(c) that are currently UNCOVERED.
    #   This is only useful if we are currently in an invalid state (holes).
    #   However, in this specific strategy, we usually stay valid.
    #   The "Add" move is typically used in a 2-swap or if we allow temporary invalidity.
    #   
    # The prompt suggests "Add-Best / Remove-Worst" exploiting submodularity.
    # A common effective heuristic for Minimum Set Cover is:
    # 1. Try to remove a word (if it doesn't break coverage).
    # 2. If no word can be removed, try a 2-swap: Remove A, Add B, Remove C... 
    #    Or simpler: The "Add" priority queue is used when we are in a state 
    #    where we *want* to add something to enable a better removal later?
    #    
    # Actually, a pure "Remove if safe" loop is what the current program does.
    # The improvement comes from allowing **temporary invalidity** or **swaps**.
    # 
    # Let's implement a "Best-Improvement" local search with a queue.
    # We maintain `cover_count`.
    # 
    # Priority Queue for Removals (Min-Heap of (damage, word_idx)):
    #   damage(w) = number of points in N(w) with cover_count == 1.
    #   If damage == 0, we can remove w.
    #   
    # Priority Queue for Additions (Max-Heap of (gain, word_idx)):
    #   gain(w) = number of points in N(w) with cover_count == 0.
    #   This is used if we decide to allow a "swap" where we remove one and add one.
    #   But usually, we just try to remove.
    #   
    # To beat the current program (which does sequential removal + 1-swap), 
    # we need a more powerful search. 
    # Let's try **Simulated Annealing** or **Tabu Search** on the set of codewords?
    # No, the prompt specifically asks for "lazy deletion mechanism that maintains a priority queue... allowing for O(1) amortized updates".
    # This implies a specific data structure optimization for the local search.
    
    # Let's build the `cover_count` array.
    cover_count = np.zeros(total_points, dtype=np.int32)
    for c in chosen_set:
        cover_count[neighbors[c]] += 1

    # Priority Queue for Removals: (damage, word_idx)
    # We want to pop the item with smallest damage.
    removal_pq = []
    
    def calculate_damage(c):
        # Number of points in N(c) that have cover_count == 1
        # If we remove c, these points become 0 (holes).
        nbs = neighbors[c]
        return int(np.sum(cover_count[nbs] == 1))

    def calculate_gain(c):
        # Number of points in N(c) that have cover_count == 0
        # If we add c, these points become covered.
        nbs = neighbors[c]
        return int(np.sum(cover_count[nbs] == 0))

    # Initialize PQ
    for c in chosen_set:
        d = calculate_damage(c)
        heapq.heappush(removal_pq, (d, c))

    # Lazy Deletion: We don't remove items from PQ when cover_count changes.
    # Instead, when we pop, we check if the item is still in chosen_set 
    # and if its damage is still valid.
    # If not, we discard it and pop the next.
    # When cover_count changes, we *push* the new damage for affected words.
    # This ensures the PQ always has a valid (though potentially stale) entry for each word.
    # The "valid" entry is the one with the most recent damage? 
    # Actually, since damage can go up or down, we just push new values.
    # When popping, we check: 
    # 1. Is word in chosen_set?
    # 2. Does the popped damage match the current damage? 
    #    If not, it's stale, discard.
    #    If yes, it's the best current candidate for removal.

    # To make this efficient, we need to update the PQ for all words affected by a change.
    # When we add/remove a word, cover_count changes for its neighbors.
    # The damage of a word w depends on cover_count of N(w).
    # So if cover_count[p] changes, the damage of all w in N(p) might change.
    # This is a global update. 
    # 
    # Wait, the "lazy deletion" usually applies to the *candidates* themselves.
    # If we just want to find the best word to remove:
    # We pop (d, c) from PQ.
    # If c not in chosen_set: discard.
    # If calculate_damage(c) != d: 
    #    This entry is stale. Push (calculate_damage(c), c) and discard.
    #    (This ensures we eventually get the correct damage).
    # Else:
    #    This is the valid best candidate.
    #    If d == 0: Remove c.
    #    Else: No word can be removed (all have damage > 0). 
    #          We need a different move (Swap or Add).
    
    # Strategy:
    # 1. Try to remove.
    # 2. If no removal possible (min damage > 0), try a 2-swap.
    #    A 2-swap: Remove A, Add B.
    #    Net change in size: 0.
    #    Condition: After removing A, adding B must cover all holes created by A.
    #    Holes created by A: Points p in N(A) with cover_count[p] == 1.
    #    B must cover all these holes.
    #    So B must be a neighbor of all holes? No, B just needs to be in N(h) for all holes h.
    #    Actually, if we remove A, holes are H = {p | p in N(A) and cover_count[p]==1}.
    #    We need a B such that H is a subset of N(B).
    #    Also, B should ideally not introduce new holes (it doesn't, since we only add).
    #    Wait, if we remove A and add B, the size is constant.
    #    Does this help? It changes the *structure* of the cover.
    #    It might allow future removals.
    
    # Let's implement the removal loop with lazy PQ first.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try to find a removable word
        while removal_pq:
            d, c = heapq.heappop(removal_pq)
            
            if c not in chosen_set:
                continue
                
            current_d = calculate_damage(c)
            if current_d != d:
                # Stale entry
                heapq.heappush(removal_pq, (current_d, c))
                continue
                
            # Valid entry with min damage
            if current_d == 0:
                # Safe to remove
                chosen_set.remove(c)
                cover_count[neighbors[c]] -= 1
                
                # Update PQ for affected words
                # Words whose damage might have changed are those in N(N(c)).
                # This is expensive. 
                # Alternative: Just push updated damages for neighbors of c?
                # Damage of w depends on cover_count[N(w)].
                # cover_count changed for p in N(c).
                # So damage of w changes if N(w) intersects N(c).
                # i.e., w is a neighbor of some p in N(c).
                # So w is in N(N(c)).
                
                # To keep it simple and fast, we might just let the PQ get stale 
                # and rely on the "check on pop" mechanism.
                # We don't need to push new entries for *all* affected words immediately.
                # We just need to ensure that if a word becomes removable (damage 0),
                # it eventually gets a correct entry in the PQ.
                # If we don't push new entries, the PQ might contain old high-damage entries 
                # for words that are now damage 0.
                # When we pop an old high-damage entry, we check current_d.
                # If current_d == 0, we push (0, c) and continue. 
                # Next time we pop, we get (0, c) and remove it.
                # This works! We don't need to proactively push updates.
                # We just need to ensure that *every* word in chosen_set has *at least one* 
                # entry in the PQ. 
                # Since we only remove from chosen_set, and we push when we find stale entries,
                # this invariant holds.
                
                improved = True
                break # Restart search for next removable word
            else:
                # Min damage > 0. No word can be removed.
                # We need to try a swap or accept this as local optimum.
                # For now, let's try a random 2-swap to escape local minima.
                
                # Pick a random word A from chosen_set
                # Try to find a word B such that removing A and adding B is valid.
                # Holes = {p in N(A) | cover_count[p] == 1}
                # B must cover all Holes.
                
                # To avoid O(N^2) search, we can sample B from neighbors of the holes.
                # If Holes is empty, A is removable (contradiction with min damage > 0).
                # So Holes is not empty.
                
                # Let's pick a random A.
                a_list = list(chosen_set)
                if not a_list:
                    break
                A = random.choice(a_list)
                
                holes = np.where((cover_count[neighbors[A]] == 1))[0]
                # holes are indices of points in N(A) with count 1.
                # We need a B that covers all these points.
                # B must be in the intersection of N(h) for all h in holes.
                
                # If len(holes) is large, intersection is likely empty.
                # Let's just try a few random candidates from the union of N(h).
                
                candidates = set()
                for h in holes:
                    candidates.update(neighbors[h].tolist())
                
                # Remove A from candidates (we're replacing A)
                candidates.discard(A)
                
                # Try a few random candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                found_swap = False
                for B in cand_list[:10]: # Limit search
                    if B in chosen_set:
                        continue # Swapping with existing word is pointless (no change in set)
                    
                    # Check if B covers all holes
                    # B covers h if h is in N(B)
                    # i.e., B is in N(h)
                    # We already ensured B is in union of N(h).
                    # We need B in intersection.
                    
                    # Efficient check:
                    # count how many holes are covered by B.
                    # holes_covered = sum(1 for h in holes if h in neighbors[B])
                    # But neighbors[B] is an array. 
                    # We can use np.isin or set intersection.
                    
                    # Since holes is small (<= 17), let's just check.
                    b_nbs = set(neighbors[B].tolist())
                    if all(h in b_nbs for h in holes):
                        # Valid swap!
                        chosen_set.remove(A)
                        chosen_set.add(B)
                        
                        # Update cover_count
                        cover_count[neighbors[A]] -= 1
                        cover_count[neighbors[B]] += 1
                        
                        # Push updated entries for A and B? 
                        # A is gone, B is new.
                        # B needs an entry in PQ.
                        d_b = calculate_damage(B)
                        heapq.heappush(removal_pq, (d_b, B))
                        
                        # A's entries will be ignored as it's not in chosen_set.
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart loop to try removals again
                else:
                    # No swap found for this A. 
                    # If we tried all A's and found nothing, we are stuck.
                    # For simplicity, if we fail a few random A's, we break.
                    # This is a heuristic.
                    break

    # Final check: ensure we return a valid cover
    # If we ran out of time or got stuck, we might have a valid cover.
    # The loop above only removes if safe, or swaps if safe.
    # So the result should always be valid.
    
    result = [all_words[i] for i in chosen_set]
    return result
