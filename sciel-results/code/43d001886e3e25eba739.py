import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Pre-compute bit_count for all masks (or use int.bit_count if available, but let's be safe with a list or just use .bit_count())
    # Python 3.10+ has int.bit_count(). Let's assume it's available or use bin(x).count('1') if not.
    # Given the environment, int.bit_count() is standard in modern Python.
    
    def subset_count(mask):
        return mask.bit_count()
    
    # Initialize with a greedy base to ensure a high starting score
    # Base sets: Empty, Singletons (12), Full (1), Size-11 (12) = 26 sets.
    # We need 14 more.
    # Good candidates for the remaining 14: Sets of size 10 and 9 that are "compatible" with the base.
    # A set S is compatible with base if it contains many singletons and is contained in the full set (always true) or contains many size-11 sets (i.e., S is a subset of many size-11 sets? No, S is a subset of size-11 sets if S is small. Size-11 sets are supersets of S if |S| <= 11. So any S of size <= 11 is a subset of 12 - |S| size-11 sets? No.
    # Let A be a set of size k. It is a subset of the size-11 set B if A \subseteq B. B is the full set minus one element. So A \subseteq B iff A does not contain the missing element of B.
    # If A has size k, the number of size-11 sets that are supersets of A is 12 - k (for each element in A, it must NOT be the missing one? No. B = V \ {v}. A \subseteq V \ {v} iff v \notin A. So there are 12 - k choices for v. Thus, A is a subset of 12-k size-11 sets.
    # Also A is a superset of all its subsets in the base. The base contains all singletons. So A contains k singletons.
    # So the gain of adding A to the base (Empty, Singletons, Full, Size-11) is:
    # Pairs with Empty: 1 (Empty < A)
    # Pairs with Singletons: k (Singleton < A)
    # Pairs with Full: 1 (A < Full)
    # Pairs with Size-11: 12 - k (A < Size-11 set)
    # Total gain = 1 + k + 1 + 12 - k = 14.
    # Wait, this is constant for any k (1 <= k <= 11)?
    # Let's re-verify.
    # Empty < A: 1 pair.
    # Singleton {i} < A: 1 if i in A. Sum over i in A is k.
    # A < Full: 1 pair.
    # A < B (where B is size 11): 1 if A \subseteq B. Number of such B is 12 - |A|.
    # Sum = 1 + k + 1 + 12 - k = 14.
    # So any set of size 1..11 gives 14 new pairs with the base.
    # However, we also need to count pairs among the *newly added* sets.
    # If we add multiple sets, we want them to have containment relations among themselves.
    # So we should pick sets that form a chain or a dense subgraph in the containment poset.
    # A chain of length L has L(L-1)/2 pairs.
    # If we pick 14 sets that form a chain, we get 14*13/2 = 91 extra pairs.
    # Plus 14 * 14 = 196 pairs with the base.
    # Total = 196 + 91 = 287.
    # Can we do better?
    # What if we pick sets that are all subsets of a specific size-10 set S10?
    # Let the 14 new sets be subsets of S10.
    # Then for any two new sets A, B, if A \subset B, it's a pair.
    # Also, A is a subset of S10 (if S10 is in the base? No, S10 is one of the new sets).
    # Let's just use a greedy algorithm to pick the 14 sets that maximize the total score, starting from the base.
    
    base_sets = []
    base_sets.append([])
    for i in range(n):
        base_sets.append([i])
    base_sets.append(list(range(n)))
    for i in range(n):
        s = [x for x in range(n) if x != i]
        base_sets.append(s)
    
    current_sets = base_sets
    current_masks = set(frozenset(s) for s in current_sets)
    # Represent as integer masks for speed
    mask_to_list = {}
    mask_list = []
    for s in current_sets:
        mask = 0
        for i in s:
            mask |= (1 << i)
        mask_to_list[mask] = s
        mask_list.append(mask)
    
    # Candidate pool: All sets of size 2 to 10?
    # Size 1 and 11 are already in base.
    # Size 0 and 12 are in base.
    # Let's generate all masks of size 2..10.
    candidates = []
    for r in range(2, 11):
        for combo in itertools.combinations(range(n), r):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            candidates.append(mask)
            
    if time.time() > deadline:
        return [mask_to_list[mm] for mm in mask_list]

    # Greedy selection of remaining m - len(current_sets) sets
    while len(current_sets) < m:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_mask = None
        
        for c_mask in candidates:
            if c_mask in current_masks:
                continue
                
            gain = 0
            c_count = c_mask.bit_count()
            
            # Pre-calculate contribution to base-like sets? No, just iterate current_masks.
            # current_masks is a set of integers.
            for s_mask in current_masks:
                if s_mask == c_mask:
                    continue
                # Check subset relation
                # s_mask < c_mask  <=> (s_mask & c_mask) == s_mask and s_mask != c_mask
                # c_mask < s_mask  <=> (c_mask & s_mask) == c_mask and c_mask != s_mask
                if (s_mask & c_mask) == s_mask:
                    gain += 1
                elif (c_mask & s_mask) == c_mask:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_mask = c_mask
                
        if best_mask is None:
            break
            
        current_sets.append(mask_to_list.get(best_mask, None)) # Wait, mask_to_list only has base sets.
        # We need to create the list for best_mask
        l = []
        for i in range(n):
            if best_mask & (1 << i):
                l.append(i)
        current_sets.append(l)
        mask_to_list[best_mask] = l
        mask_list.append(best_mask)
        current_masks.add(best_mask)
        
        # Remove best_mask from candidates to avoid re-checking?
        # candidates is a list. Removing is O(N). N is ~4000. 14 iterations. 56k ops. Fast.
        try:
            candidates.remove(best_mask)
        except ValueError:
            pass

    # Optional: Local Search to improve
    # The greedy method might be stuck in a local optimum.
    # Try to swap one set for another to improve the score.
    # Score = sum of containment pairs.
    
    def calc_score(masks):
        score = 0
        m = len(masks)
        for i in range(m):
            mi = masks[i]
            for j in range(i+1, m):
                mj = masks[j]
                if (mi & mj) == mi:
                    score += 1
                elif (mi & mj) == mj:
                    score += 1
        return score

    best_score = calc_score(mask_list)
    best_list = list(mask_list)
    
    # Local Search: Simulated Annealing or Hill Climbing
    # Given time constraints, let's try a simple hill climbing with random restarts if time permits.
    # But 40s is tight for complex search. Let's stick to the greedy result if it's fast.
    # The greedy part takes < 1s. We have ~70s left.
    # Let's try to optimize the 14 non-base sets.
    # Fix the base sets (Empty, Singletons, Full, Size-11).
    # Optimize the 14 others.
    
    # Actually, the greedy selection already picked the best 14 given the base.
    # Are the base sets optimal?
    # Maybe replacing some singletons with larger sets yields more?
    # Unlikely, singletons are very versatile.
    
    # Let's try a few random restarts of the greedy selection for the 14 non-base sets?
    # Or just perturb the current solution.
    
    # Let's try to improve the current solution via hill climbing on the full set of 40.
    # Move: Replace one set with a different set from the universe (excluding current).
    # Universe is huge (4096).
    # Gain calculation:
    # Remove set S_old. Loss = number of sets S_new in current \ {S_old} such that S_new < S_old or S_old < S_new.
    # Add set S_new. Gain = number of sets S_keep in current \ {S_old} such that S_keep < S_new or S_new < S_keep.
    # Delta = Gain - Loss.
    
    current_masks_set = set(mask_list)
    current_score = best_score
    
    # To speed up hill climbing, we can precompute "neighborhood" or just iterate.
    # 40 sets. 4096 candidates. 40 * 4096 * 40 swaps = 6.5 million ops per full scan.
    # 1 full scan ~ 0.1s in Python? Maybe 0.5s.
    # We can do ~100 scans in 50s.
    
    improved = True
    max_iters = 100 # Safety valve
    
    while improved and time.time() < deadline - 10.0:
        improved = False
        best_delta = 0
        best_move = None # (idx_to_replace, new_mask)
        
        # Pre-compute interactions of each current mask with all candidates?
        # That's 40 * 4096 = 163k comparisons.
        # For each candidate c, and each current s, determine if they are comparable.
        # This can be done in bulk?
        
        # Let's just iterate.
        # We can prune candidates: only consider masks with size 2..10 (since base is fixed, and we can swap any set, but swapping a base set for a non-base set is less likely to be beneficial? No, we should allow it).
        # Actually, let's just allow swapping any set for any other set.
        
        # To make it faster, let's only consider swapping non-base sets?
        # The base sets (Empty, Singletons, Full, Size-11) form a very strong core.
        # Let's identify the indices of base sets.
        base_indices = set()
        for i, s in enumerate(current_sets[:26]): # First 26 are base
            base_indices.add(i)
            
        # Only swap non-base sets?
        # If we swap a base set, we lose many pairs.
        # Let's try swapping only the 14 non-base sets first.
        
        non_base_indices = [i for i in range(len(mask_list)) if i not in base_indices]
        
        for idx in non_base_indices:
            old_mask = mask_list[idx]
            
            # Calculate loss from removing old_mask
            loss = 0
            for j in range(len(mask_list)):
                if j == idx:
                    continue
                s_mask = mask_list[j]
                if (s_mask & old_mask) == s_mask:
                    loss += 1
                elif (s_mask & old_mask) == old_mask:
                    loss += 1
                    
            # Find best replacement
            best_gain_for_old = -1
            best_new_mask = None
            
            for c_mask in candidates: # Reuse candidates list? It was reduced.
                if c_mask in current_masks_set:
                    continue
                if c_mask == old_mask:
                    continue
                
                gain = 0
                for j in range(len(mask_list)):
                    if j == idx:
                        continue
                    s_mask = mask_list[j]
                    if (s_mask & c_mask) == s_mask:
                        gain += 1
                    elif (s_mask & c_mask) == c_mask:
                        gain += 1
                
                delta = gain - loss
                if delta > best_gain_for_old:
                    best_gain_for_old = delta
                    best_new_mask = c_mask
                    
            if best_gain_for_old > 0:
                # Apply move
                current_score += best_gain_for_old
                mask_list[idx] = best_new_mask
                current_masks_set.remove(old_mask)
                current_masks_set.add(best_new_mask)
                
                # Update best_score if improved
                if current_score > best_score:
                    best_score = current_score
                    best_list = list(mask_list)
                
                improved = True
                # Break to restart the loop? Or continue?
                # Continuing might find more improvements in the same pass.
                # But the loss calculation for other indices changes.
                # So we should restart the pass.
                break
                
        max_iters -= 1

    # Reconstruct lists from best_list
    result = []
    for mask in best_list:
        l = []
        for i in range(n):
            if mask & (1 << i):
                l.append(i)
        result.append(l)
        
    # Ensure types are correct
    # "integers must be Python int ... sequences must be list or tuple"
    # My list construction uses ints.
    
    return result
