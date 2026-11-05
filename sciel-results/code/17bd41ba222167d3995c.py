import time
import random
import itertools
import sys

def construct():
    """
    Constructs a linear triple system on 16 points with a small independence number.
    Strategy: Simulated Annealing to minimize the independence number.
    - State: A set of triples.
    - Move: Add, remove, or swap a triple.
    - Constraint: Any two triples share at most one point.
    - Objective: Minimize the size of the largest subset of points containing no triple.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (or 150s if that's the hard kill, but 120 is specified in contract)
    
    n_points = 16
    points = list(range(n_points))
    
    # Pre-compute all possible triples
    all_triples = list(itertools.combinations(points, 3))
    
    # Helper to check if a triple can be added to a set given pair counts
    # pair_count[i][j] is the number of triples containing both i and j
    # For linearity, we need pair_count[i][j] <= 1 for all triples in the set.
    # Actually, the condition is "any two triples share at most one point".
    # This is equivalent to saying no pair of points is contained in more than one triple.
    
    def get_pair_counts(triples):
        counts = [[0]*n_points for _ in range(n_points)]
        for t in triples:
            a, b, c = t
            counts[a][b] += 1
            counts[b][a] += 1
            counts[a][c] += 1
            counts[c][a] += 1
            counts[b][c] += 1
            counts[c][b] += 1
        return counts

    def calc_independence_number(triples):
        # Returns the size of the largest subset of points that contains no triple.
        # A subset S is independent if no triple is fully contained in S.
        # We want to find the max |S| such that for all t in triples, t is not a subset of S.
        # This is equivalent to finding the max |S| such that S does not contain any triple.
        # This is the complement of the vertex cover problem on the 3-uniform hypergraph?
        # No, it's the independence number of the hypergraph.
        
        # Since n=16 is small, we can check all subsets? 2^16 = 65536.
        # For each subset, check if it contains any triple.
        # To make it fast:
        # Represent triples as bitmasks.
        # A subset S (bitmask) contains a triple T (bitmask) if (S & T) == T.
        
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
            
        max_ind = 0
        # Iterate through all subsets
        # Optimization: If a subset S contains a triple, any superset of S also contains that triple.
        # We are looking for the largest S that does NOT contain any triple.
        # This is equivalent to: Find the largest S such that for all t in triple_masks, (S & t) != t.
        
        # We can iterate by size descending? Or just check all.
        # 65536 subsets * (number of triples) checks.
        # If we have ~50 triples, that's 3.2M operations. Very fast in Python if optimized?
        # Maybe use numpy or just efficient bit ops.
        
        # Alternative: 
        # The set of "forbidden" subsets are those that contain at least one triple.
        # We want the max size of a subset NOT in the forbidden set.
        
        # Let's just iterate all masks from (1<<16)-1 down to 0? 
        # No, we want the max size. So we can group by size or just check all and keep max.
        
        # Faster check:
        # Pre-compute for each point, the list of triples containing it?
        
        # Let's try a simple loop first, see if it's too slow.
        # If too slow, we can cache or use a different method.
        
        # Actually, we can use the property: 
        # A set S is independent iff it does not contain any hyperedge.
        
        # Let's use a heuristic check for speed during SA:
        # We don't need the exact alpha for every move if we can approximate, 
        # but for the final result we need exact.
        # For the SA cost function, we can use a faster proxy or just the exact one if it's fast enough.
        
        # Let's implement the exact check efficiently.
        # Sort triple masks?
        
        # Trick: If we represent S as a bitmask, we can check if any triple is contained.
        # We can pre-compute a "coverage" or use a SAT solver? No, std lib only.
        
        # Let's stick to:
        # for mask in range(1<<16):
        #    if popcount(mask) > current_best:
        #       check if independent
        
        # But checking all 65536 every move is expensive.
        # Number of moves in SA: maybe 100,000?
        # 100k * 65536 * (avg triples check) -> too slow.
        
        # We need a faster cost function or fewer moves.
        # Alternative: 
        # Independence number is the max k such that there exists a subset of size k with no triple.
        # We can binary search on k? 
        # Or, we can just check subsets of size k for k = 16, 15, ... down to 1.
        # Stop when we find one.
        # The independence number is likely small (e.g., 5-8).
        # So we check subsets of size 16, 15, 14...
        # Number of subsets of size 16: 1
        # Size 15: 16
        # Size 14: 120
        # Size 13: 560
        # Size 12: 1820
        # Size 11: 4368
        # Size 10: 8008
        # Size 9: 11440
        # Size 8: 12870
        # Size 7: 11440
        # Size 6: 8008
        # Size 5: 4368
        # Size 4: 1820
        # Size 3: 560
        # Size 2: 120
        # Size 1: 16
        
        # If alpha is 8, we check 1+16+120+560+1820+4368+8008+11440+12870 = ~40k subsets.
        # For each subset, check if it contains a triple.
        # If we have 50 triples, 40k * 50 = 2M ops.
        # This might be okay for a few hundred moves, but not 100k.
        
        # We need to be smarter.
        
        # Idea: 
        # Maintain the independence number incrementally? Hard.
        
        # Alternative Strategy: 
        # Use a greedy construction that specifically targets covering points to reduce independence number.
        # Or, use the SA but with a very fast approximation for the cost, and only compute exact alpha every N moves.
        
        # Let's try a different approach: 
        # **Greedy Set Cover-like approach**.
        # We want to choose triples such that every large subset contains at least one triple.
        # This is the "hitting set" dual.
        # We want to hit all subsets of size > k.
        # Equivalently, we want to minimize k such that the triples hit all subsets of size k+1.
        
        # Let's try a randomized greedy algorithm with local search (Swap).
        
        # Initial state: Random set of ~20-30 valid triples.
        
        # Cost function: Exact independence number.
        # To speed up cost calculation:
        # We can pre-compute for each triple its mask.
        # To check if a subset S (mask) is independent:
        #   for t_mask in triple_masks:
        #      if (S & t_mask) == t_mask: return False
        #   return True
        
        # We can optimize the check by ordering triple_masks by "popularity" or just random.
        
        # Let's implement the exact alpha calculation and see if it's fast enough for ~5000 moves.
        
        def calc_alpha(triples):
            if not triples:
                return n_points
            
            triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
            
            # Check sizes from n_points down to 1
            # We can use itertools.combinations for subsets of a specific size
            # But generating combinations is slow.
            # Bitmask iteration for a specific size is also slow in Python.
            
            # Let's try a different check:
            # The independence number is the maximum size of a set S such that no triple is contained in S.
            
            # We can use a simple branch and bound or just check all masks if n is small.
            # But 2^16 is 65k.
            # Let's just iterate all masks, but skip those with popcount <= current_best.
            
            # Actually, let's just iterate masks from (1<<16)-1 down to 0.
            # Keep track of max size found.
            # If we find a mask of size 10 that is independent, we know alpha >= 10.
            # We don't need to check all masks of size < 10 if we are looking for the MAX.
            # Wait, we want the MAX size.
            # So we can iterate sizes from 16 down to 1.
            # For each size k, generate all subsets of size k.
            # If any subset of size k is independent, return k.
            
            # Generating subsets of size k:
            # for subset in itertools.combinations(points, k):
            #    mask = ...
            #    check independence
            
            # Number of combinations for k=8 is 12870.
            # Check independence: loop over triples (~30). 12870 * 30 = 386k ops.
            # If we check k=16, 15, 14, 13, 12, 11, 10, 9, 8...
            # Sum of C(16,k) for k=16..8 is ~40k.
            # 40k * 30 = 1.2M ops per alpha calculation.
            # 1.2M ops in Python is ~0.1s? Maybe 0.05s.
            # If we do 1000 moves, that's 50s. Too slow for 120s if we have overhead.
            
            # We need a faster alpha check.
            
            # Observation:
            # If a set S contains a triple, it is not independent.
            # We can pre-compute for each point pair (i,j), the list of triples containing them? No.
            
            # Let's try to optimize the inner loop.
            # Instead of checking all triples for each subset, we can check if the subset is a "vertex cover" of the hypergraph? No.
            
            # Alternative: 
            # Use a SAT-like constraint propagation?
            
            # Let's try to reduce the number of triples in the system.
            # A Steiner Triple System on 16 points doesn't exist (16 mod 6 != 1 or 3).
            # Max number of disjoint triples is 5.
            # But we can have more triples if they share points, as long as no pair is repeated.
            # Max number of triples in a linear triple system on v points is bounded by v(v-1)/6.
            # For v=16, 16*15/6 = 40.
            # So max 40 triples.
            
            # If we have 40 triples, the system is dense.
            # The independence number should be small.
            
            # Let's try a construction based on **Z_15 + infinity**? No, 16 points.
            # Maybe use a known design?
            # There is no STS(16).
            # But there are near Steiner triple systems.
            
            # Let's try a specific construction:
            # Points 0..14 form a Z_15 structure?
            # Or use a finite field?
            
            # Let's stick to SA but optimize the alpha check.
            # We can cache the result of "is this mask independent"?
            # There are 65536 masks. We can pre-compute a boolean array `is_independent[mask]`.
            # When the set of triples changes, we need to update this array.
            # This is expensive.
            
            # Better: 
            # In the SA, we only change one triple at a time (add/remove/swap).
            # We can update the "covered" subsets incrementally?
            # A subset S becomes non-independent if it contains the new triple.
            # So if we add triple T, all supersets of T become non-independent.
            # If we remove triple T, some subsets might become independent again (if T was the only triple in them).
            
            # This suggests maintaining a count of how many triples are contained in each subset.
            # `cover_count[S]` = number of triples T such that T is a subset of S.
            # A subset S is independent iff `cover_count[S] == 0`.
            # The independence number is the max |S| such that `cover_count[S] == 0`.
            
            # Updating `cover_count` when adding a triple T:
            # For all supersets S of T, `cover_count[S]` += 1.
            # Number of supersets of a triple (size 3) in 16 points is 2^(16-3) = 2^13 = 8192.
            # So adding a triple takes 8192 updates.
            # Removing a triple takes 8192 updates.
            # Swapping takes 16384 updates.
            
            # This is very fast!
            # 8192 operations per move.
            # We can do 100,000 moves in ~800M operations. In Python, 800M simple ops is too slow (~80s+).
            # But we can optimize the update loop using NumPy?
            # Or just accept ~10,000 moves in 10s?
            # 10,000 moves * 8192 = 82M ops. Should be doable in <10s in Python if tight loop.
            
            # Let's implement this!
            
            # State:
            # `triples`: list of triples (tuples)
            # `cover_count`: array of size 1<<16, where cover_count[S] is the number of triples contained in S.
            # `current_alpha`: current independence number.
            
            # Initialization:
            # Start with empty set.
            # cover_count is all 0s.
            # alpha is 16.
            
            # Move:
            # 1. Add a random valid triple T.
            #    - Check if T is valid (no pair repeated).
            #    - Update cover_count for all supersets of T.
            #    - Update alpha: 
            #      Since cover_count only increases, alpha can only decrease or stay same.
            #      We need to find the new max |S| with cover_count[S] == 0.
            #      We can maintain a list of "candidate" sizes?
            #      Or just re-scan from current_alpha down?
            #      Actually, since cover_count increases, the set of independent sets shrinks.
            #      So alpha is non-increasing.
            #      We can keep `current_alpha` and check if the set of size `current_alpha` is still independent.
            #      If not, decrement and check next.
            
            # 2. Remove a random triple T.
            #    - Update cover_count for all supersets of T (decrease).
            #    - Alpha can increase.
            #    - We need to check if there are now independent sets of size > current_alpha.
            #    - We can check sizes from n_points down to current_alpha+1.
            
            # 3. Swap: Remove T1, Add T2.
            
            # To find new alpha:
            # We can maintain an array `max_size_independent`? No.
            # Just scan.
            # Scanning for alpha:
            # For k from n_points down to 1:
            #   Check if any subset of size k has cover_count == 0.
            #   This is expensive if we do it from scratch.
            
            # Optimization:
            # Maintain a list of all independent sets? No, too many.
            
            # Let's just re-calculate alpha by scanning sizes from `current_alpha` down (for add) or up (for remove).
            # But scanning all subsets of size k is expensive (C(16,k)).
            
            # Alternative:
            # Keep track of `cover_count` and just check the specific subsets that were affected?
            # No, a subset S is independent if cover_count[S] == 0.
            # When we add T, only supersets of T change.
            # So if `current_alpha` was k, and the only independent sets of size k were supersets of T, then alpha decreases.
            # We can just check if there exists ANY independent set of size `current_alpha`.
            # If yes, alpha stays.
            # If no, try `current_alpha - 1`.
            
            # How to check "exists independent set of size k"?
            # Iterate all subsets of size k.
            # If we do this every move, it's slow.
            
            # Heuristic: 
            # We can just accept a slightly slower alpha update.
            # Or, we can use the fact that we are doing SA, so we don't need the EXACT alpha for the cost function, 
            # we just need a proxy that correlates with it.
            # But the score is the EXACT alpha.
            # So we need to return the exact alpha for the final best solution.
            # During SA, we can use a fast approximation for the cost (e.g., just check a few random subsets of size current_alpha).
            # And at the end, compute the exact alpha for the best candidate.
            
            # Let's do that.
            # Cost function for SA:
            # 1. Check if the move is valid (linearity).
            # 2. Estimate alpha change.
            #    - If adding: alpha can only decrease.
            #    - If removing: alpha can only increase.
            #    - Use a random sample of subsets of size `current_alpha` to see if they are still independent.
            #      If we find one, alpha likely stays.
            #      If we find none, alpha likely decreased.
            
            # This is getting complex. Let's try a simpler, robust approach:
            # **Greedy Construction with Random Restarts**.
            
            # 1. Generate a random valid linear triple system.
            #    - Start with empty.
            #    - Randomly shuffle all triples.
            #    - Add triple if valid.
            #    - This gives a maximal linear triple system.
            # 2. Calculate exact alpha.
            # 3. Try local search: 
            #    - Swap a triple with another valid triple.
            #    - If new alpha < old alpha, accept.
            #    - If new alpha == old alpha, accept with probability p (to escape local minima).
            #    - If new alpha > old alpha, reject.
            
            # This is a standard local search.
            # The bottleneck is the alpha calculation.
            # Let's optimize the alpha calculation using the `cover_count` idea, but only update it when the system changes.
            # Since we are doing local search, we change one triple at a time.
            # We can maintain `cover_count` and update it incrementally.
            # And we can maintain `current_alpha` and update it by checking the boundary.
            
            # Let's implement this.
            
            n = 16
            N = 1 << n
            
            # Pre-compute supersets for each triple?
            # Supersets of a triple T (mask m) are all S such that (S & m) == m.
            # We can generate them by iterating over the 2^(n-3) combinations of the remaining points.
            
            # To speed up update:
            # def update_cover_count(triple_mask, delta):
            #     # delta is +1 or -1
            #     # Iterate all supersets
            #     rem_bits = [i for i in range(n) if not (triple_mask >> i) & 1]
            #     for r in itertools.product([0,1], repeat=len(rem_bits)):
            #         s_mask = triple_mask
            #         for i, b in zip(rem_bits, r):
            #             if b:
            #                 s_mask |= (1 << i)
            #         cover_count[s_mask] += delta
            
            # This is slow in Python.
            
            # Let's use NumPy for the cover_count array?
            # cover_count = np.zeros(N, dtype=np.int8)
            # Updating 8192 elements in NumPy is fast.
            # But generating the indices of supersets is still needed.
            
            # Actually, we can pre-compute the list of supersets for each of the 560 possible triples.
            # There are only 560 triples.
            # superset_indices[t_idx] = list of masks S that contain triple t_idx.
            
            # Pre-computation:
            triple_list = list(itertools.combinations(range(n), 3))
            triple_to_idx = {t: i for i, t in enumerate(triple_list)}
            triple_masks = [(1<<a)|(1<<b)|(1<<c) for a,b,c in triple_list]
            
            superset_indices = []
            for t_mask in triple_masks:
                idxs = []
                # Iterate all masks
                # This is 560 * 65536 = 36M ops. Too slow for startup?
                # 36M ops in Python is ~10s. Might be okay.
                # But we can do better.
                # For a given t_mask, the supersets are formed by taking t_mask and OR-ing with any subset of the complement.
                # comp_mask = ((1<<n)-1) ^ t_mask
                # Iterate subsets of comp_mask.
                # There are 2^(n-3) = 8192 subsets.
                # So for each triple, we generate 8192 masks.
                # Total 560 * 8192 = 4.6M ops. Very fast.
                
                comp = ((1<<n)-1) ^ t_mask
                sub = comp
                while sub:
                    idxs.append(t_mask | sub)
                    sub = (sub - 1) & comp
                idxs.append(t_mask) # Don't forget the triple itself
            
            # Now, we can update cover_count using NumPy fancy indexing.
            # cover_count[superset_indices[t_idx]] += 1
            
            # This should be very fast.
            
            # Let's implement the local search with this.
            
            import numpy as np
            
            cover_count = np.zeros(N, dtype=np.int8)
            
            def update_alpha(cover_count, current_alpha):
                # Find the largest k such that there exists a subset of size k with cover_count == 0.
                # We can check k from n down to 1.
                # To check if there is an independent set of size k:
                # We need to know if any mask of popcount k has cover_count == 0.
                # We can pre-compute lists of masks by popcount.
                
                # Pre-compute masks_by_popcount
                # This is a one-time cost.
                
                return current_alpha # Placeholder
            
            # Pre-compute masks_by_popcount
            masks_by_popcount = [[] for _ in range(n+1)]
            for m in range(N):
                pc = bin(m).count('1')
                masks_by_popcount[pc].append(m)
            
            # Now update_alpha:
            def get_alpha(cover_count, start_k):
                # Check from n down to 1
                for k in range(n, 0, -1):
                    # Check if any mask in masks_by_popcount[k] has cover_count == 0
                    # We can use np.any(cover_count[masks_by_popcount[k]] == 0)
                    # But converting list to array every time is slow.
                    # Let's pre-convert masks_by_popcount to arrays.
                    pass
            
            # Pre-convert
            masks_by_popcount_arr = [np.array(lst, dtype=np.int32) for lst in masks_by_popcount]
            
            def get_alpha(cover_count, start_k=n):
                for k in range(start_k, 0, -1):
                    if np.any(cover_count[masks_by_popcount_arr[k]] == 0):
                        return k
                return 0
            
            # This is still slow if we call it every move.
            # np.any on an array of size C(16,8) ~ 12k elements is fast.
            # But we might check multiple k's.
            
            # Let's try it.
            
            # Initial state: Empty
            triples = []
            current_alpha = n
            
            # Random restarts
            best_alpha = n
            best_triples = []
            
            for restart in range(100):
                if time.time() - start_time > deadline - 10:
                    break
                
                # Reset
                triples = []
                cover_count[:] = 0
                current_alpha = n
                
                # Greedy fill
                shuffled_triples = list(triple_list)
                random.shuffle(shuffled_triples)
                
                # We need to track which triples are added to check linearity.
                # We can use a set of pairs? Or just check against current triples.
                # Since we are adding one by one, we can check if the new triple shares a pair with any existing triple.
                # Pair check: 
                # For new triple (a,b,c), check if any existing triple contains (a,b), (a,c), or (b,c).
                # We can maintain a set of used pairs.
                
                used_pairs = set()
                
                for t in shuffled_triples:
                    a,b,c = t
                    pairs = [(a,b), (a,c), (b,c)]
                    if any(p in used_pairs for p in pairs):
                        continue
                    
                    # Add triple
                    t_idx = triple_to_idx[t]
                    t_mask = triple_masks[t_idx]
                    
                    # Update cover_count
                    cover_count[superset_indices[t_idx]] += 1
                    
                    triples.append(t)
                    used_pairs.update(pairs)
                    
                    # Update alpha
                    # Since we only added, alpha can only decrease.
                    # We can check if current_alpha is still valid.
                    # If not, decrement.
                    # But checking "is current_alpha valid" requires scanning.
                    # Let's just re-calculate alpha every 10 additions to save time?
                    # Or just calculate it now.
                    current_alpha = get_alpha(cover_count, current_alpha)
                
                # Local Search
                # Try to improve alpha by swapping.
                for step in range(1000):
                    if time.time() - start_time > deadline - 5:
                        break
                    
                    if current_alpha <= 4: # Good enough?
                        break
                        
                    # Pick a random triple to remove
                    if not triples:
                        break
                    rem_idx = random.randint(0, len(triples)-1)
                    rem_t = triples[rem_idx]
                    rem_t_idx = triple_to_idx[rem_t]
                    
                    # Pick a random triple to add (not currently in set, and valid)
                    # We need to find a valid triple to add.
                    # This is tricky.
                    # Let's just try random triples from the list and see if they are valid.
                    
                    added = False
                    for _ in range(100): # Try 100 random candidates
                        cand_t = random.choice(triple_list)
                        if cand_t in triples:
                            continue
                        
                        a,b,c = cand_t
                        pairs = [(a,b), (a,c), (b,c)]
                        # Check if any pair is used by OTHER triples (excluding the one we are removing)
                        # We can just check against used_pairs, but we need to temporarily remove rem_t's pairs.
                        
                        # Temporarily remove rem_t from used_pairs
                        rem_pairs = [(rem_t[0], rem_t[1]), (rem_t[0], rem_t[2]), (rem_t[1], rem_t[2])]
                        temp_used = used_pairs - set(rem_pairs)
                        
                        if any(p in temp_used for p in pairs):
                            continue
                        
                        # Candidate is valid
                        # Perform swap
                        # Remove rem_t
                        cover_count[superset_indices[rem_t_idx]] -= 1
                        triples.pop(rem_idx)
                        used_pairs -= set(rem_pairs)
                        
                        # Add cand_t
                        cand_t_idx = triple_to_idx[cand_t]
                        cover_count[superset_indices[cand_t_idx]] += 1
                        triples.append(cand_t)
                        used_pairs.update(pairs)
                        
                        # Update alpha
                        new_alpha = get_alpha(cover_count, current_alpha)
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            added = True
                            break
                        elif new_alpha == current_alpha:
                            # Accept with probability 0.5?
                            if random.random() < 0.5:
                                current_alpha = new_alpha
                                added = True
                                break
                            else:
                                # Revert
                                cover_count[superset_indices[cand_t_idx]] -= 1
                                triples.pop()
                                used_pairs -= set(pairs)
                                cover_count[superset_indices[rem_t_idx]] += 1
                                triples.insert(rem_idx, rem_t)
                                used_pairs.update(rem_pairs)
                        else:
                            # Revert
                            cover_count[superset_indices[cand_t_idx]] -= 1
                            triples.pop()
                            used_pairs -= set(pairs)
                            cover_count[superset_indices[rem_t_idx]] += 1
                            triples.insert(rem_idx, rem_t)
                            used_pairs.update(rem_pairs)
                    
                    if not added:
                        # No improvement found in 100 tries, maybe break?
                        pass
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = triples.copy()
            
            # Final verification and return
            # We need to return the list of triples.
            # The score is the independence number.
            # We return the triples. The evaluator will calculate the score.
            # But we should ensure the returned triples are valid.
            
            # Let's do a final sanity check on best_triples.
            # And return it.
            
            return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
