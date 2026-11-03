import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Problem parameters are fixed by the task: q=3, n=7, radius=1
    # But we use the arguments for generality if possible, though the logic is tuned for this case.
    if q != 3 or n != 7 or radius != 1:
        # Fallback to simple greedy if parameters differ
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    N = len(all_words)
    
    # Precompute neighbors (Hamming distance <= radius)
    # For efficiency, we can compute this on the fly or precompute indices.
    # Since N=2187, precomputing a list of neighbor indices for each word is fast.
    # Let's use a dictionary for index lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i in range(N):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < q:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    idx = word_to_idx[tuple(new_word)]
                    neigh.append(idx)
        neighbors.append(neigh)
    
    # Initial Solution: Direct Sum of Tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            init_indices.append(word_to_idx[tuple(word)])
    
    S = set(init_indices)
    best_S = set(S)
    best_size = len(S)
    
    # Coverage counts
    cover_count = np.zeros(N, dtype=np.int32)
    for j in S:
        for i in neighbors[j]:
            cover_count[i] += 1

    # Simulated Annealing / Local Search Hybrid
    # State: S (set of selected indices), cover_count (array of counts)
    # Moves:
    # 1. Remove redundant (if possible)
    # 2. Swap (if it maintains validity or is accepted by SA)
    # 3. Add (if necessary to fix invalidity, but we try to avoid)
    
    # We will maintain validity strictly.
    # The "Annealing" part is in accepting swaps that don't reduce size immediately
    # but might enable future reductions? No, usually we want to reduce size.
    # Standard approach:
    # 1. Greedy removal of redundant nodes.
    # 2. Swap search: Find (j in S, k not in S) such that S' = S \ {j} U {k} is valid.
    #    If such a swap exists, it doesn't change size. Why do it?
    #    It changes the configuration, potentially allowing more removals later.
    #    This is "neutral" moves.
    #    To escape local minima in the "removal" phase, we need to move to a different
    #    local optimum of the same size.
    
    # To make it more aggressive, we can allow temporary increase in size?
    # No, the score is minimized size. Increasing size is bad.
    # But if we get stuck at 190, maybe a swap leads to a state where we can remove 2 nodes.
    
    # Let's implement a robust local search:
    # Loop:
    #   1. Try to remove any redundant node. If found, remove and restart loop.
    #   2. If no removals, try to find a valid swap.
    #      - Iterate over all j in S.
    #      - For each j, find all k not in S such that swap is valid.
    #      - Pick one (random or best heuristic) and perform swap.
    #      - This changes the "landscape" for removals.
    #   3. If no valid swap found, try "kick":
    #      - Remove a random j, add a random k (even if invalid).
    #      - Then run a repair phase to add necessary nodes to restore validity.
    #      - Then run removal phase.
    #      - If size is still better or equal, keep. If worse, revert.
    #      - Actually, if we allow temporary invalidity, we need a repair mechanism.
    #      - Simpler: Just do random valid swaps to explore the space of minimum-size (for current config) solutions.
    
    # To speed up swap search:
    # For a fixed j, which k are valid?
    # Condition: For all i in neighbors[j], if cover_count[i] == 1, then i must be in neighbors[k].
    # Let U_j = { i in neighbors[j] | cover_count[i] == 1 }
    # We need k such that U_j is a subset of neighbors[k].
    # We can precompute for each i, the set of k that cover i?
    # Or, for each k, precompute the set of i it covers (neighbors[k]).
    # We have neighbors[k].
    # So for a given j, we need k where U_j subset neighbors[k].
    # We can iterate k not in S and check.
    # Optimization: If U_j is empty, any k not in S is a valid swap (trivially, since no critical words are lost).
    #    Wait, if U_j is empty, it means no word is exclusively covered by j.
    #    So removing j is valid? No, we just need to ensure no count drops to 0.
    #    If U_j is empty, then for all i in neighbors[j], cover_count[i] >= 2.
    #    So removing j leaves all counts >= 1.
    #    So j is redundant!
    #    So if U_j is empty, we should have removed j in step 1.
    #    Therefore, if step 1 fails, U_j is non-empty for all j in S.
    
    # So we need k that covers U_j.
    
    # Let's try to find such k efficiently.
    # We can precompute a "coverage mask" or just use the list.
    # Since N=2187, and |neighbors[k]|=15, checking subset is fast.
    
    # Strategy:
    # 1. Removal Phase
    # 2. Swap Phase:
    #    - Pick a random j in S.
    #    - Compute U_j.
    #    - Find all k not in S such that U_j subset neighbors[k].
    #    - If found, pick one (maybe the one that maximizes "flexibility" or just random) and swap.
    #    - If not found, try another j.
    #    - If no j has a valid swap, we are in a local optimum.
    #    - To escape, we can try a "double swap" or accept a "worse" state?
    #    - Or just restart with a different initial seed?
    
    # Let's add a "kick" mechanism if stuck:
    # If no removals and no swaps:
    #   - Pick a random j in S.
    #   - Pick a random k not in S.
    #   - Remove j, Add k. (This might break validity)
    #   - Now we have a set S' that might be invalid.
    #   - Run a "repair" greedy:
    #       While invalid:
    #           Find the word i with min cover_count (which is 0).
    #           Find the candidate k' that covers the most uncovered words (or just any that covers i).
    #           Add k'.
    #   - Then run Removal Phase on the repaired set.
    #   - If size is smaller than before, keep. Else, revert.
    #   - This allows jumping over energy barriers.
    
    stuck_counter = 0
    
    while time.time() < deadline:
        # 1. Removal Phase
        removed_any = False
        # We iterate until no more removals are possible
        while True:
            redundant_found = False
            # To avoid bias, shuffle S or iterate in random order?
            # Iterating in fixed order is fine, but might miss opportunities if order matters?
            # Removal is commutative in the sense that if j is redundant, removing it doesn't affect the redundancy of k
            # unless k relied on j? No, redundancy means j's removal doesn't break coverage.
            # If j is redundant, removing it is always safe.
            # So we can just scan all.
            for j in list(S):
                # Check if j is redundant
                is_redundant = True
                for i in neighbors[j]:
                    if cover_count[i] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    S.remove(j)
                    for i in neighbors[j]:
                        cover_count[i] -= 1
                    removed_any = True
                    redundant_found = True
                    break # Restart scan to update counts
            if not redundant_found:
                break
        
        if len(S) < best_size:
            best_S = set(S)
            best_size = len(S)
            stuck_counter = 0
        elif removed_any:
            stuck_counter = 0
        else:
            stuck_counter += 1
        
        if not removed_any:
            # 2. Swap Phase
            swap_done = False
            # Try to find a valid swap
            # Iterate over j in S
            # To be random, shuffle S
            for j in random.sample(list(S), len(S)):
                # Compute U_j: words uniquely covered by j
                U_j = [i for i in neighbors[j] if cover_count[i] == 1]
                if not U_j:
                    # Should have been removed
                    continue
                
                # Find k not in S such that U_j subset neighbors[k]
                # Optimization: Pick a random i in U_j.
                # k must cover i.
                # So k must be in the "reverse neighbors" of i.
                # Let's precompute reverse neighbors: rev_neigh[i] = list of k such that i in neighbors[k]
                # This is the same as neighbors! Since distance is symmetric.
                # So k must be in neighbors[i].
                
                # So we can restrict search for k to intersection of neighbors[i] for i in U_j.
                # But U_j can be large.
                # Let's just pick a random i from U_j and search in neighbors[i].
                # This is a heuristic: if k covers i, it *might* cover the rest of U_j.
                
                i_target = random.choice(U_j)
                candidates = [k for k in neighbors[i_target] if k not in S]
                
                # Shuffle candidates
                random.shuffle(candidates)
                
                for k in candidates:
                    # Check if k covers ALL of U_j
                    valid = True
                    for i in U_j:
                        if i not in neighbors[k]:
                            valid = False
                            break
                    if valid:
                        # Perform swap
                        S.remove(j)
                        for i in neighbors[j]:
                            cover_count[i] -= 1
                        S.add(k)
                        for i in neighbors[k]:
                            cover_count[i] += 1
                        swap_done = True
                        break
                if swap_done:
                    break
            
            if swap_done:
                stuck_counter = 0
            else:
                # 3. Kick / Perturbation
                # If stuck for a while, try a random perturbation
                if stuck_counter > 10:
                    # Save state
                    S_old = set(S)
                    cc_old = cover_count.copy()
                    
                    # Remove 2 random, Add 2 random?
                    # Or Remove 1, Add 1, then repair.
                    if len(S) > 2:
                        j1, j2 = random.sample(list(S), 2)
                        S.remove(j1)
                        S.remove(j2)
                        for i in neighbors[j1]:
                            cover_count[i] -= 1
                        for i in neighbors[j2]:
                            cover_count[i] -= 1
                        
                        # Add 2 random not in S
                        candidates_add = [k for k in range(N) if k not in S]
                        if candidates_add:
                            k1, k2 = random.sample(candidates_add, 2)
                            S.add(k1)
                            S.add(k2)
                            for i in neighbors[k1]:
                                cover_count[i] += 1
                            for i in neighbors[k2]:
                                cover_count[i] += 1
                        
                        # Repair
                        # While any cover_count[i] == 0:
                        #   Find i with 0
                        #   Find k that covers i and maximizes coverage of other 0s
                        #   Add k
                        while True:
                            # Find uncovered
                            uncovered_idx = np.where(cover_count == 0)[0]
                            if len(uncovered_idx) == 0:
                                break
                            
                            # Pick a random uncovered word
                            i_unc = random.choice(uncovered_idx)
                            
                            # Find best k to add
                            best_k = -1
                            best_gain = -1
                            # Candidates are neighbors of i_unc
                            for k in neighbors[i_unc]:
                                if k in S:
                                    continue
                                # Count how many uncovered words k covers
                                gain = 0
                                for i in neighbors[k]:
                                    if cover_count[i] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_k = k
                            
                            if best_k == -1:
                                break # Should not happen unless S is empty or something weird
                            
                            S.add(best_k)
                            for i in neighbors[best_k]:
                                cover_count[i] += 1
                        
                        # Check if we improved
                        if len(S) < best_size:
                            best_S = set(S)
                            best_size = len(S)
                            stuck_counter = 0
                        elif len(S) <= len(S_old):
                            # Keep the perturbed state if size is same or better
                            stuck_counter = 0
                        else:
                            # Revert
                            S = S_old
                            cover_count = cc_old
                            stuck_counter += 10 # Penalize failure
    
    # Convert best_S to list of lists
    return [all_words[j].tolist() for j in best_S]

def _generic_greedy(q, n, radius):
    # Fallback
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    # Simple greedy
    covered = [False] * N
    S = []
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = []
        for j, v in enumerate(all_words):
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                neigh.append(j)
        neighbors.append(neigh)
        
    while any(not c for c in covered):
        best_idx = -1
        best_gain = -1
        for j in range(N):
            if j in S:
                continue
            gain = sum(1 for i in neighbors[j] if not covered[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        if best_idx == -1:
            break
        S.append(best_idx)
        for i in neighbors[best_idx]:
            covered[i] = True
            
    return [all_words[j] for j in S]
