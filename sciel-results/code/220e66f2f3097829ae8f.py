import time
import random
import sys
from itertools import combinations

# Increase recursion limit
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES) # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# --- Precomputation: K(2,2,2) Configurations ---
# A K(2,2,2) is defined by 3 disjoint pairs of vertices.
# There are C(16,2)*C(14,2)*C(12,2)/... structures, but we can enumerate them by picking 6 vertices and 3 pairings.
# Total number of such structures:
# Choose 6 vertices: C(16,6) = 8008
# Number of ways to partition 6 items into 3 pairs: 15
# Total K222s = 8008 * 15 = 120120

# We need:
# 1. For each triple, the list of K222 indices it belongs to.
# 2. For each K222, the list of triple indices it contains.

# To save memory and time, we will build:
# triple_to_k222s: list of lists, where triple_to_k222s[i] is a list of K222 indices containing triple i.
# k222_triples: list of lists, where k222_triples[k] is a list of triple indices in K222 k.

def _precompute_k222s():
    triple_to_k222s = [[] for _ in range(N_TRIPLES)]
    k222_triples = []
    
    # Iterate over all sets of 6 vertices
    for verts in combinations(range(V), 6):
        # Generate all 15 pairings of these 6 vertices
        # Pairing is a list of 3 pairs.
        # We can generate these by fixing the partner of the first vertex, then the partner of the next available, etc.
        # Let verts be sorted: v0, v1, v2, v3, v4, v5
        # Pairs: (v0, v1), (v2, v3), (v4, v5) -> one structure
        # We need all perfect matchings.
        
        # Efficient generation of matchings for 6 elements:
        # There are 15 matchings.
        # We can hardcode the logic or use a small recursive generator.
        
        def get_matchings(verts):
            # verts is a tuple of 6 distinct integers
            # Returns list of tuples of 3 tuples (pairs)
            matchings = []
            v0 = verts[0]
            for i in range(1, 6):
                p1 = (v0, verts[i])
                remaining = [v for j, v in enumerate(verts) if j != 0 and j != i]
                # remaining has 4 elements
                # Matchings of 4 elements: 3 ways
                # (r0, r1), (r2, r3)
                # (r0, r2), (r1, r3)
                # (r0, r3), (r1, r2)
                r0, r1, r2, r3 = remaining
                matchings.append((p1, (r0, r1), (r2, r3)))
                matchings.append((p1, (r0, r2), (r1, r3)))
                matchings.append((p1, (r0, r3), (r1, r2)))
            return matchings

        for matching in get_matchings(verts):
            p1, p2, p3 = matching
            # Generate the 8 transversals
            t_indices = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(_triple_idx[t])
            
            k_idx = len(k222_triples)
            k222_triples.append(t_indices)
            
            for t_idx in t_indices:
                triple_to_k222s[t_idx].append(k_idx)
                
    return triple_to_k222s, k222_triples

# Run precomputation once at module load
triple_to_k222s, k222_triples = _precompute_k222s()
N_K222 = len(k222_triples)

# For faster access in local search, we might want sets or bitmasks for triple_to_k222s?
# The lists are small (each triple is in ~ C(13,3)*... K222s? 
# A triple (a,b,c) is in a K222 if we can extend it to 3 pairs.
# The pairs must contain a, b, c respectively? No.
# The pairs partition the 6 vertices.
# If T={a,b,c}, the K222 must have pairs P1,P2,P3 such that T is a transversal.
# This means one pair contains a, one b, one c.
# The other vertex in a's pair can be any of the 13 others.
# The other vertex in b's pair can be any of the 12 others (excluding a's partner? No, pairs are disjoint).
# Actually, the number of K222s containing a specific triple is:
# Choose partner for a: 13 choices.
# Choose partner for b: 12 choices (must be distinct from a, a_partner, c).
# Choose partner for c: 11 choices (must be distinct from a, a_partner, b, b_partner).
# So 13 * 12 * 11 = 1716 K222s per triple.
# This is a lot. 560 * 1716 = ~960k entries.
# The list approach is fine.

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    best_size = 0
    best_active = [False] * N_TRIPLES
    best_counts = [0] * N_K222
    
    # Precompute sets of K222 indices for each triple for faster subset checks if needed?
    # Actually, we just need to iterate.
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # Greedy Initialization
    def greedy_init(seed):
        random.seed(seed)
        active = [False] * N_TRIPLES
        counts = [0] * N_K222
        current_size = 0
        
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        for t_idx in order:
            # Check if adding this triple completes any K222
            # It completes a K222 if the count for that K222 is already 7
            valid = True
            for k_idx in triple_to_k222s[t_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                active[t_idx] = True
                current_size += 1
                for k_idx in triple_to_k222s[t_idx]:
                    counts[k_idx] += 1
        
        return active, counts, current_size

    # Local Search
    def local_search(active, counts, current_size):
        improved = True
        while improved and time_left() > 1.0:
            improved = False
            
            # Phase 1: Add any possible triple
            # We iterate through all inactive triples.
            # If a triple is inactive, it was blocked.
            # It becomes addable if no K222 containing it has count 7.
            # Note: Adding a triple only increases counts, so a triple that was NOT addable
            # (blocked by a K222 with count 7) will never become addable unless we REMOVE a triple
            # from that K222.
            # So in Phase 1, we only need to check triples that might have become unblocked?
            # No, if we only ADD, counts only increase. So if it wasn't addable before, it's not addable now.
            # Wait, this logic applies if we start from an empty set and only add.
            # But in local search, we might have removed triples in previous iterations (swaps).
            # So we must check all inactive triples.
            
            # Optimization: Keep a list of inactive triples?
            # Or just scan 0..N-1. N=560 is small.
            
            added_any = False
            for i in range(N_TRIPLES):
                if time_left() < 0.5:
                    return active, counts, current_size
                if not active[i]:
                    # Check if addable
                    addable = True
                    for k_idx in triple_to_k222s[i]:
                        if counts[k_idx] == 7:
                            addable = False
                            break
                    if addable:
                        active[i] = True
                        current_size += 1
                        for k_idx in triple_to_k222s[i]:
                            counts[k_idx] += 1
                        improved = True
                        added_any = True
                        # We could break here and restart, but continuing is faster if we don't restart
                        # However, adding i might block other triples that were previously addable.
                        # But we are checking them one by one. If j was addable before i was added,
                        # and i and j share a K222, j might now be blocked.
                        # But we already checked j? No, we are in a single pass.
                        # If j < i, we already checked j. If j was addable, we added it.
                        # If j > i, we haven't checked j yet.
                        # So a single pass is fine for "add all currently addable".
                        # It might miss some if the order matters?
                        # Suppose i blocks j. If we process i first, j becomes blocked.
                        # If we process j first, j is added, then i is blocked.
                        # Since we want to maximize size, the order doesn't change the set of "maximally addable"
                        # in a single pass? No, it can.
                        # Greedy choice.
                        # Let's just do a single pass. It's a heuristic.
            
            if added_any:
                continue # Restart loop to see if more can be added after this pass?
                         # Actually, if we added in this pass, we should restart the while loop
                         # to do another pass?
                         # No, if we added i, it only increases counts.
                         # So any triple that was NOT addable before this pass is still NOT addable.
                         # Any triple that WAS addable before this pass and was processed after i
                         # might now be blocked.
                         # Any triple that WAS addable before this pass and was processed before i
                         # was already added.
                         # So after one full pass, there are NO addable triples left.
                         # Because:
                         # - If j was addable at start of pass:
                         #   - If j processed before blocker i: j added.
                         #   - If j processed after blocker i: j might be blocked.
                         #     But if j is blocked, it's not addable anymore.
                         #     Is it possible that j was addable, i was added, j became blocked,
                         #     but some k (processed after j) is now addable?
                         #     No, adding i only increases counts. It cannot make k addable.
                         #     So if k was not addable before i, it's not addable now.
                         #     If k was addable before i, it would have been added if processed.
                         #     If k processed after i, and k is still addable, it will be added.
                         #     If k is blocked by i, it won't be added.
                         #     So after the pass, no triple is addable.
                         # Therefore, we do NOT need to restart the while loop just because we added.
                         # We can just proceed to swaps.
                         # But `improved` is True, so we go to next iteration.
                         # In next iteration, Phase 1 will find nothing to add.
                         # So we can just set improved=False if we want to go to swaps immediately?
                         # No, let's keep the logic simple.
                         # If we added, improved=True. Next iteration Phase 1 finds nothing.
                         # Then we go to swaps.
                         # This is fine.
            
            # Phase 2: Swaps (1-for-1)
            # Try to remove one active triple and add one inactive triple.
            # This is useful to escape local optima where no single addition is possible,
            # but a swap is.
            
            # We need to find a pair (t_remove, t_add) such that:
            # 1. t_remove is active.
            # 2. t_add is inactive.
            # 3. Removing t_remove makes t_add addable.
            #    This means: For all k in K222s(t_add), counts[k] < 7 AFTER removal.
            #    i.e., counts[k] - (1 if k in K222s(t_remove) else 0) < 7.
            #    Since counts[k] <= 7 always (valid state), this means:
            #    If counts[k] == 7, then k MUST be in K222s(t_remove).
            #    If counts[k] < 7, it's fine.
            
            # So the condition is:
            # Blockers(t_add) = { k in K222s(t_add) | counts[k] == 7 }
            # Condition: Blockers(t_add) is a subset of K222s(t_remove).
            
            # Strategy:
            # Iterate over inactive t_add.
            # Compute Blockers(t_add).
            # If Blockers is empty, t_add is addable (should have been caught in Phase 1).
            # If Blockers is not empty, find an active t_remove such that Blockers(t_add) subset of K222s(t_remove).
            
            # To speed up, we can precompute for each K222 index, the list of active triples containing it?
            # Or for each triple, the list of K222s it contains.
            
            # Let's try a randomized approach for swaps to avoid O(N^2) per iteration.
            # Or a structured one.
            
            # Given N=560, N^2 is 313,600.
            # Checking subset is O(|Blockers|) which is small.
            # So O(N^2) per swap attempt might be too slow if we do many.
            # But we only do swaps if Phase 1 failed.
            
            # Let's limit swap attempts per iteration.
            
            inactive_indices = [i for i in range(N_TRIPLES) if not active[i]]
            active_indices = [i for i in range(N_TRIPLES) if active[i]]
            
            if not inactive_indices or not active_indices:
                break
                
            random.shuffle(inactive_indices)
            
            # Precompute K222 sets for active triples?
            # triple_to_k222s[i] is a list. Converting to set every time is expensive.
            # But we can just check `k in triple_to_k222s[t_remove]`.
            # Since lists are ~1700 long, `in` is O(1700).
            # This is too slow.
            
            # Optimization: Use bitmasks or sets for K222 membership?
            # N_K222 = 120,120.
            # A bitmask of 120k bits is 15KB.
            # 560 triples * 15KB = 8.4 MB.
            # This is feasible.
            
            # Let's create a global list of bitmasks for K222 membership.
            # triple_k222_mask[i] is an integer bitmask where bit k is 1 if triple i is in K222 k.
            # Wait, 120,120 bits is huge for a Python int?
            # Python ints are arbitrary precision. 120k bits is ~15KB.
            # Operations like `&` are fast.
            
            # Let's precompute these masks.
            
            # Actually, we can precompute them once in the module scope or in construct.
            
            # Let's do it in construct to keep module load fast?
            # Or just use sets?
            # Sets of 1700 ints.
            # `issubset` is fast if the smaller set is small.
            # Blockers(t_add) is usually small (1-3 K222s).
            # So `blockers.issubset(set(triple_to_k222s[t_remove]))` might be okay?
            # Creating the set for t_remove is expensive.
            
            # Alternative:
            # For each K222 k, maintain a count.
            # For each triple i, we have the list of K222s.
            
            # Let's stick to the list check but optimize by checking the smallest blocker first.
            
            found_swap = False
            for t_add in inactive_indices:
                if time_left() < 0.5:
                    return active, counts, current_size
                
                # Find blockers
                blockers = []
                for k_idx in triple_to_k222s[t_add]:
                    if counts[k_idx] == 7:
                        blockers.append(k_idx)
                
                if not blockers:
                    # Should be addable, but maybe Phase 1 missed it?
                    # Add it.
                    active[t_add] = True
                    current_size += 1
                    for k_idx in triple_to_k222s[t_add]:
                        counts[k_idx] += 1
                    improved = True
                    found_swap = True
                    break
                
                # Find t_remove: must contain ALL blockers
                # Candidates: triples in intersection of k222_triples[k] for all blockers
                if len(blockers) == 1:
                    candidates = k222_triples[blockers[0]]
                else:
                    candidates = set(k222_triples[blockers[0]])
                    for k_block in blockers[1:]:
                        candidates.intersection_update(k222_triples[k_block])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                    candidates = list(candidates)
                
                # Find an active t_remove in candidates
                for t_remove in candidates:
                    if active[t_remove]:
                        # Perform swap
                        active[t_remove] = False
                        for k_idx in triple_to_k222s[t_remove]:
                            counts[k_idx] -= 1
                        
                        active[t_add] = True
                        for k_idx in triple_to_k222s[t_add]:
                            counts[k_idx] += 1
                        
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            # If no 1-for-1 swap found, try 2-for-1 (remove 2, add 1) or 1-for-2
            # This helps escape local optima
            if not found_swap and time_left() > 2.0:
                # Try 2-for-1: find t_add that needs 2 removals
                # Or 1-for-2: find t_remove whose removal allows adding 2 triples
                # Let's try a simple kick: remove a random active triple, then greedily add
                if random.random() < 0.3 and len(active_indices) > 5:
                    # Kick: remove 1-3 random active triples
                    kick_size = random.randint(1, 3)
                    kick_triples = random.sample(active_indices, min(kick_size, len(active_indices)))
                    
                    # Save state
                    saved_active = active[:]
                    saved_counts = counts[:]
                    saved_size = current_size
                    
                    # Remove kick triples
                    for t in kick_triples:
                        active[t] = False
                        current_size -= 1
                        for k_idx in triple_to_k222s[t]:
                            counts[k_idx] -= 1
                    
                    # Greedily add back as many as possible
                    for i in range(N_TRIPLES):
                        if not active[i]:
                            addable = True
                            for k_idx in triple_to_k222s[i]:
                                if counts[k_idx] == 7:
                                    addable = False
                                    break
                            if addable:
                                active[i] = True
                                current_size += 1
                                for k_idx in triple_to_k222s[i]:
                                    counts[k_idx] += 1
                    
                    # Keep the kick if it improved or maintained size
                    if current_size >= saved_size:
                        improved = True
                    else:
                        # Revert
                        active = saved_active
                        counts = saved_counts
                        current_size = saved_size
            
            if not improved:
                # No swap found in this iteration
                # Try 2-for-1 or 1-for-2?
                # Or just break to outer loop?
                # If we break, we return.
                # But maybe a different random shuffle would find a swap?
                # Let's do a few more iterations of random swaps?
                # Or just break.
                break
        
        return active, counts, current_size

    # Main Loop
    # Run multiple attempts with different seeds
    seed = 42
    while time_left() > 5.0:
        active, counts, current_size = greedy_init(seed)
        active, counts, current_size = local_search(active, counts, current_size)
        
        if current_size > best_size:
            best_size = current_size
            best_active = active[:]
            best_counts = counts[:]
        
        seed += 1
        
        # Periodically perturb the best solution and re-search
        if seed % 5 == 0 and time_left() > 10.0 and best_size > 0:
            perturbed_active = best_active[:]
            perturbed_counts = best_counts[:]
            perturbed_size = best_size
            
            # Remove 3-8 random active triples
            active_list = [i for i in range(N_TRIPLES) if perturbed_active[i]]
            if len(active_list) > 10:
                remove_count = random.randint(3, min(8, len(active_list) // 3))
                to_remove = random.sample(active_list, remove_count)
                for t in to_remove:
                    perturbed_active[t] = False
                    perturbed_size -= 1
                    for k_idx in triple_to_k222s[t]:
                        perturbed_counts[k_idx] -= 1
                
                # Re-run local search from perturbed state
                perturbed_active, perturbed_counts, perturbed_size = local_search(
                    perturbed_active, perturbed_counts, perturbed_size
                )
                
                if perturbed_size > best_size:
                    best_size = perturbed_size
                    best_active = perturbed_active
                    best_counts = perturbed_counts
    
    # Construct result
    result = []
    for i in range(N_TRIPLES):
        if best_active[i]:
            result.append(list(ALL_TRIPLES[i]))
    
    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
