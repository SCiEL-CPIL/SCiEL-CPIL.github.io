import itertools
import random
import sys
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles relationships
# pair_to_qs: dict mapping (idx_i, idx_j) to set of idx_k such that {i,j,k} is isosceles
pair_to_qs = {}

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = POINTS[i]
        p_j = POINTS[j]
        d_ij = dist2(p_i, p_j)
        
        blocked = set()
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p_k = POINTS[k]
            d_ik = dist2(p_i, p_k)
            d_jk = dist2(p_j, p_k)
            
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                blocked.add(k)
        
        pair_to_qs[(i, j)] = blocked
        pair_to_qs[(j, i)] = blocked

def construct():
    best_size = NUM_POINTS + 1
    best_set = None
    
    # Run multiple restarts to escape local minima
    num_restarts = 10
    start_time = time.time()
    time_limit = 250  # Leave some time for finalization
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = set()
        blocked_count = [0] * NUM_POINTS
        
        def is_unblocked(q):
            return q not in S and blocked_count[q] == 0

        def add_point(p):
            S.add(p)
            for r in S:
                if r == p:
                    continue
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and q not in S:
                        blocked_count[q] += 1

        def remove_point(p):
            S.remove(p)
            for r in S:
                key = (p, r) if p < r else (r, p)
                for q in pair_to_qs[key]:
                    if q != p and q not in S:
                        blocked_count[q] -= 1

        # 1. Greedy Construction
        # Randomize the order of points to get different initial sets
        candidates = list(range(NUM_POINTS))
        random.shuffle(candidates)
        
        # We iterate through candidates, but we might add points not in the current "unblocked" list?
        # No, standard greedy: at each step, pick the best from unblocked.
        # To introduce randomness, we can shuffle the "unblocked" list or break ties randomly.
        # Let's shuffle the global list of indices to break symmetry in tie-breaking.
        # Actually, the current code picks the one with max block count. If tie, it picks the first one found in the loop.
        # The loop iterates `for p in unblocked`. `unblocked` is built by iterating `range(NUM_POINTS)`.
        # So it's deterministic unless we shuffle the range.
        
        while True:
            unblocked = [q for q in range(NUM_POINTS) if is_unblocked(q)]
            if not unblocked:
                break
            
            # Select best candidate
            best_p = None
            best_score = -1
            best_ties = []
            
            # Shuffle unblocked to randomize tie-breaking
            random.shuffle(unblocked)
            
            for p in unblocked:
                new_blocked = 0
                for r in S:
                    key = (p, r) if p < r else (r, p)
                    for q in pair_to_qs[key]:
                        if q != p and is_unblocked(q):
                            new_blocked += 1
                if new_blocked > best_score:
                    best_score = new_blocked
                    best_ties = [p]
                elif new_blocked == best_score:
                    best_ties.append(p)
            
            if best_ties:
                best_p = random.choice(best_ties)
            else:
                best_p = None
                
            if best_p is None:
                break
            add_point(best_p)

        # 2. Local Search
        # We try to minimize size.
        # Moves:
        # A. Remove p if S\{p} is maximal. (Size -1)
        # B. Swap p with q if (S\{p\} U {q}) is maximal. (Size 0, changes config)
        
        improved = True
        # To avoid infinite loops in swaps, we limit the number of swap-only iterations
        # or use a tabu list. Simple approach: if no improvement in a full pass of removals,
        # do one pass of swaps. If a swap is made, restart the removal pass.
        # If no swap is found, stop.
        
        while improved and (time.time() - start_time < time_limit):
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            # Try Removals
            for p in s_list:
                if p not in S:
                    continue
                
                old_S = set(S)
                old_blocked = blocked_count[:]
                
                remove_point(p)
                
                # Check if maximal
                is_max = True
                for q in range(NUM_POINTS):
                    if is_unblocked(q):
                        is_max = False
                        break
                
                if is_max:
                    improved = True
                    # State is already updated (S and blocked_count reflect removal)
                    break # Restart loop with new S
                else:
                    # Restore
                    S.clear()
                    S.update(old_S)
                    for i in range(NUM_POINTS):
                        blocked_count[i] = old_blocked[i]
            
            if improved:
                continue
                
            # Try Swaps
            # If no removal worked, try to find a valid swap to escape local optimum
            swap_found = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p in s_list:
                if swap_found:
                    break
                if p not in S:
                    continue
                    
                old_S = set(S)
                old_blocked = blocked_count[:]
                
                remove_point(p)
                
                # Try to add some q not in S
                # We want (S\{p\} U {q}) to be maximal.
                # Note: S\{p\} is not maximal, so there are unblocked points.
                # Adding q might block them.
                
                # Candidate q's:
                # Ideally, q should be one of the currently unblocked points?
                # Or any point not in S\{p\}.
                # If we add q, it must not form isosceles with S\{p\}.
                # So q must be unblocked in S\{p\}.
                
                unblocked_after_remove = [q for q in range(NUM_POINTS) if is_unblocked(q)]
                
                if not unblocked_after_remove:
                    # This implies S\{p\} was maximal, which we already checked and rejected?
                    # Wait, if S\{p\} is maximal, we would have accepted the removal.
                    # So if we are here, S\{p\} is NOT maximal, so unblocked_after_remove is not empty.
                    pass
                
                random.shuffle(unblocked_after_remove)
                
                for q in unblocked_after_remove:
                    # Try adding q
                    add_point(q)
                    
                    # Check if maximal
                    is_max = True
                    for r in range(NUM_POINTS):
                        if is_unblocked(r):
                            is_max = False
                            break
                    
                    if is_max:
                        # Valid swap
                        swap_found = True
                        improved = True # We changed the set, treat as progress to restart search
                        # State is now S\{p\} U {q}
                        break
                    else:
                        # Restore to S\{p\} state? No, we need to restore to original S to try next q?
                        # Or we can just restore to original S and break?
                        # It's more efficient to try multiple q's for the same p.
                        # But `add_point` modified S and blocked_count.
                        # We need to undo `add_point(q)`.
                        remove_point(q)
                        
                        # Now S is S\{p\}.
                        # We can try next q.
                        continue
                
                if swap_found:
                    break
                
                # If no q worked for this p, restore to original S
                S.clear()
                S.update(old_S)
                for i in range(NUM_POINTS):
                    blocked_count[i] = old_blocked[i]
            
            if not swap_found:
                # No moves possible
                break

        # End of local search for this restart
        if len(S) < best_size:
            best_size = len(S)
            best_set = set(S)

    if best_set is None:
        # Fallback: return a simple greedy result if search failed or timed out badly
        # But construct() must return a valid set.
        # The loop above should always produce *some* set.
        # If best_set is still None, it means no restart completed?
        # Unlikely, but let's handle it.
        # Re-run one simple greedy
        S = set()
        blocked_count = [0] * NUM_POINTS
        # ... (duplicate greedy logic) ...
        # For safety, just return the last S from the loop if best_set is None?
        # The loop sets best_set if len(S) < best_size.
        # If all restarts failed to produce a valid set (impossible, greedy always produces a maximal set),
        # best_set will be set.
        
        # If time ran out during the first restart, best_set might be set.
        # If time ran out before any restart finished?
        # The `while True` in greedy might take long? No, it's fast.
        # Local search might take long.
        # If we break out of restart loop due to time, best_set might be from a previous restart.
        # If no restart finished, best_set is None.
        
        # Let's ensure we have a valid set.
        if best_set is None:
             # Just do a quick greedy
             S = set()
             blocked_count = [0] * NUM_POINTS
             while True:
                 unblocked = [q for q in range(NUM_POINTS) if q not in S and blocked_count[q] == 0]
                 if not unblocked: break
                 best_p = unblocked[0] # Just pick first
                 # Proper greedy
                 best_score = -1
                 for p in unblocked:
                     nb = 0
                     for r in S:
                         key = (p, r) if p < r else (r, p)
                         for q in pair_to_qs[key]:
                             if q != p and q not in S:
                                 nb += 1
                     if nb > best_score:
                         best_score = nb
                         best_p = p
                 S.add(best_p)
                 for r in S:
                     if r == best_p: continue
                     key = (best_p, r) if best_p < r else (r, best_p)
                     for q in pair_to_qs[key]:
                         if q != best_p and q not in S:
                             blocked_count[q] += 1
             best_set = S

    return [POINTS[q] for q in best_set]
