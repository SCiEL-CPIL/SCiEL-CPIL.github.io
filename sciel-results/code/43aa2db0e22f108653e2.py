import time
import numpy as np
import random
import itertools

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    # Parameters specific to the task instance expected: q=3, n=8, radius=1
    # If parameters differ, we fall back to a simple greedy.
    if q != 3 or n != 8 or radius != 1:
        return _simple_greedy(q, n, radius)

    total = q ** n  # 6561
    num_points = total
    
    # 1. Precompute balls and neighbor lists
    # ball[x] is a list of points covered by codeword x (radius 1)
    # We also need the inverse: for each point y, which balls cover it?
    # For ILS, we need fast "uncovered" counts.
    
    # Since N=6561 is small, we can store balls as numpy arrays of indices.
    # But for "count of uncovered", we need to track state.
    
    # Precompute powers of q for digit conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    # Generate all balls
    balls = []
    for x in range(total):
        ball = [x]
        for i in range(n):
            old_digit = (x // powers[i]) % q
            for new_digit in range(q):
                if new_digit != old_digit:
                    y = x + (new_digit - old_digit) * powers[i]
                    ball.append(y)
        balls.append(np.array(ball, dtype=np.int32))
        
    if time.time() > deadline - 5.0:
        return _int_to_digits([0] * 10, q, n) # Should not happen
        
    # 2. Initial Solution via Greedy
    # Greedy Set Cover: Pick the ball covering the most uncovered points.
    
    covered_mask = np.zeros(num_points, dtype=bool)
    uncovered_count_per_ball = np.array([len(b) for b in balls], dtype=np.int32)
    
    # For efficient update, we need to know, for each point y, which balls it belongs to.
    # inv[y] = list of ball indices containing y.
    # Since ball size is 17, and total points 6561, this is 6561 * 17 ~ 100k entries.
    inv = [[] for _ in range(num_points)]
    for i, b in enumerate(balls):
        for y in b:
            inv[y].append(i)
            
    current_selection = []
    while not np.all(covered_mask):
        if time.time() > deadline - 10.0:
            break
        # Find ball with max uncovered count
        max_idx = np.argmax(uncovered_count_per_ball)
        if uncovered_count_per_ball[max_idx] == 0:
            break
            
        current_selection.append(max_idx)
        # Update covered mask and counts
        ball = balls[max_idx]
        for y in ball:
            if not covered_mask[y]:
                covered_mask[y] = True
                # Decrement count for all balls covering y
                for i in inv[y]:
                    uncovered_count_per_ball[i] -= 1
                    
    # Ensure we actually covered everything (greedy should, but check)
    if not np.all(covered_mask):
        # Force add remaining?
        for i in range(total):
             if not np.all(covered_mask):
                 if i not in current_selection:
                     current_selection.append(i)
                     for y in balls[i]:
                         covered_mask[y] = True
                         for j in inv[y]:
                             uncovered_count_per_ball[j] -= 1
             else:
                 break

    best_selection = set(current_selection)
    best_size = len(best_selection)
    
    # 3. Iterated Local Search
    # State: set of selected codewords
    # Metric: size of set. Constraint: full coverage.
    
    # Helper to check coverage and find holes
    def get_uncovered(selection_set):
        # Returns boolean array of uncovered points
        # Using numpy boolean operations is faster than loops for large sets?
        # Let's just sum the contributions.
        # Actually, maintaining the count is faster during search.
        # For a fresh evaluation (after restart or big move), we recompute.
        # Since we do local search, we maintain incremental counts.
        
        # For now, let's use a fast recompute for simplicity in the ILS loop
        # if the number of swaps is not huge.
        # Better: maintain `uncovered_count` array globally and update it.
        
        # Let's switch to a stateful approach for the ILS loop.
        pass
        
    # Stateful ILS
    # State:
    #   selected: set of int indices
    #   covered_mask: np.bool_ array
    #   uncov_counts: np.int32 array of length total, where uncov_counts[i] is # uncovered points in ball[i]
    
    # Initialize state for ILS
    selected = set(best_selection)
    covered_mask = np.zeros(num_points, dtype=bool)
    # Recompute initial covered mask and counts from scratch for the selected set
    # This is O(|selected| * 17)
    for i in selected:
        for y in balls[i]:
            covered_mask[y] = True
            
    uncov_counts = np.array([len(b) for b in balls], dtype=np.int32)
    for i in range(num_points):
        if covered_mask[i]:
             for j in inv[i]:
                 uncov_counts[j] -= 1
                 
    # Verify no holes
    if not np.all(covered_mask):
         # Repair
         holes = np.where(~covered_mask)[0]
         for h in holes:
             # Find ball with max uncov count covering h?
             # Just pick best globally
             best_i = np.argmax(uncov_counts)
             if best_i not in selected:
                 selected.add(best_i)
                 for y in balls[best_i]:
                     if not covered_mask[y]:
                         covered_mask[y] = True
                         for j in inv[y]:
                             uncov_counts[j] -= 1
             # If best_i is already selected, that means it's not covering h? 
             # No, if h is uncovered, and best_i is selected, it means best_i doesn't cover h?
             # But uncov_counts counts uncovered in ball. If h is in ball[i], and h is uncovered, 
             # then uncov_counts[i] should be > 0.
             # So if we pick max uncov, it should cover some hole.
             # Let's loop until covered.
             while not np.all(covered_mask):
                 best_i = np.argmax(uncov_counts)
                 if uncov_counts[best_i] == 0:
                     break # Stuck? Should not happen
                 if best_i not in selected:
                     selected.add(best_i)
                     for y in balls[best_i]:
                         if not covered_mask[y]:
                             covered_mask[y] = True
                             for j in inv[y]:
                                 uncov_counts[j] -= 1
                 else:
                     # This should not happen if logic is correct
                     break
         
    current_size = len(selected)
    
    # Local Search Loop
    # Moves:
    # 1. Delete: If a codeword covers 0 unique uncovered points? 
    #    Actually, we maintain `uncov_counts`. 
    #    If we remove i, the new uncovered points are those y in ball[i] such that 
    #    y is NOT covered by any other selected codeword.
    #    We don't track "unique coverage" directly.
    #    Alternative: 
    #    Let's try to remove i. 
    #    Check if any y in ball[i] becomes uncovered.
    #    To do this fast, we can check the count of selected balls covering each y in ball[i].
    #    This requires maintaining a `coverage_depth[y]` = number of selected balls covering y.
    
    # Let's add coverage_depth to the state.
    coverage_depth = np.zeros(num_points, dtype=np.int32)
    for i in selected:
        for y in balls[i]:
            coverage_depth[y] += 1
            
    # Now deletion check is fast:
    # Can we delete i?
    # Yes if for all y in ball[i], coverage_depth[y] > 1.
    # i.e., min(coverage_depth[ball[i]]) > 1.
    
    # Swap check:
    # Remove i, Add j.
    # Valid if after removing i and adding j, no point is uncovered.
    # Equivalent: For all y, (depth[y] - 1_{y in ball[i]} + 1_{y in ball[j]}) >= 1.
    # Or: depth[y] - 1_{y in ball[i]} + 1_{y in ball[j]} >= 1
    # 1_{y in ball[i]} is 1 if y in ball[i] else 0.
    # So we need: depth[y] + 1_{y in ball[j]} - 1_{y in ball[i]} >= 1.
    # Violation if depth[y] == 1 and y in ball[i] and y NOT in ball[j].
    # So: No y such that (depth[y]==1 AND y in ball[i] AND y not in ball[j]).
    
    # We can precompute for each pair (i,j) if this swap is valid? 
    # Pairs are 6561^2 ~ 43M. Too many to precompute all.
    # But we can compute on the fly for candidate swaps.
    
    # Strategy:
    # 1. Try to delete any i where min(coverage_depth[ball[i]]) > 1.
    # 2. Try to swap i (where depth[i] is low?) with j.
    #    Candidate j should be one that covers many points currently covered only by i.
    #    Let S_i = { y in ball[i] | coverage_depth[y] == 1 }.
    #    If S_i is empty, i can be deleted.
    #    If not, we need a j such that S_i subset of ball[j].
    #    And we want to maximize "gain" or just find any valid swap that allows further deletions?
    #    Actually, a swap keeps the size the same. It's useful to "move" coverage to enable deletion.
    #    
    #    Let's do:
    #    1. Attempt deletions.
    #    2. If no deletion, pick a random i. Compute S_i.
    #       If S_i is empty, delete i (should have been caught in 1, unless depth changed? No).
    #       If S_i not empty, find j that covers S_i. 
    #       Among valid j's, pick one that maximizes coverage of points with depth > 1? 
    #       Or just pick random valid j?
    #       Actually, if we swap i->j, size stays same. But it might expose new deletion opportunities.
    #       
    #    Let's refine:
    #    Loop:
    #       improved = False
    #       # 1. Deletion
    #       for i in list(selected):
    #           if min(coverage_depth[balls[i]]) > 1:
    #               # Delete i
    #               for y in balls[i]:
    #                   coverage_depth[y] -= 1
    #               selected.remove(i)
    #               improved = True
    #               break # Restart loop
    #               
    #       if not improved:
    #           # 2. Swap
    #           # Pick random i
    #           i = random.choice(list(selected))
    #           S_i = balls[i][coverage_depth[balls[i]] == 1]
    #           if len(S_i) == 0:
    #               # Should have been deleted
    #               pass
    #           else:
    #               # Find j such that S_i subset ball[j]
    #               # Heuristic: Check candidates j that are "close" to i or cover high density areas?
    #               # For simplicity, check all j not in selected? 
    #               # 6561 candidates. For each, check if S_i subset ball[j].
    #               # S_i size is small (<= 17).
    #               # Check: all(y in S_i, y in ball[j])
    #               # We can precompute a bitmask for each ball?
    #               # N=6561. Bitmask needs 6561 bits ~ 1KB. 6561 * 1KB = 6MB. OK.
    #               # But Python sets are slower.
    #               # Let's stick to the array check.
    #               
    #               valid_js = []
    #               for j in range(total):
    #                   if j in selected: continue
    #                   # Check if ball[j] covers S_i
    #                   # S_i is an array of indices.
    #                   # We need to check if all elements in S_i are in balls[j].
    #                   # balls[j] is a numpy array. 
    #                   # Efficient check: 
    #                   # Create a boolean mask for ball[j]? No, too slow per iteration.
    #                   # Alternative: 
    #                   # We have `balls` as list of arrays.
    #                   # We can use `np.intersect1d(S_i, balls[j])` and check len?
    #                   # Or, since S_i is small, just `all(y in set(balls[j]) for y in S_i)`.
    #                   # Precomputing sets for balls is fast (17 elements).
    #                   pass
                 
    # Precompute ball sets for O(1) membership
    ball_sets = [set(b) for b in balls]
    
    # Precompute bitmasks? 
    # Let's use the set approach for now, it's N=6561, S_i size <= 17.
    # 6561 * 17 ops per swap check. 100k ops.
    # If we try 100 swaps, 10M ops. OK in Python? Maybe tight.
    # Let's optimize: Only check j that are "relevant".
    # Relevant j: those that share at least one point with S_i?
    # If j covers S_i, it must cover S_i[0].
    # So candidates for j are in inv[S_i[0]].
    # Size of inv[S_i[0]] is 17.
    # So we only check 17 candidates for each swap attempt!
    # This is HUGE speedup.
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Deletions
        # Iterate over a copy of selected
        for i in list(selected):
            # Check if i can be deleted
            # min(coverage_depth[balls[i]]) > 1
            if np.min(coverage_depth[balls[i]]) > 1:
                # Delete i
                for y in balls[i]:
                    coverage_depth[y] -= 1
                selected.remove(i)
                improved = True
                break # Restart deletion phase
        
        if improved:
            # Update best if smaller
            if len(selected) < best_size:
                best_size = len(selected)
                best_selection = set(selected)
            continue # Restart while loop
        
        # Phase 2: Swaps
        # Pick a random i to try to swap out
        # To be systematic, we could iterate, but random is fine for ILS
        i = random.choice(list(selected))
        
        # S_i: points covered ONLY by i
        s_i_mask = coverage_depth[balls[i]] == 1
        s_i_indices = balls[i][s_i_mask]
        
        if len(s_i_indices) == 0:
            # Should have been deleted in Phase 1
            # But maybe min was 1? No, if len is 0, min is inf? No, min of empty is error.
            # If s_i_indices is empty, it means all points in ball[i] have depth > 1?
            # Or depth 0? Depth 0 means uncovered, which is invalid.
            # So if we are valid, depth >= 1.
            # If min(depth) > 1, we delete.
            # If min(depth) == 1, then s_i_indices is not empty.
            # So if len(s_i_indices)==0, it implies min(depth) > 1, which should have triggered deletion.
            # Unless we have a bug. Let's assume it's a candidate for deletion and force it.
            for y in balls[i]:
                coverage_depth[y] -= 1
            selected.remove(i)
            improved = True
            if len(selected) < best_size:
                best_size = len(selected)
                best_selection = set(selected)
            continue
        
        # Find candidates j that cover s_i_indices
        # Candidates must cover s_i_indices[0]
        candidates = inv[s_i_indices[0]]
        
        swapped = False
        # Shuffle candidates to randomize
        random.shuffle(candidates)
        
        for j in candidates:
            if j in selected:
                continue
            # Check if j covers all of s_i_indices
            # j covers s if all(s in ball_sets[j])
            if all(s in ball_sets[j] for s in s_i_indices):
                # Valid swap i -> j
                # Execute swap
                # 1. Remove i
                for y in balls[i]:
                    coverage_depth[y] -= 1
                selected.remove(i)
                
                # 2. Add j
                for y in balls[j]:
                    coverage_depth[y] += 1
                selected.add(j)
                
                swapped = True
                break
        
        if swapped:
            # Swap doesn't change size, so best_size doesn't update immediately.
            # But it enables further deletions.
            # We don't mark improved here to force restart of while loop
            # which will check for deletions again.
            pass
        else:
            # No valid swap found for this i.
            # To avoid infinite loop if no swaps/deletions possible, we need a restart or accept stagnation.
            # If we can't improve, we should restart the ILS from a different random state or greedy state.
            # Let's count stagnation steps.
            # For simplicity, if no swap, we break out of this specific ILS run and start a new one.
            break

    # End of ILS run
    # Save best found so far
    if len(selected) < best_size:
        best_size = len(selected)
        best_selection = set(selected)
        
    # If we broke out due to stagnation, start a new ILS run from a fresh greedy?
    # Or just restart the while loop with a random perturbation?
    # Let's just start a new greedy initialization to diversify.
    
    # Check time
    if time.time() > deadline - 5.0:
        break
        
    # Re-initialize for next ILS run
    # Randomly perturb best_selection?
    # Or just run Greedy again?
    # Let's try to perturb the current best.
    # Remove k random elements from best_selection, then repair.
    
    # For now, let's just run a new Greedy from scratch to ensure diversity.
    # It's fast enough.
    
    # Re-run Greedy
    covered_mask = np.zeros(num_points, dtype=bool)
    uncovered_count_per_ball = np.array([len(b) for b in balls], dtype=np.int32)
    current_selection = []
    
    while not np.all(covered_mask) and time.time() < deadline - 5.0:
        max_idx = np.argmax(uncovered_count_per_ball)
        if uncovered_count_per_ball[max_idx] == 0:
            break
        current_selection.append(max_idx)
        for y in balls[max_idx]:
            if not covered_mask[y]:
                covered_mask[y] = True
                for i in inv[y]:
                    uncovered_count_per_ball[i] -= 1
                    
    # Init state for next ILS
    selected = set(current_selection)
    coverage_depth = np.zeros(num_points, dtype=np.int32)
    for i in selected:
        for y in balls[i]:
            coverage_depth[y] += 1
            
    # Ensure valid
    if not np.all(coverage_depth >= 1):
         holes = np.where(coverage_depth < 1)[0]
         for h in holes:
             best_i = np.argmax(uncovered_count_per_ball) # Approx
             # Just add any ball covering h
             for i in inv[h]:
                 if i not in selected:
                     selected.add(i)
                     for y in balls[i]:
                         coverage_depth[y] += 1
                     break

# Convert best_selection to digits
    codewords = []
    for idx in best_selection:
        digits = []
        x = idx
        for i in range(n):
            d = (x // powers[i]) % q
            digits.append(d)
        codewords.append(digits)
        
    return codewords

def _int_to_digits(ints, q, n):
    return [ [(x // powers[i]) % q for i in range(n)] for x in ints]

def _simple_greedy(q, n, radius):
    total = q ** n
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    balls = []
    for x in range(total):
        ball = {x}
        # Radius 1
        for i in range(n):
            old = (x // powers[i]) % q
            for new in range(q):
                if new != old:
                    y = x + (new - old) * powers[i]
                    ball.add(y)
        balls.append(list(ball))
        
    covered = set()
    selected = []
    remaining = list(range(total))
    
    while len(covered) < total:
        best_idx = -1
        best_count = -1
        for i in remaining:
            cnt = len(set(balls[i]) - covered)
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        selected.append(best_idx)
        covered.update(balls[best_idx])
        remaining.remove(best_idx)
        
    return [ [(x // powers[i]) % q for i in range(n)] for x in selected]

# EVOLVE-BLOCK-END
