import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)

dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
# Use int32 to save memory and speed up operations, though int64 is safer for large N.
# N=64 -> max dist2 = 2*63^2 = 7938, fits in int16 even. int32 is safe.
dist2 = (dx * dx + dy * dy).astype(np.int32)

# We need to check if a set of points is "no-isosceles".
# Condition: For every point P in S, the distances d(P, Q) for all Q in S, Q!=P must be distinct.
# This is equivalent to: In the submatrix of dist2 induced by S, every row (and col) 
# has distinct non-diagonal entries.

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = []
    best_size = 0
    
    # Heuristic ordering: 
    # Points with more "constraints" or "conflicts" should be decided earlier?
    # Actually, MRV says: pick the variable with the smallest domain.
    # Here, all points start with domain {0, 1}.
    # However, we can prioritize points that are "central" or have many close neighbors,
    # as they are more likely to cause conflicts.
    # Let's sort points by their number of close neighbors (e.g., dist2 < threshold).
    # Or just random order with restarts, but with backtracking.
    
    # To make backtracking efficient, we need a fast way to check validity.
    # State: 
    # - selected: set of indices
    # - For each selected point p, we keep a set of distances to other selected points: dists_from_p
    
    # Backtracking algorithm:
    # def backtrack(idx, selected, dists_from_p, current_size):
    #   if idx == TOTAL: return current_size
    #   # Try adding idx
    #   # Check if idx can be added:
    #   # 1. Distances from idx to all in selected must be distinct among themselves.
    #   # 2. For each p in selected, dist(idx, p) must not be in dists_from_p.
    #   # If valid, recurse.
    #   # Try not adding idx (recurse with idx+1)
    
    # However, simple backtracking on 4096 variables is too slow even with pruning.
    # We need a hybrid: 
    # 1. Use a strong heuristic to select the NEXT point to consider, not just sequential index.
    # 2. Use "Lookahead": If adding a point blocks many others, prefer not adding it?
    
    # Alternative: Local Search with Backtracking (Simulated Annealing style with restarts)
    # Let's try a different structural approach: 
    # **Conflict-Driven Backtracking (CDCL-inspired)**
    # We maintain a set of selected points.
    # We try to add points. If a conflict arises, we identify a "reason" (a subset of points causing the conflict).
    # We learn a clause (negative clause) that prevents this combination.
    # This is complex to implement from scratch.
    
    # Let's stick to a robust **Greedy with Backtracking** (Depth Limited).
    # We will use a stack to allow undoing decisions.
    
    # Optimization: 
    # Instead of checking all previous points, we maintain `dists_from_p` as a set.
    # Adding point p:
    # 1. Compute d_p = dist2[p, selected].
    # 2. Check if len(set(d_p)) == len(selected). If not, conflict at p.
    # 3. Check if any d in d_p is in dists_from_p[i] for corresponding i. If so, conflict at i.
    
    # To speed up, we can pre-sort candidates.
    
    # Let's try a **Randomized Backtracking** with a limit on depth of backtracking.
    # This is essentially a greedy algorithm that allows "undoing" the last few decisions.
    
    def try_add(p_idx, selected_list, dist_sets):
        """
        Tries to add p_idx to the current state.
        Returns (success, new_dist_sets, conflict_info)
        conflict_info: list of indices in selected_list that caused conflict, or [p_idx] if self-conflict
        """
        k = len(selected_list)
        if k == 0:
            return True, [set()], []
        
        # Get distances to all selected points
        # selected_list is a list of indices
        # We need dist2[p_idx, selected_list]
        # Since dist2 is a global matrix, this is a fancy index.
        d_p = dist2[p_idx, selected_list]
        
        # Check 1: Self-distinctness
        # If any distance repeats in d_p, conflict.
        if len(set(d_p.tolist())) != k:
            return False, None, [p_idx]
        
        # Check 2: Cross-distinctness
        # For each i in 0..k-1, d_p[i] must not be in dist_sets[i]
        conflict_indices = []
        for i in range(k):
            if d_p[i] in dist_sets[i]:
                conflict_indices.append(i)
        
        if conflict_indices:
            return False, None, conflict_indices
        
        # Success
        # Update dist_sets
        # New dist_sets:
        # For existing i, add d_p[i] to dist_sets[i]
        # For new point, add set(d_p)
        
        # We must copy the sets to allow backtracking (undo).
        # This is expensive if k is large.
        # Optimization: Use persistent data structures or just copy lists if k is small.
        # Given the score ~10, k is likely small (< 20-30). Copying sets of size < 30 is fast.
        
        new_dist_sets = []
        for i in range(k):
            new_set = set(dist_sets[i])
            new_set.add(d_p[i])
            new_dist_sets.append(new_set)
        
        new_dist_sets.append(set(d_p.tolist()))
        
        return True, new_dist_sets, []

    # Main Search Loop
    # We will perform multiple random restarts.
    # In each restart, we use a backtracking search with a limited depth (e.g., backtrack up to 5 steps).
    
    all_indices = list(range(TOTAL))
    
    # Pre-compute a "conflict potential" score for each point to order them.
    # Points with many close neighbors are more restrictive.
    # Let's just use random order but with backtracking.
    
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        random.shuffle(all_indices)
        
        # Stack for backtracking
        # Each entry: (index_in_all_indices, selected_list, dist_sets)
        # To save memory, we store the state compactly.
        
        current_selected = []
        current_dist_sets = []
        path = [] # List of (p_idx, dist_sets_before)
        
        max_backtrack_depth = 5
        
        for p_idx in all_indices:
            if time.time() > deadline:
                break
            
            success, new_dist_sets, conflicts = try_add(p_idx, current_selected, current_dist_sets)
            
            if success:
                # Commit
                path.append((p_idx, current_dist_sets))
                current_selected.append(p_idx)
                current_dist_sets = new_dist_sets
            else:
                # Conflict
                # If we have backtracked too many times recently, restart?
                # Or try to backtrack.
                
                # Simple heuristic: If conflict, just skip this point (Greedy).
                # To add backtracking, we need to "undo" a previous decision.
                # Let's implement a simple "kick":
                # If we have a conflict and we have room to backtrack, 
                # we can remove a point from the end and try a different one?
                # This is complex.
                
                # Let's try a different strategy: **Iterative Deepening A* (IDA*)** or **Simulated Annealing**.
                # Given the time limit, let's stick to a very fast Greedy but with **Lookahead**.
                # Lookahead: Before adding a point, check if it blocks too many future points?
                # Too slow.
                
                # Let's try **Randomized Local Search** (WalkSAT style).
                # Start with a random valid set (generated by greedy).
                # Then perform moves:
                # 1. Remove a random point.
                # 2. Try to add a random point from the removed set or unselected set.
                # 3. If size increases, keep. If same, keep with prob p. If decreases, revert.
                
                pass # Placeholder for logic below

        # If we just did greedy, we are back to the seed.
        # We need the backtracking logic.
        
        # Let's implement a proper backtracking search with a limit.
        # We will use a recursive function inside the loop to keep state local.
        
        def backtrack_search(order, pos, selected, dist_sets, depth_limit):
            if pos == len(order):
                return len(selected)
            
            if time.time() > deadline:
                return len(selected)
            
            p_idx = order[pos]
            
            # Try adding
            success, new_dist_sets, _ = try_add(p_idx, selected, dist_sets)
            
            if success:
                # Recurse
                size = backtrack_search(order, pos + 1, selected + [p_idx], new_dist_sets, depth_limit)
                if size > len(selected):
                    return size
                # Backtrack
                # Note: selected + [p_idx] creates a new list, so we don't need to explicitly pop.
                # But dist_sets is passed by reference in the closure? No, we pass new_dist_sets.
                # So the state is immutable in this functional style.
                
            # Try not adding
            size = backtrack_search(order, pos + 1, selected, dist_sets, depth_limit)
            
            return size

        # This pure backtracking is O(2^N) in worst case. Even with pruning, it might be too slow for N=4096.
        # We need to restrict the search.
        
        # Revised Plan:
        # 1. Generate a good initial set using Greedy (fast).
        # 2. Use Local Search to improve it.
        #    Moves:
        #    - Swap: Remove one point, add another.
        #    - Add: Add a point (if possible).
        #    - Remove: Remove a point (to allow adding two later? No, we want max size).
        
        # Let's implement Local Search.
        
        # First, get a baseline greedy set.
        greedy_selected = []
        greedy_dist_sets = []
        
        for p_idx in all_indices:
            success, new_dist_sets, _ = try_add(p_idx, greedy_selected, greedy_dist_sets)
            if success:
                greedy_selected.append(p_idx)
                greedy_dist_sets = new_dist_sets
        
        current_selected = greedy_selected
        current_dist_sets = greedy_dist_sets
        current_size = len(current_selected)
        
        if current_size > best_size:
            best_size = current_size
            best_set = [(idx // N, idx % N) for idx in current_selected]
        
        # Local Search Phase
        # We maintain a set of "rejected" points that we might want to add later.
        # We also try to swap points.
        
        unselected = set(all_indices) - set(current_selected)
        
        # To make local search fast, we need to quickly check if a point can be added.
        # try_add is O(K).
        
        max_iter = 10000
        iter_count = 0
        no_improve_count = 0
        
        while iter_count < max_iter and time.time() < deadline - 10:
            iter_count += 1
            
            # 1. Try to add a random unselected point
            if unselected:
                p_idx = random.choice(list(unselected))
                success, new_dist_sets, _ = try_add(p_idx, current_selected, current_dist_sets)
                if success:
                    current_selected.append(p_idx)
                    current_dist_sets = new_dist_sets
                    unselected.remove(p_idx)
                    current_size += 1
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_set = [(idx // N, idx % N) for idx in current_selected]
                    continue
            
            # 2. Try to swap: Remove a random selected point, then try to add a random unselected point.
            # Actually, swapping might not increase size. We want to increase size.
            # So: Remove a point P. Try to add 2 points Q, R.
            # This is "2-opt" style.
            
            if current_size > 0:
                # Pick a point to remove
                rem_idx_pos = random.randint(0, len(current_selected) - 1)
                rem_p = current_selected[rem_idx_pos]
                
                # Remove it
                # We need to reconstruct dist_sets without rem_p.
                # This is O(K^2) if we rebuild from scratch.
                # Or we can just rebuild from the list of points.
                
                # Rebuild state without rem_p
                new_selected = [p for i, p in enumerate(current_selected) if i != rem_idx_pos]
                new_dist_sets = []
                for p in new_selected:
                    # Distances to all other points in new_selected
                    # This is O(K^2) to compute all distances.
                    # Since K is small (~10-20), this is acceptable.
                    ds = set()
                    for q in new_selected:
                        if p != q:
                            ds.add(dist2[p, q])
                    new_dist_sets.append(ds)
                
                current_selected = new_selected
                current_dist_sets = new_dist_sets
                current_size -= 1
                unselected.add(rem_p)
                
                # Try to add 2 points
                added_any = False
                for _ in range(2):
                    if not unselected:
                        break
                    # Try a few random candidates
                    for _ in range(10):
                        if not unselected:
                            break
                        p_idx = random.choice(list(unselected))
                        success, new_dist_sets, _ = try_add(p_idx, current_selected, current_dist_sets)
                        if success:
                            current_selected.append(p_idx)
                            current_dist_sets = new_dist_sets
                            unselected.remove(p_idx)
                            current_size += 1
                            added_any = True
                            break
                
                if added_any:
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_set = [(idx // N, idx % N) for idx in current_selected]
                else:
                    no_improve_count += 1
                    # If we didn't improve, revert? 
                    # Or just keep the state and try again.
                    # If we keep the state, we might get stuck in a local minimum.
                    # Let's restart local search if no improvement for X steps.
                    if no_improve_count > 50:
                        break
            
            if time.time() > deadline - 5:
                break

    return best_set

# Run a quick self-test to ensure it works
if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
