import time
import random
import math

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    # Point representation: index 0..4095
    # x = i // N, y = i % N
    num_points = N * N
    
    # Data structures for incremental validity checks
    selected = []
    point_set = set()
    # dist_sets[q] = set of squared distances from q to other selected points
    dist_sets = {}
    
    def _dist2_idx(i, j):
        xi = i // N
        yi = i % N
        xj = j // N
        yj = j % N
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy

    def _can_add(p_idx):
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique (no isosceles with vertex p)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx] (no isosceles with vertex q)
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        point_set.remove(p_idx)
        # Remove from list (O(K), K is small)
        try:
            idx = selected.index(p_idx)
            selected.pop(idx)
        except ValueError:
            pass
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[q_idx].remove(d)
        
        del dist_sets[p_idx]

    rng = random.Random(12345)
    
    # 1. Greedy Initialization
    # Try a few different greedy orders to find a good starting point
    best_size = 0
    best_state = None
    
    # We will run the main search loop, but start with a greedy fill
    candidates = list(range(num_points))
    
    # Perform a greedy fill with a fixed seed first to ensure a baseline
    rng_init = random.Random(42)
    rng_init.shuffle(candidates)
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected)
    # Save state
    best_selected = selected[:]
    best_point_set = point_set.copy()
    best_dist_sets = {k: set(v) for k, v in dist_sets.items()}

    # 2. Local Search / Simulated Annealing
    # We will try to improve upon the current state.
    # Moves:
    # - Add: Pick random unselected point, try to add.
    # - Remove: Pick random selected point, remove it.
    
    # To make "Add" efficient, we don't iterate all unselected. We pick random indices.
    # To make "Remove" strategic, we remove random points when stuck.
    
    stuck_count = 0
    max_stuck_before_remove = 200 # Try 200 random adds before forcing a removal
    temp = 1.0
    cooling = 0.9999
    
    # We need to be careful with time.
    # Each _can_add is O(K). K is ~10-20.
    # 200 adds * 20 ops = 4000 ops per "stuck" phase.
    # This is very fast. We can do thousands of these.
    
    while time.time() < deadline:
        # 1. Try to add points
        added_any = False
        
        # Try a batch of random additions
        # The number of tries can be dynamic based on stuck count, but fixed is safer
        for _ in range(100):
            if time.time() > deadline:
                break
                
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in point_set:
                continue
                
            if _can_add(p_idx):
                _add(p_idx)
                added_any = True
                stuck_count = 0
                # If we added, we might be able to add more, so continue
                # But to keep loop tight, we break out of this inner batch if we added?
                # No, let's keep trying to add more in this batch.
                
        if added_any:
            # Update best if improved
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected[:]
                best_point_set = point_set.copy()
                best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
            continue # Go back to start of while loop, try adding more
        
        # 2. If no adds, we are stuck
        stuck_count += 1
        
        # If stuck for a while, try to remove a point to escape local optimum
        # Probability of removal increases with stuck_count
        # Or just remove if stuck_count > threshold
        if stuck_count > max_stuck_before_remove:
            # Remove a random point
            if selected:
                # Pick a random point to remove
                remove_idx = rng.randint(0, len(selected) - 1)
                p_idx_to_remove = selected[remove_idx]
                _remove(p_idx_to_remove)
                
                # After removing, we are "unstuck" potentially, so reset counter
                stuck_count = 0
                
                # Immediately try to add points back to see if we can recover
                # This is a "kick"
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    p_idx = rng.randint(0, num_points - 1)
                    if p_idx in point_set:
                        continue
                    if _can_add(p_idx):
                        _add(p_idx)
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_selected = selected[:]
                            best_point_set = point_set.copy()
                            best_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                
                # Cool down temperature if we were using it, but we are using a simpler heuristic
                # We can also reset the state to best if we degrade too much?
                # For now, we just continue searching from the current state.
                
        # If we haven't removed, we just loop and try adding again.
        # To avoid infinite loop of failing adds without progress, the stuck_count handles it.
        
        # Optional: If stuck for a very long time, restart from best state?
        # This might be useful if the current state is much worse than best.
        if stuck_count > 1000:
             # Restore best state
             selected[:] = best_selected
             point_set.clear()
             point_set.update(best_point_set)
             dist_sets.clear()
             for k, v in best_dist_sets.items():
                 dist_sets[k] = set(v)
             stuck_count = 0
             # Also, to avoid same failure, we can slightly perturb the "best" state?
             # Or just rely on the randomness of the add attempts.
             # Let's just reset and continue.

    # Convert best_selected indices back to (x, y) tuples
    result = []
    for idx in best_selected:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result
# EVOLVE-BLOCK-END
