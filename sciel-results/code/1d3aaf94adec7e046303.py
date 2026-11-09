"""Combined approach: minimum-degree greedy + local search refinement.
Builds a maximal no-isosceles set via greedy heuristic, then refines via
point-swap local search to escape local optima and reduce set size."""

import time
import random
import itertools

N = 16
DEADLINE_SECONDS = 290  # safety margin before 300s kill

def _sqdist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _is_iso_with_set(p, S):
    """Check if adding p to S creates any isosceles triangle with two points in S."""
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a = S[i]
            b = S[j]
            d_pa = _sqdist(p, a)
            d_pb = _sqdist(p, b)
            d_ab = _sqdist(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _can_add(p, S):
    """True if p can be added to S without creating an isosceles triangle."""
    return not _is_iso_with_set(p, S)

def _build_minimal_greedy(order_seed=0):
    """
    Greedy construction using minimum-degree heuristic.
    At each step, among all addable points, pick the one that blocks the fewest others.
    """
    rng = random.Random(order_seed)
    all_points = list(itertools.product(range(N), repeat=2))
    S = []
    addable = set(all_points)
    
    while addable:
        best_p = None
        best_score = float('inf')
        
        candidates = list(addable)
        rng.shuffle(candidates)
        
        for p in candidates:
            if not _can_add(p, S):
                continue
            
            new_S = S + [p]
            block_count = 0
            for q in addable:
                if q == p:
                    continue
                if not _can_add(q, new_S):
                    block_count += 1
            
            score = block_count + rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_p = p
        
        if best_p is None:
            break
        
        S.append(best_p)
        addable.discard(best_p)
        
        new_addable = set()
        for q in addable:
            if _can_add(q, S):
                new_addable.add(q)
        addable = new_addable
    
    return S

def _build_greedy_order(order):
    """Simple greedy in a given order."""
    S = []
    for p in order:
        if _can_add(p, S):
            S.append(p)
    return S

def _local_search(S, all_points, max_iterations=200, time_limit=30):
    """
    Local search refinement: try to reduce set size by:
    1. Removing a point and checking if set remains maximal (if not, try re-adding others)
    2. Swapping one point for another to maintain maximality with fewer total points
    """
    S = list(S)
    S_set = set(S)
    start_time = time.time()
    best_size = len(S)
    best_S = list(S)
    
    # Precompute which points can be added given current S
    def compute_addable(S_list):
        S_s = set(S_list)
        addable = set()
        for p in all_points:
            if p not in S_s and _can_add(p, S_list):
                addable.add(p)
        return addable
    
    addable = compute_addable(S)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing each point in S
        for idx, removed in enumerate(S):
            if time.time() - start_time > time_limit:
                break
            
            S_trial = S[:idx] + S[idx+1:]
            S_trial_set = set(S_trial)
            
            # After removal, the set may no longer be maximal.
            # Check if any previously-blocked point can now be added.
            # If the set is still maximal (no addable points), we've reduced size!
            # But we need to verify no isosceles was created (removal can't create new isosceles)
            # Removal only removes constraints, so S_trial is always no-isosceles.
            # The question is: is S_trial maximal?
            
            # Check if any point in all_points can be added to S_trial
            still_maximal = True
            for p in all_points:
                if p not in S_trial_set:
                    if _can_add(p, S_trial):
                        still_maximal = False
                        break
            
            if still_maximal:
                # We found a smaller maximal set!
                S = S_trial
                S_set = set(S)
                best_size = len(S)
                best_S = list(S)
                addable = compute_addable(S)
                break  # restart iteration after a successful removal
        
        # Also try swap: remove one, add another, if net effect reduces size
        # (This is harder; for now, removal-only local search)
        
        if len(S) == best_size:
            # No improvement this round; try random perturbations
            # Remove a random point, then greedily re-add to get a different maximal set
            if random.random() < 0.3:
                idx = random.randrange(len(S))
                S_trial = S[:idx] + S[idx+1:]
                # Greedily re-add points to make it maximal again
                S_rebuilt = list(S_trial)
                S_rebuilt_set = set(S_rebuilt)
                addable = compute_addable(S_rebuilt)
                # Greedily add points in random order until maximal
                rng_local = random.Random(iteration)
                add_list = list(addable)
                rng_local.shuffle(add_list)
                for p in add_list:
                    if _can_add(p, S_rebuilt):
                        S_rebuilt.append(p)
                        S_rebuilt_set.add(p)
                        # Recompute addable
                        addable = compute_addable(S_rebuilt)
                        if not addable:
                            break
                if len(S_rebuilt) < best_size:
                    S = S_rebuilt
                    S_set = set(S)
                    best_size = len(S)
                    best_S = list(S)
                    addable = compute_addable(S)
    
    return best_S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = float('inf')
    all_points = list(itertools.product(range(N), repeat=2))
    
    # First, try row-major greedy as baseline
    row_major_order = list(itertools.product(range(N), repeat=2))
    S1 = _build_greedy_order(row_major_order)
    best_set = S1
    best_size = len(S1)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        return best_set
    
    # Try minimum-degree greedy with different seeds, then local search
    attempts = 50
    for seed in range(attempts):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        S = _build_minimal_greedy(order_seed=seed)
        
        # Apply local search refinement
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, all_points, max_iterations=100, time_limit=min(remaining, 20))
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
        elif len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Try column-major order
    col_major_order = list(itertools.product(range(N), repeat=2))
    col_major_order.sort(key=lambda p: (p[1], p[0]))
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        S2 = _build_greedy_order(col_major_order)
        S2_refined = _local_search(S2, all_points, max_iterations=100, time_limit=15)
        if len(S2_refined) < best_size:
            best_size = len(S2_refined)
            best_set = S2_refined
    
    # Try diagonal orderings
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        diag_order = sorted(itertools.product(range(N), repeat=2), 
                          key=lambda p: (abs(p[0]-p[1]), p[0], p[1]))
        S3 = _build_greedy_order(diag_order)
        S3_refined = _local_search(S3, all_points, max_iterations=100, time_limit=15)
        if len(S3_refined) < best_size:
            best_size = len(S3_refined)
            best_set = S3_refined
    
    # Random orderings with local search
    rng_global = random.Random(42)
    for _ in range(10):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        random_order = list(itertools.product(range(N), repeat=2))
        rng_global.shuffle(random_order)
        S = _build_greedy_order(random_order)
        remaining = DEADLINE_SECONDS - 5 - (time.time() - start_time)
        if remaining > 5:
            S_refined = _local_search(S, all_points, max_iterations=50, time_limit=min(remaining, 10))
            if len(S_refined) < best_size:
                best_size = len(S_refined)
                best_set = S_refined
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
