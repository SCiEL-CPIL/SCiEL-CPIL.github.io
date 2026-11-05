import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for fast addability check
# bad_pairs[p] is a set of (i, j) with i < j such that {p, i, j} is isosceles
BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    # Use numpy vectorization for speed
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    dij = dist2[i_idx, j_idx]
    
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        
        # Check for isosceles: dp_i == dp_j OR dp_i == dij OR dp_j == dij
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        
        for k in bad_indices:
            i = i_idx[k]
            j = j_idx[k]
            BAD_PAIRS[p].add((i, j))

# Run precomputation at module load
_start_time = time.time()
_precompute_bad_pairs()
_precompute_time = time.time() - _start_time

def _is_addable(p, S_set):
    """Check if p can be added to S_set (set of indices)."""
    if p in S_set:
        return False
    
    # Check 1: Distances from p to S_set must be distinct.
    dists_from_p = set()
    for q in S_set:
        d = dist2[p, q]
        if d in dists_from_p:
            return False
        dists_from_p.add(d)
        
    # Check 2: For each q in S_set, d(p, q) must not be equal to d(q, r) for any r in S_set, r != p.
    # This is equivalent to checking if adding p creates any isosceles triangle with existing pairs.
    # We can use BAD_PAIRS[p] to check if any pair (i, j) in BAD_PAIRS[p] is contained in S_set.
    # If so, then p, i, j form an isosceles triangle.
    for i, j in BAD_PAIRS[p]:
        if i in S_set and j in S_set:
            return False
            
    return True

def _get_addable(S_set):
    """Get set of points that can be added to S_set."""
    addable = set()
    for p in range(N_PTS):
        if p not in S_set:
            if _is_addable(p, S_set):
                addable.add(p)
    return addable

def _greedy_initialization(deadline):
    """Greedy construction of a maximal no-isosceles set."""
    S = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        # Find all addable points
        addable = _get_addable(S)
        if not addable:
            break
            
        # Choose the point that blocks the most other addable points
        best_p = -1
        best_block_count = -1
        
        # To speed up, we might not need to check all addable points if we find a good one quickly,
        # but for correctness of greedy, we check all.
        for p in addable:
            # Simulate adding p
            new_S = S | {p}
            new_addable = _get_addable(new_S)
            # Count how many points are no longer addable
            blocked = len(addable) - len(new_addable)
            if blocked > best_block_count:
                best_block_count = blocked
                best_p = p
                
        if best_p == -1:
            break
            
        S.add(best_p)
        
    return S

def _local_search(S_initial, deadline):
    """
    Local search to minimize |S|.
    Uses removal and swap moves.
    """
    S_set = set(S_initial)
    current_size = len(S_set)
    best_set = set(S_set)
    best_size = current_size
    
    temp = 10.0
    cooling = 0.995
    no_improve_steps = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if temp < 0.1:
            temp = 0.1
            
        # Try to remove a random point
        if not S_set:
            break
            
        p_out = random.choice(list(S_set))
        new_S_set = S_set - {p_out}
        
        # Find addable points for new_S_set
        addable = _get_addable(new_S_set)
        
        # If no addable points, we have a smaller maximal set
        if not addable:
            S_set = new_S_set
            current_size = len(S_set)
            if current_size < best_size:
                best_size = current_size
                best_set = set(S_set)
                no_improve_steps = 0
            continue
            
        # If there are addable points, we must add some back to maintain maximality
        # Choose the point that blocks the most other addable points
        best_p_in = -1
        best_block_count = -1
        
        # Limit the search to a few candidates to save time
        candidates = list(addable)
        random.shuffle(candidates)
        for p_in in candidates[:30]:
            new_S_set2 = new_S_set | {p_in}
            new_addable = _get_addable(new_S_set2)
            blocked = len(addable) - len(new_addable)
            if blocked > best_block_count:
                best_block_count = blocked
                best_p_in = p_in
                
        if best_p_in == -1:
            continue
            
        new_S_set = new_S_set | {best_p_in}
        new_size = len(new_S_set)
        
        delta_e = new_size - current_size
        if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
            S_set = new_S_set
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = set(S_set)
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Kick if stuck
        if no_improve_steps > max_no_improve:
            # Randomly swap two points
            if len(S_set) > 1:
                p1, p2 = random.sample(list(S_set), 2)
                S_set.remove(p1)
                S_set.remove(p2)
                # Add two random addable points
                addable = _get_addable(S_set)
                if len(addable) >= 2:
                    new_p1, new_p2 = random.sample(list(addable), 2)
                    S_set.add(new_p1)
                    S_set.add(new_p2)
                    current_size = len(S_set)
            no_improve_steps = 0
            temp = max(temp, 5.0)
            
        temp *= cooling
        
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 300.0
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline - 10.0:
        S_init = _greedy_initialization(deadline - 5.0)
        S_opt = _local_search(S_init, deadline - 2.0)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
    if best_set is None:
        best_set = _greedy_initialization(deadline)
        
    return [ALL_POINTS[i] for i in best_set]
