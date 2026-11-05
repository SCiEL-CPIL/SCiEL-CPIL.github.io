"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N

    # Create coordinate arrays
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)

    # Calculate squared distances using int32 to save memory and speed up
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)

    # State:
    # S_set: set of indices in S
    # D_dict: {idx: set of squared distances from idx to other points in S}
    # For each idx in S, D_dict[idx] = {dist(idx, j) : j in S, j != idx}

    def is_valid_addition(idx_P, S_indices, D_dict):
        """Check if point idx_P can be added to set S_indices."""
        if not S_indices:
            return True
        
        dists_from_P = set()
        for idx_Q in S_indices:
            d_val = int(dist2[idx_P, idx_Q])
            # Condition A: distances from P to all Q in S must be distinct
            if d_val in dists_from_P:
                return False
            dists_from_P.add(d_val)
            # Condition B: d_val must not be in D_dict[idx_Q]
            if d_val in D_dict[idx_Q]:
                return False
        return True

    def add_point(idx_P, S_indices, D_dict, S_set, complement):
        S_indices.append(idx_P)
        S_set.add(idx_P)
        complement.remove(idx_P)
        
        dists_from_P = set()
        for idx_Q in S_indices:
            if idx_Q == idx_P:
                continue
            d = int(dist2[idx_P, idx_Q])
            dists_from_P.add(d)
            D_dict[idx_Q].add(d)
        D_dict[idx_P] = dists_from_P

    def remove_point(idx_P, S_indices, D_dict, S_set, complement):
        if idx_P in S_indices:
            S_indices.remove(idx_P)
        S_set.discard(idx_P)
        complement.append(idx_P)
        
        if idx_P in D_dict:
            for idx_Q in S_indices:
                d = int(dist2[idx_P, idx_Q])
                if d in D_dict[idx_Q]:
                    D_dict[idx_Q].discard(d)
            del D_dict[idx_P]

    # Phase 1: Greedy with multiple orderings
    rng = random.Random(123)
    best_size = 0
    best_solution = []
    best_D = {}
    best_S_set = set()
    best_complement = list(range(num_points))

    # Generate several orderings
    orderings = []
    
    # Random orderings
    for _ in range(8):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
    
    # Sorted by x
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0], points[i][1])))
    # Sorted by y
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][1], points[i][0])))
    # Sorted by x+y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] + points[i][1]))
    # Sorted by x-y
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0] - points[i][1]))
    # Sorted by distance from center
    cx, cy = N/2 - 0.5, N/2 - 0.5
    orderings.append(sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2))
    # Sorted by distance from corner
    orderings.append(sorted(range(num_points), key=lambda i: points[i][0]**2 + points[i][1]**2))

    for order in orderings:
        S_indices = []
        D_dict = {}
        S_set = set()
        complement = list(range(num_points))
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if is_valid_addition(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
        
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_solution = S_indices[:]
            best_D = {k: v.copy() for k, v in D_dict.items()}
            best_S_set = S_set.copy()
            best_complement = complement[:]

    # Phase 2: Local search with batch swaps and 2-opt improvements
    S_indices = best_solution[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    current_size = len(S_indices)
    best_size = current_size
    
    def try_add_random(S_indices, D_dict, S_set, complement, max_attempts=5):
        """Try to add random points from complement."""
        added = 0
        for _ in range(max_attempts):
            if not complement or time.time() > deadline - 15:
                break
            idx_P_add = rng.choice(complement)
            if is_valid_addition(idx_P_add, S_indices, D_dict):
                add_point(idx_P_add, S_indices, D_dict, S_set, complement)
                added += 1
        return added

    def try_batch_swap(S_indices, D_dict, S_set, complement, batch_size=2):
        """Try to remove batch_size points and add more points."""
        if len(S_indices) < batch_size or not complement:
            return False, 0
        
        # Select random points to remove
        to_remove = rng.sample(S_indices, min(batch_size, len(S_indices)))
        
        # Save state before modification
        orig_S_indices = S_indices[:]
        orig_D_dict = {k: v.copy() for k, v in D_dict.items()}
        orig_S_set = S_set.copy()
        orig_complement = complement[:]
        
        # Remove the selected points
        temp_S_indices = S_indices[:]
        temp_D_dict = {k: v.copy() for k, v in D_dict.items()}
        temp_S_set = S_set.copy()
        temp_complement = complement[:]
        
        for idx in to_remove:
            if idx in temp_S_indices:
                temp_S_indices.remove(idx)
            temp_S_set.discard(idx)
            temp_complement.append(idx)
            
            if idx in temp_D_dict:
                for idx_R in temp_S_indices:
                    d = int(dist2[idx, idx_R])
                    if d in temp_D_dict[idx_R]:
                        temp_D_dict[idx_R].discard(d)
                del temp_D_dict[idx]
        
        # Try to add as many points as possible from complement
        added_count = 0
        for idx_P in temp_complement[:]:
            if time.time() > deadline - 15:
                break
            if idx_P in temp_S_set:
                continue
            if is_valid_addition(idx_P, temp_S_indices, temp_D_dict):
                add_point(idx_P, temp_S_indices, temp_D_dict, temp_S_set, temp_complement)
                added_count += 1
        
        # If we added more than we removed, accept the change
        if added_count > len(to_remove):
            # Apply the changes
            S_indices.clear()
            S_indices.extend(temp_S_indices)
            D_dict.clear()
            D_dict.update({k: v for k, v in temp_D_dict.items()})
            S_set.clear()
            S_set.update(temp_S_set)
            complement.clear()
            complement.extend(temp_complement)
            return True, added_count - len(to_remove)
        else:
            # Roll back
            S_indices.clear()
            S_indices.extend(orig_S_indices)
            D_dict.clear()
            D_dict.update({k: v for k, v in orig_D_dict.items()})
            S_set.clear()
            S_set.update(orig_S_set)
            complement.clear()
            complement.extend(orig_complement)
            return False, 0

    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline - 20:
        improved = False
        
        # Strategy 1: Try single swaps
        for _ in range(15):
            if time.time() > deadline - 20 or not complement or not S_indices:
                break
                
            idx_P = rng.choice(complement)
            idx_Q = rng.choice(S_indices)
            
            # Check if swapping Q for P is valid
            temp_D = {k: v.copy() for k, v in D_dict.items()}
            if idx_Q in temp_D:
                for idx_R in S_indices:
                    if idx_R != idx_Q:
                        d = int(dist2[idx_Q, idx_R])
                        if d in temp_D[idx_R]:
                            temp_D[idx_R].discard(d)
                del temp_D[idx_Q]
            
            valid = True
            dists_from_P = set()
            for idx_R in S_indices:
                if idx_R == idx_Q:
                    continue
                d_val = int(dist2[idx_P, idx_R])
                if d_val in dists_from_P:
                    valid = False
                    break
                dists_from_P.add(d_val)
                if d_val in temp_D.get(idx_R, set()):
                    valid = False
                    break
            
            if valid:
                # Perform the swap
                if idx_Q in S_indices:
                    S_indices.remove(idx_Q)
                S_set.discard(idx_Q)
                complement.append(idx_Q)
                
                if idx_Q in D_dict:
                    for idx_R in S_indices:
                        d = int(dist2[idx_Q, idx_R])
                        if d in D_dict[idx_R]:
                            D_dict[idx_R].discard(d)
                    del D_dict[idx_Q]
                
                S_indices.append(idx_P)
                S_set.add(idx_P)
                complement.remove(idx_P)
                
                dists_from_P_new = set()
                for idx_R in S_indices:
                    if idx_R == idx_P:
                        continue
                    d = int(dist2[idx_P, idx_R])
                    dists_from_P_new.add(d)
                    D_dict[idx_R].add(d)
                D_dict[idx_P] = dists_from_P_new
                
                improved = True
                break
        
        # Strategy 2: Try batch swaps (remove 2-3 points, add more)
        if not improved and time.time() < deadline - 20:
            for batch_size in [2, 3]:
                success, gain = try_batch_swap(S_indices, D_dict, S_set, complement, batch_size)
                if success:
                    improved = True
                    break
        
        # Strategy 3: Try to add more points after any improvement
        if improved:
            try_add_random(S_indices, D_dict, S_set, complement, max_attempts=8)
            no_improve = 0
            
            if len(S_indices) > best_size:
                best_size = len(S_indices)
                best_solution = S_indices[:]
                best_D = {k: v.copy() for k, v in D_dict.items()}
                best_S_set = S_set.copy()
                best_complement = complement[:]
        else:
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Restart from best with perturbation
                S_indices = best_solution[:]
                D_dict = {k: v.copy() for k, v in best_D.items()}
                S_set = best_S_set.copy()
                complement = best_complement[:]
                no_improve = 0
                
                # Perturb: remove some random points to explore new region
                if S_indices:
                    num_remove = max(1, len(S_indices) // 8)
                    to_remove = rng.sample(S_indices, min(num_remove, len(S_indices)))
                    for idx in to_remove:
                        remove_point(idx, S_indices, D_dict, S_set, complement)
                
                # Try to fill back up
                try_add_random(S_indices, D_dict, S_set, complement, max_attempts=10)

    # Phase 3: Final polish - systematic scan for addable points
    S_indices = best_solution[:]
    D_dict = {k: v.copy() for k, v in best_D.items()}
    S_set = best_S_set.copy()
    complement = best_complement[:]
    
    # Try to add any remaining valid points in multiple passes
    for pass_num in range(3):
        changed = False
        for idx_P in complement[:]:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(idx_P, S_indices, D_dict):
                add_point(idx_P, S_indices, D_dict, S_set, complement)
                changed = True
        if not changed:
            break
    
    if len(S_indices) > len(best_solution):
        best_solution = S_indices[:]

    return [points[i] for i in best_solution]
# EVOLVE-BLOCK-END
