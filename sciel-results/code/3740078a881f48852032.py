# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using Local Search / Simulated Annealing.
    
    Strategy:
    1. Represent the problem as maximizing the size of an excluded set E.
    2. A point x is "bad" for E if it is covered by E in ALL 50 directions.
    3. We maintain 'crit[x]' = number of directions where x is covered by E.
    4. A move (add/remove point z to/from E) is valid if it doesn't create any new "bad" points.
    5. We use simulated annealing to escape local optima.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Deadline guard
    deadline = time.time() + 570.0

    # --- Field Arithmetic Setup ---
    # F_49 = F_7[alpha], alpha^2 = 3
    k_const = 3
    
    # Map (a, b) -> index a*p + b
    # idx_to_elem[i] = (a, b)
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    
    # Precompute addition and multiplication tables for F_q
    # add[i, j] = i + j
    # mul[i, j] = i * j
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            # Addition
            aa_add = (ai + aj) % p
            bb_add = (bi + bj) % p
            add_table[i, j] = aa_add * p + bb_add
            
            # Multiplication: (ai + bi*alpha)(aj + bj*alpha) = (ai*aj + bi*bj*k) + (ai*bj + bi*aj)*alpha
            aa_mul = (ai * aj + bi * bj * k_const) % p
            bb_mul = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa_mul * p + bb_mul

    # --- Directions ---
    # v = (0, 1) and v = (1, m) for m in F_q
    # There are 1 + q = 50 directions.
    num_dirs = q + 1
    
    # Precompute the list of other points on the line for each point and direction.
    # line_other[z, v_idx] = array of q-1 point indices.
    # Note: We only need to know the counts, so we can precompute the structure.
    
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    
    t_vals = np.arange(1, q, dtype=np.int32) # t != 0
    
    line_other = np.zeros((total_points, num_dirs, q - 1), dtype=np.int32)
    
    # Direction 0: v = (0, 1)
    # dx=0, dy=1
    # t*dx = 0, t*dy = t
    # x' = zx, y' = zy + t
    tdx0 = np.zeros(q-1, dtype=np.int32)
    tdy0 = t_vals # since 1*t = t
    nx0 = add_table[zx_arr[:, None], tdx0[None, :]]
    ny0 = add_table[zy_arr[:, None], tdy0[None, :]]
    line_other[:, 0, :] = nx0 * q + ny0
    
    # Directions 1..49: v = (1, m)
    for m_idx in range(1, num_dirs):
        m = m_idx - 1 # m is the field element index
        dx = 1 # field element 1
        dy = m
        
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        line_other[:, m_idx, :] = nx * q + ny

    # --- Local Search State ---
    # in_E: boolean array
    # crit: int array, crit[x] = number of directions v such that line(x, v) \ {x} is fully in E.
    # If crit[x] == num_dirs, x is "bad" (violates Nikodym condition).
    # We want to maximize |E| subject to no "bad" points.
    
    best_E = np.zeros(total_points, dtype=bool)
    best_size = 0
    best_crit = np.zeros(total_points, dtype=np.int32)
    
    def is_valid_state(in_E, crit):
        return np.all(crit < num_dirs)
    
    def apply_move(z, add_to_E, in_E, crit):
        """
        Tries to add or remove z from E.
        Returns True if the move is valid (no bad points created), False otherwise.
        Modifies in_E and crit in-place if valid.
        """
        if add_to_E:
            if in_E[z]:
                return False
            # Adding z to E increases crit[x] for all x in line_other[z, v] for all v.
            # Specifically, for each v, the points on line(z, v) excluding z get their crit incremented.
            # Wait, the definition of crit[x] is: number of directions v where ALL other points on line(x, v) are in E.
            # This is a global property.
            
            # Let's re-evaluate the increment logic.
            # If we add z to E:
            # 1. For any point x such that z is on line(x, v), the set of "other points" for x in direction v gains z.
            #    Previously, if all OTHER points were in E, then x was "covered" in v.
            #    Now, since z is in E, x is STILL covered in v (actually, it's easier to be covered).
            #    Wait. "Covered" means: line(x, v) \ {x} is a subset of E.
            #    If we add z to E, the condition "subset of E" becomes EASIER to satisfy.
            #    So crit[x] can only INCREASE or stay same.
            
            # Which x are affected?
            # x is affected if z is in line(x, v) \ {x}.
            # This is equivalent to x being in line(z, v) \ {z}.
            # So for each v, we look at pts = line_other[z, v].
            # For each x in pts, we need to check if line(x, v) \ {x} is now fully in E.
            # This is expensive to check naively.
            
            # Alternative: Maintain a counter 'cnt_in_E_on_line[x, v]' = number of points in E on line(x, v) excluding x.
            # Then crit[x] = number of v where cnt_in_E_on_line[x, v] == q-1.
            
            # Let's switch to maintaining cnt_in_E_on_line.
            # This makes updates O(1) per affected (x, v) pair.
            # Affected pairs for adding z: (x, v) where x in line(z, v).
            # There are num_dirs * (q-1) such pairs. ~ 50 * 48 = 2400.
            # This is very fast.
            
            pass # Logic moved to main loop for clarity with cnt array
            
        else:
            if not in_E[z]:
                return False
            pass

    # We need a state that tracks cnt_in_E_on_line[x, v]
    # cnt[x, v] = number of points in E on line(x, v) excluding x.
    # Bad if cnt[x, v] == q-1 for ALL v.
    # i.e., min_v (q-1 - cnt[x, v]) == 0.
    # Let's track 'missing[x, v] = (q-1) - cnt[x, v]'.
    # x is bad if min_v missing[x, v] == 0.
    
    # State:
    # in_E (bool)
    # missing (int array, shape total_points x num_dirs)
    
    best_missing = np.full((total_points, num_dirs), q-1, dtype=np.int32) # Initially E is empty, so 0 points in E, missing = q-1
    
    def try_flip(z, in_E, missing):
        """
        Flips z in E. Returns True if valid, False otherwise.
        """
        is_adding = not in_E[z]
        
        # Collect all (x, v) pairs affected.
        # If we add z to E, cnt[x, v] increases by 1 for all x in line(z, v).
        # So missing[x, v] decreases by 1.
        # If we remove z from E, cnt[x, v] decreases by 1.
        # So missing[x, v] increases by 1.
        
        # Affected x's are in line_other[z, :, :]
        # We need to update missing[x, v] for these pairs.
        
        # To check validity, we need to ensure that no x becomes "bad" (min missing == 0) 
        # if we are adding (missing decreases).
        # If we are removing (missing increases), validity is always preserved 
        # (since we are moving away from 0).
        
        if is_adding:
            # Check if any affected x has min(missing[x, :]) == 1 for the specific v we are about to decrement?
            # No, we decrement missing[x, v].
            # If missing[x, v] was 1, it becomes 0.
            # x becomes bad ONLY IF missing[x, v'] == 0 for ALL OTHER v' as well.
            # i.e., min(missing[x, :]) == 0 BEFORE update? No.
            # x is bad AFTER update if min(missing_new[x, :]) == 0.
            # missing_new[x, v] = missing[x, v] - 1.
            # missing_new[x, v'] = missing[x, v'] for v' != v.
            
            # So x becomes bad if:
            # 1. missing[x, v] == 1 (so it becomes 0)
            # 2. AND min(missing[x, v'] for v' != v) == 0.
            
            # Let's just apply the changes, check if any x has min==0, and revert if so.
            # Since the number of affected x's is small (unique points in line_other[z]), this is fast.
            
            # Identify unique affected points
            affected_pts = np.unique(line_other[z, :, :].flatten())
            
            # Store old values for reversion
            # We are modifying missing[x, v] for x in affected_pts, v in 0..num_dirs-1
            # But we only modify the specific v where x is in line(z, v).
            # Actually, for a fixed z and v, the points are line_other[z, v].
            # So for each v, we update missing[line_other[z, v], v] -= 1.
            
            revert_indices = []
            
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                # Update
                old_vals = missing[pts, v].copy()
                missing[pts, v] -= 1
                revert_indices.append((pts, v, old_vals))
                
            in_E[z] = True
            
            # Check validity
            # We only need to check affected_pts.
            # If any x in affected_pts has min(missing[x, :]) == 0, it's invalid.
            min_counts = np.min(missing[affected_pts, :], axis=1)
            
            if np.any(min_counts == 0):
                # Revert
                for pts, v, old_vals in revert_indices:
                    missing[pts, v] = old_vals
                in_E[z] = False
                return False
                
            return True
        else:
            # Removing z from E
            # missing[x, v] += 1
            # This can only make things "better" (increase missing).
            # So it's always valid.
            
            for v in range(num_dirs):
                pts = line_other[z, v, :]
                missing[pts, v] += 1
                
            in_E[z] = False
            return True

    # --- Simulated Annealing Loop ---
    
    # Initialize with a random set E
    # A good starting point might be a random subset of size ~ 1000?
    # Or start empty and grow?
    # Let's start with a random permutation and greedily add until stuck, then anneal.
    
    num_restarts = 20
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        in_E_curr = np.zeros(total_points, dtype=bool)
        missing_curr = np.full((total_points, num_dirs), q-1, dtype=np.int32)
        
        # Greedy initialization: try to add points in random order
        order = list(range(total_points))
        rng.shuffle(order)
        
        size = 0
        for z in order:
            if try_flip(z, in_E_curr, missing_curr):
                size += 1
                
        # Annealing phase
        # We want to maximize size.
        # Moves:
        # 1. Add a random point z not in E. (Increases size)
        # 2. Remove a random point z in E. (Decreases size, allows adding others later)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        current_size = size
        best_local_size = size
        best_local_E = in_E_curr.copy()
        best_local_missing = missing_curr.copy()
        
        # Time budget per restart
        restart_deadline = time.time() + 20.0 
        
        while time.time() < restart_deadline and T > min_T:
            # Pick a random move
            # 80% chance to try adding, 20% chance to try removing
            if rng.random() < 0.8:
                # Try to add a random point
                # Pick a random index
                z = rng.randint(0, total_points - 1)
                if in_E_curr[z]:
                    continue
                if try_flip(z, in_E_curr, missing_curr):
                    current_size += 1
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_E = in_E_curr.copy()
                        best_local_missing = missing_curr.copy()
                else:
                    # Move rejected, no state change
                    pass
            else:
                # Try to remove a random point
                # Pick a random index that is in E
                # We can pick a random index and check if it's in E, or maintain a list.
                # For simplicity, pick random z. If not in E, skip.
                z = rng.randint(0, total_points - 1)
                if not in_E_curr[z]:
                    continue
                try_flip(z, in_E_curr, missing_curr) # Always valid
                current_size -= 1
                
            T *= cooling
            
        # End of restart
        if best_local_size > best_size:
            best_size = best_local_size
            best_E = best_local_E.copy()
            best_missing = best_local_missing.copy()

    # Build N: points NOT in E
    points_list = []
    for pidx in range(total_points):
        if not best_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        # Fallback to all points if something went wrong
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)

# EVOLVE-BLOCK-END
