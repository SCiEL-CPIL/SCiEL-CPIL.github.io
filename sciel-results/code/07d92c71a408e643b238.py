import random
import time
import itertools
import numpy as np

N = 16
N_PTS = N * N
ALL_POINTS = list(itertools.product(range(N), repeat=2))

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute ISO_PAIR[p][u] = bitmask of q such that {p, q, u} is isosceles
# Note: {p, q, u} is isosceles if d(p,q)==d(p,u) or d(p,q)==d(q,u) or d(p,u)==d(q,u)
# We fix p and u. We iterate q.
ISO_PAIR = [[0] * N_PTS for _ in range(N_PTS)]

def _precompute_iso_pairs():
    for p in range(N_PTS):
        for u in range(N_PTS):
            if p == u:
                continue
            mask = 0
            dp_u = dist2[p, u]
            for q in range(N_PTS):
                if q == p or q == u:
                    continue
                dp_q = dist2[p, q]
                du_q = dist2[u, q]
                # Check if isosceles
                if dp_q == dp_u or dp_q == du_q or dp_u == du_q:
                    mask |= (1 << q)
            ISO_PAIR[p][u] = mask

_start_time = time.time()
_precompute_iso_pairs()
# print("Precomputation done")

def _greedy_bitset(S_init_mask=0, C_init_mask=None, deadline=None):
    """
    Greedy construction using bitmasks.
    S_init_mask: bitmask of points already in S.
    C_init_mask: bitmask of valid candidates. If None, computes valid candidates for S_init_mask.
    Returns (S_mask, C_mask) where C_mask is 0 (maximal set reached).
    """
    S_mask = S_init_mask
    
    # Compute initial C if not provided
    if C_init_mask is None:
        C_mask = (1 << N_PTS) - 1
        # Remove points in S
        C_mask &= ~S_mask
        # Remove points that form isosceles with any pair in S
        # For each q in C, check if there exist u, v in S such that {q, u, v} is isosceles.
        # This is equivalent to: for each pair (u, v) in S, remove all q such that {q, u, v} is isosceles.
        # We can precompute BAD_FOR_PAIR[u][v] = mask of q such that {q, u, v} is isosceles.
        # Or just iterate u, v in S and use ISO_PAIR?
        # ISO_PAIR[u][v] is mask of q such that {u, q, v} is isosceles.
        # So for each u, v in S, C_mask &= ~ISO_PAIR[u][v].
        
        # Extract points in S
        s_points = []
        temp_s = S_mask
        while temp_s:
            lsb = temp_s & -temp_s
            idx = lsb.bit_length() - 1
            s_points.append(idx)
            temp_s ^= lsb
            
        for i in range(len(s_points)):
            for j in range(i+1, len(s_points)):
                u = s_points[i]
                v = s_points[j]
                C_mask &= ~ISO_PAIR[u][v]
    else:
        C_mask = C_init_mask

    # Greedy loop
    while C_mask:
        if deadline and time.time() > deadline:
            break
            
        best_p = -1
        best_c_size = float('inf')
        best_new_c = 0
        
        # Iterate through candidates in C_mask
        c_temp = C_mask
        while c_temp:
            lsb = c_temp & -c_temp
            p = lsb.bit_length() - 1
            c_temp ^= lsb
            
            # Calculate F_p: points q blocked by adding p to S
            # F_p = union of ISO_PAIR[p][u] for all u in S
            f_p = 0
            s_temp = S_mask
            while s_temp:
                lsb_u = s_temp & -s_temp
                u = lsb_u.bit_length() - 1
                s_temp ^= lsb_u
                f_p |= ISO_PAIR[p][u]
            
            # New C = C \ {p} \ F_p
            new_c = C_mask & ~(1 << p) & ~f_p
            c_size = new_c.bit_count()
            
            if c_size < best_c_size:
                best_c_size = c_size
                best_p = p
                best_new_c = new_c
        
        if best_p == -1:
            break
            
        S_mask |= (1 << best_p)
        C_mask = best_new_c
        
    return S_mask

def _mask_to_list(mask):
    pts = []
    temp = mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        pts.append(idx)
        temp ^= lsb
    return pts

def _is_maximal_mask(S_mask):
    """Check if S_mask is maximal. Returns True if no point can be added."""
    # A set is maximal if for every x not in S, S U {x} is invalid.
    # S U {x} is invalid if there exist u, v in S such that {x, u, v} is isosceles.
    # So for each x not in S, we check if ISO_PAIR[u][v] contains x for any u, v in S?
    # No, {x, u, v} is isosceles means x is in ISO_PAIR[u][v].
    # So x is blocked if x is in (OR of ISO_PAIR[u][v] for all u, v in S).
    
    s_points = _mask_to_list(S_mask)
    if len(s_points) < 2:
        # If S has 0 or 1 points, it's not maximal unless N_PTS is small
        # With 0 points, any x can be added. Not maximal.
        # With 1 point u, x can be added unless {x, u, v} is iso for some v in S? No, S only has u.
        # So {x, u, v} requires v in S. If |S|<2, no such v exists.
        # So any x can be added. Not maximal.
        return False

    # Precompute the union of all ISO_PAIR[u][v] for u, v in S
    blocked_mask = 0
    for i in range(len(s_points)):
        for j in range(i+1, len(s_points)):
            u = s_points[i]
            v = s_points[j]
            blocked_mask |= ISO_PAIR[u][v]
            
    # If there is any x not in S and not in blocked_mask, S is not maximal
    all_mask = (1 << N_PTS) - 1
    available = all_mask & ~S_mask & ~blocked_mask
    return available == 0

def _local_search_bitset(S_mask, deadline, max_iter=100):
    """
    Local search to minimize size of maximal set.
    Tries to remove points and re-greedy.
    """
    current_S = S_mask
    current_size = current_S.bit_count()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        s_points = _mask_to_list(current_S)
        random.shuffle(s_points)
        
        # Try removing 1 point
        for p in s_points:
            if time.time() > deadline:
                break
            S_temp = current_S & ~(1 << p)
            # Re-greedy from S_temp
            # We need to calculate C for S_temp
            # C = {q | q not in S_temp and S_temp U {q} is valid}
            # Validity: no u, v in S_temp such that {q, u, v} is iso.
            # So q is blocked if q in ISO_PAIR[u][v] for any u, v in S_temp.
            
            # Calculate blocked mask for S_temp
            s_temp_pts = _mask_to_list(S_temp)
            blocked = 0
            for i in range(len(s_temp_pts)):
                for j in range(i+1, len(s_temp_pts)):
                    u = s_temp_pts[i]
                    v = s_temp_pts[j]
                    blocked |= ISO_PAIR[u][v]
            
            C_temp = ((1 << N_PTS) - 1) & ~S_temp & ~blocked
            
            # Run greedy to completion
            S_new = _greedy_bitset(S_temp, C_temp, deadline)
            
            new_size = S_new.bit_count()
            if new_size < current_size:
                current_S = S_new
                current_size = new_size
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing 2 points (if 1 didn't work)
        if len(s_points) >= 2:
            for i in range(min(len(s_points), 10)):
                if time.time() > deadline:
                    break
                p1 = s_points[i]
                for j in range(i+1, min(len(s_points), i+10)):
                    p2 = s_points[j]
                    S_temp = current_S & ~(1 << p1) & ~(1 << p2)
                    
                    s_temp_pts = _mask_to_list(S_temp)
                    blocked = 0
                    for k in range(len(s_temp_pts)):
                        for l in range(k+1, len(s_temp_pts)):
                            u = s_temp_pts[k]
                            v = s_temp_pts[l]
                            blocked |= ISO_PAIR[u][v]
                    
                    C_temp = ((1 << N_PTS) - 1) & ~S_temp & ~blocked
                    S_new = _greedy_bitset(S_temp, C_temp, deadline)
                    
                    new_size = S_new.bit_count()
                    if new_size < current_size:
                        current_S = S_new
                        current_size = new_size
                        improved = True
                        break
                if improved:
                    break
                    
        if not improved:
            break
            
    return current_S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_S_mask = 0
    best_size = float('inf')
    
    # Multiple restarts
    seed = 42
    while time.time() < deadline - 20.0:
        random.seed(seed)
        
        # Randomize the greedy choice slightly by shuffling candidates?
        # The greedy algorithm picks the one with min C_size.
        # Ties are broken by the order of iteration (LSB first).
        # To diversify, we can randomize the initial S or the tie-breaking.
        # Let's just run the standard greedy first.
        
        S_mask = _greedy_bitset(0, None, deadline - 10.0)
        
        # Polish
        S_mask = _local_search_bitset(S_mask, deadline - 5.0, max_iter=50)
        
        size = S_mask.bit_count()
        if size < best_size:
            best_size = size
            best_S_mask = S_mask
            
        # Try a different strategy: Randomized Greedy
        # Pick a random point from C, not the best one.
        # This might lead to different local minima.
        if seed % 2 == 0:
             # Standard greedy
             pass
        else:
             # Randomized greedy
             S_mask = 0
             C_mask = (1 << N_PTS) - 1
             while C_mask:
                 if time.time() > deadline - 10.0:
                     break
                 # Pick random point from C
                 c_list = _mask_to_list(C_mask)
                 p = random.choice(c_list)
                 
                 # Update S and C
                 S_mask |= (1 << p)
                 
                 # Calculate new C
                 # F_p = union of ISO_PAIR[p][u] for u in S
                 s_pts = _mask_to_list(S_mask)
                 f_p = 0
                 for u in s_pts:
                     f_p |= ISO_PAIR[p][u]
                     
                 C_mask = C_mask & ~(1 << p) & ~f_p
                 
             # Polish
             S_mask = _local_search_bitset(S_mask, deadline - 5.0, max_iter=50)
             size = S_mask.bit_count()
             if size < best_size:
                 best_size = size
                 best_S_mask = S_mask
                 
        seed += 1
        
        # If we found a very small set, we might stop early?
        # Theoretical lower bound is unknown, but 1.818 score implies ~14 points?
        # Seed score 1.0. 1.818 is improvement.
        # If best_size < 10, we are doing great.
        
    if best_S_mask == 0:
        # Fallback
        best_S_mask = _greedy_bitset(0, None, deadline - 5.0)
        
    return [ALL_POINTS[i] for i in _mask_to_list(best_S_mask)]
