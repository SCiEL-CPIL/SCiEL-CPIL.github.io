import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290.0

    # Precompute squared distances
    coords = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
    dist2 = ((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2)
    
    # Helper functions
    def can_add(p, S, D):
        if p in S:
            return False
        if not S:
            return True
        
        # Check if distances from p to all q in S are distinct
        s_list = list(S)
        d_p_to_s = dist2[p, s_list]
        if len(np.unique(d_p_to_s)) != len(s_list):
            return False
        
        # Check if any distance d(p, q) is already in D[q]
        for q in s_list:
            d_val = dist2[p, q]
            if d_val in D[q]:
                return False
        return True

    def add_point(p, S, D):
        S.add(p)
        D[p] = set()
        for q in S:
            if q == p:
                continue
            d_val = dist2[p, q]
            D[p].add(d_val)
            D[q].add(d_val)

    def remove_point(p, S, D):
        S.remove(p)
        for q in S:
            D[q].discard(dist2[p, q])
        del D[p]

    def can_swap(u, v, S, D):
        if u not in S or v in S:
            return False
        
        s_list = [q for q in S if q != u]
        if not s_list:
            return True # Adding v to empty set is always valid
        
        # Check if distances from v to s_list are distinct
        d_v_to_s = dist2[v, s_list]
        if len(np.unique(d_v_to_s)) != len(s_list):
            return False
        
        # Check conflicts
        for q in s_list:
            d_val = dist2[v, q]
            # d_val must not be in D[q], unless it is the distance to u (which is being removed)
            if d_val in D[q] and d_val != dist2[u, q]:
                return False
        return True

    def swap_points(u, v, S, D):
        remove_point(u, S, D)
        add_point(v, S, D)

    def can_swap2(u1, u2, v1, v2, S, D):
        if u1 not in S or u2 not in S or v1 in S or v2 in S:
            return False
        if u1 == u2 or v1 == v2:
            return False
        
        s_list = [q for q in S if q != u1 and q != u2]
        
        # Check distinctness for v1 and v2 against s_list and each other
        # Distances from v1 to s_list
        if s_list:
            d_v1_to_s = dist2[v1, s_list]
            if len(np.unique(d_v1_to_s)) != len(s_list):
                return False
            # Distances from v2 to s_list
            d_v2_to_s = dist2[v2, s_list]
            if len(np.unique(d_v2_to_s)) != len(s_list):
                return False
        
        # Distance between v1 and v2
        d_v1_v2 = dist2[v1, v2]
        
        # Check conflicts for v1
        for q in s_list:
            d_val = dist2[v1, q]
            # d_val must not be in D[q] \ {dist2[u1, q], dist2[u2, q]}
            if d_val in D[q]:
                if d_val != dist2[u1, q] and d_val != dist2[u2, q]:
                    return False
            # Also check if d_val == d_v1_v2? No, d_v1_v2 is distance to v2, which is not in s_list.
            # But we need to ensure d_v1_v2 is not in D[v1] (which is empty initially)
            # And d_v1_v2 is not in D[v2] (empty initially)
            # The check above handles q in s_list.
        
        # Check conflicts for v2
        for q in s_list:
            d_val = dist2[v2, q]
            if d_val in D[q]:
                if d_val != dist2[u1, q] and d_val != dist2[u2, q]:
                    return False
        
        # Check if d_v1_v2 conflicts with existing distances in D for v1 or v2?
        # v1 and v2 are new, so D[v1] and D[v2] are empty.
        # But we need to ensure d_v1_v2 is not equal to any d(v1, q) or d(v2, q) for q in s_list.
        # This is covered by the "distinctness" check?
        # No, distinctness check was d_v1_to_s vs d_v1_to_s.
        # We need d_v1_v2 != d(v1, q) for all q in s_list.
        if d_v1_v2 in set(dist2[v1, s_list]):
            return False
        if d_v1_v2 in set(dist2[v2, s_list]):
            return False
            
        return True

    def swap_points2(u1, u2, v1, v2, S, D):
        remove_point(u1, S, D)
        remove_point(u2, S, D)
        add_point(v1, S, D)
        add_point(v2, S, D)

    best_set = set()
    best_D = {}
    rng = random.Random(42)

    # Main search loop
    while time.time() < deadline - 5:
        S = set()
        D = {}
        
        # Greedy Init
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        for p in order:
            if can_add(p, S, D):
                add_point(p, S, D)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add
            unselected = [p for p in range(NUM_POINTS) if p not in S]
            rng.shuffle(unselected)
            for p in unselected:
                if can_add(p, S, D):
                    add_point(p, S, D)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try 1-swaps
            s_list = list(S)
            rng.shuffle(s_list)
            swap_found = False
            for u in s_list:
                for v in unselected: # Reuse unselected list, it's fine
                    if can_swap(u, v, S, D):
                        swap_points(u, v, S, D)
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if improved:
                continue

            # 3. Try 2-swaps (Random)
            if len(S) >= 2:
                # Pick 2 to remove
                u1, u2 = random.sample(list(S), 2)
                # Pick 2 to add
                if len(unselected) >= 2:
                    v1, v2 = random.sample(unselected, 2)
                    if can_swap2(u1, u2, v1, v2, S, D):
                        swap_points2(u1, u2, v1, v2, S, D)
                        improved = True
            
        if len(S) > len(best_set):
            best_set = S
            best_D = D
            
    return [(i // N, i % N) for i in best_set]
