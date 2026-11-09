import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 140

    # Fallback: Paley(67)
    def make_paley_prime_matrix(p):
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        edge_matrix = [[False] * p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                edge_matrix[i][j] = diff in squares
        return edge_matrix

    def build_orientation_from_matrix(edge_matrix, n):
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if edge_matrix[i][j]:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Try to find a smaller tournament using local search
    # We try n from 66 down to 50
    # If we find a valid one, return it.
    
    def check_and_optimize(n, max_time=5.0):
        """
        Try to find a valid tournament of size n using local search.
        Returns orientation list if found, else None.
        """
        if n < 31:
            return None
            
        # Initialize random tournament
        # edge_matrix[i][j] = 1 if i -> j
        edge_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    edge_matrix[i][j] = True
                    edge_matrix[j][i] = False
                else:
                    edge_matrix[i][j] = False
                    edge_matrix[j][i] = True
        
        # Precompute out masks
        out_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and edge_matrix[i][j]:
                    mask |= (1 << j)
            out_masks[i] = mask
            
        def count_bad_sets():
            bad_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        for l in range(k+1, n):
                            # Check if any of i,j,k,l dominates the other 3
                            # i dominates: i->j, i->k, i->l
                            if (out_masks[i] & (1<<j) & (1<<k) & (1<<l)) or \
                               (out_masks[j] & (1<<i) & (1<<k) & (1<<l)) or \
                               (out_masks[k] & (1<<i) & (1<<j) & (1<<l)) or \
                               (out_masks[l] & (1<<i) & (1<<j) & (1<<k)):
                                continue
                            bad_count += 1
            return bad_count

        current_bad = count_bad_sets()
        if current_bad == 0:
            return build_orientation_from_matrix(edge_matrix, n)

        # Local search
        # We iterate over edges, try flipping, see if bad count decreases
        # To be efficient, we only check affected 4-sets
        # But for simplicity in this short time budget, we might just re-count?
        # No, re-counting is O(n^4) ~ 20M ops. Too slow for many iterations.
        # We need incremental update.
        
        # Incremental update logic:
        # Flip (u, v).
        # Affected 4-sets: {u, v, a, b} for a,b in V \ {u,v}
        # For each such set, check if it was bad, and if it is now bad.
        
        def is_bad_set(u, v, a, b):
            # Check if any of u,v,a,b dominates the other 3
            # u dominates: u->v, u->a, u->b
            if (out_masks[u] & (1<<v) & (1<<a) & (1<<b)) or \
               (out_masks[v] & (1<<u) & (1<<a) & (1<<b)) or \
               (out_masks[a] & (1<<u) & (1<<v) & (1<<b)) or \
               (out_masks[b] & (1<<u) & (1<<v) & (1<<a)):
                return False
            return True

        # We'll do a simple hill climbing.
        # Randomly pick an edge, flip it, calculate delta.
        # If delta < 0, accept.
        # If delta == 0, accept with prob 0.5.
        # If delta > 0, reject (or accept with low prob for SA).
        
        # To speed up, we can precompute list of other vertices for each pair?
        # No, just iterate.
        
        # Optimization: If n is large, the number of affected sets is n-2.
        # Checking each takes O(1) bit ops.
        # So one flip evaluation is O(n).
        # Total time for T steps is O(T*n).
        # With n=60, T=10000, 600k ops. Very fast.
        
        time_limit = max_time
        start_ls = time.time()
        
        # Simple random restarts if stuck?
        # For now, just run one long search.
        
        while time.time() - start_ls < time_limit:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Check current bad status of affected sets
            # And simulate flip
            # We need to know the new state of out_masks for u and v
            # new_out_u: if u->v was True, v is removed. If False, v is added.
            # Similarly for v.
            
            old_u_v = edge_matrix[u][v]
            
            # Calculate delta
            delta = 0
            for a in range(n):
                if a == u or a == v:
                    continue
                for b in range(a+1, n):
                    if b == u or b == v:
                        continue
                    # Set is {u, v, a, b}
                    # Check if it was bad
                    was_bad = is_bad_set(u, v, a, b)
                    
                    # Simulate flip
                    # If old_u_v is True, u->v becomes False, v->u becomes True.
                    # out_masks[u] loses v, out_masks[v] gains u.
                    # If old_u_v is False, u->v becomes True, v->u becomes False.
                    # out_masks[u] gains v, out_masks[v] loses u.
                    
                    # We don't want to modify global out_masks yet.
                    # Let's just check the new condition directly.
                    
                    # New condition for {u, v, a, b}:
                    # u dominates?
                    # u->v is new_val. u->a, u->b unchanged.
                    new_u_v = not old_u_v
                    new_v_u = old_u_v # since v->u is opposite of u->v
                    
                    # u dominates: new_u_v AND (u->a) AND (u->b)
                    u_dom = new_u_v and bool(out_masks[u] & (1<<a)) and bool(out_masks[u] & (1<<b))
                    # v dominates: new_v_u AND (v->a) AND (v->b)
                    v_dom = new_v_u and bool(out_masks[v] & (1<<a)) and bool(out_masks[v] & (1<<b))
                    # a dominates: (a->u) AND (a->v) AND (a->b)
                    # a->u is opposite of u->a.
                    # a->v is opposite of v->a.
                    a_dom = (not bool(out_masks[u] & (1<<a))) and (not bool(out_masks[v] & (1<<a))) and bool(out_masks[a] & (1<<b))
                    # b dominates: (b->u) AND (b->v) AND (b->a)
                    b_dom = (not bool(out_masks[u] & (1<<b))) and (not bool(out_masks[v] & (1<<b))) and bool(out_masks[b] & (1<<a))
                    
                    is_bad_now = not (u_dom or v_dom or a_dom or b_dom)
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                        # Early exit if delta > 0? No, we need total delta.
                        # But if we just want to improve, and delta becomes positive, 
                        # we might still want to know if it goes back down?
                        # No, if we reject positive delta, we can stop calculating if delta > 0?
                        # Actually, if we only accept delta <= 0, we can stop if delta > 0 
                        # and assume we will reject? 
                        # But we need to know if it's exactly 0 or >0.
                        # For simplicity, calculate all.
                        
            if delta < 0 or (delta == 0 and random.random() < 0.5):
                # Accept
                edge_matrix[u][v] = not old_u_v
                edge_matrix[v][u] = old_u_v
                # Update out_masks
                if old_u_v:
                    # u->v was True, now False. u loses v. v gains u.
                    out_masks[u] &= ~(1 << v)
                    out_masks[v] |= (1 << u)
                else:
                    # u->v was False, now True. u gains v. v loses u.
                    out_masks[u] |= (1 << v)
                    out_masks[v] &= ~(1 << u)
                
                current_bad += delta
                if current_bad == 0:
                    return build_orientation_from_matrix(edge_matrix, n)
            
        return None

    # Try to find smaller n
    # Start from 66 down to 50
    # Allocate time based on remaining budget
    # We have ~140s.
    # Let's try a few n values.
    
    # Note: The probability of finding a valid tournament by random search is very low.
    # However, if there is a valid tournament, local search might find it if the landscape is smooth.
    # Given the density of constraints, it might be rugged.
    # But let's try.
    
    # If we don't find anything, we return Paley(67).
    
    # To save time, we only try n=66, 65, 64 if we have time.
    # Each run takes ~5s.
    
    # Let's try n=66 first.
    if time.time() - start_time < deadline - 10:
        res = check_and_optimize(66, max_time=10.0)
        if res:
            return {'n': 66, 'orientation': res}
            
    if time.time() - start_time < deadline - 10:
        res = check_and_optimize(65, max_time=10.0)
        if res:
            return {'n': 65, 'orientation': res}
            
    if time.time() - start_time < deadline - 10:
        res = check_and_optimize(64, max_time=10.0)
        if res:
            return {'n': 64, 'orientation': res}

    # Fallback to Paley(67)
    p = 67
    edge_matrix = make_paley_prime_matrix(p)
    orientation = build_orientation_from_matrix(edge_matrix, p)
    return {'n': p, 'orientation': orientation}
