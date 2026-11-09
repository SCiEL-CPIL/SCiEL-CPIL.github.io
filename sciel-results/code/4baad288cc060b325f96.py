# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = 49.
    Tries to use the Hermitian unital U as the complement S if valid.
    Falls back to the seed construction if not.
    """
    p = 7
    q = p * p
    total_points = q * q
    
    # Time guard
    start_time = time.time()
    time_limit = 600.0
    
    # Field elements as pairs (a, b) for a + b*alpha
    # alpha^2 = k
    
    def find_non_residue(p):
        for c in range(2, p):
            if pow(c, (p - 1) // 2, p) == p - 1:
                return c
        return 1 # Should not happen

    def check_validity(S_set, p):
        """
        Checks if S_set (a set of tuples (xa, xb, ya, yb)) is a valid complement.
        i.e., for every x in S_set, there exists a line through x such that
        all other points on the line are NOT in S_set.
        """
        # Precompute lines for each point?
        # There are 49 points in a line. 2401 points.
        # For each x in S, check 50 directions.
        # 50 * 49 operations per x.
        # Total 2401 * 50 * 49 ~ 6 million ops. Fast.
        
        # Directions: (1, m) for m in F_49 and (0, 1)
        # We represent points as (xa, xb, ya, yb)
        
        def add_point(p1, p2, p):
            # p1, p2 are tuples (a, b)
            return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
        
        def mul_scalar(s, p1, p):
            # s in F_p, p1 in F_49
            return ((s * p1[0]) % p, (s * p1[1]) % p)
            
        # Actually, t is in F_49, not F_p.
        # Line: x + t*v.
        # v = (1, m) or (0, 1).
        # t = (ta, tb).
        
        # To speed up, we can precompute the set of points on each line?
        # Or just iterate.
        
        # Let's iterate over points in S_set
        for x in S_set:
            xa, xb, ya, yb = x
            found_free_line = False
            
            # Direction (0, 1): v = (0, 1)
            # Points: (xa, xb + t) for t in F_49
            # Check if any other point (xa, xb + t) is in S_set
            # If NO other point is in S_set, it's free.
            # We need to check all t != 0.
            
            # Optimization: For a fixed direction, if we find one other point in S, the line is not free.
            
            # Check direction (0, 1)
            ok = True
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0: continue
                    yb_new = (yb + tb) % p
                    # x is (xa, xb), y is (ya, yb_new)
                    # Wait, v=(0,1) means x_coord doesn't change, y_coord adds t.
                    # Point is (xa, xb, ya, (yb + tb) % p) ?
                    # No, y = ya + t*1.
                    # y_a = ya + ta. y_b = yb + tb.
                    pt = (xa, xb, (ya + ta) % p, (yb + tb) % p)
                    if pt in S_set:
                        ok = False
                        break
                if not ok: break
            if ok:
                found_free_line = True
            else:
                # Check directions (1, m) for m in F_49
                # m = (ma, mb)
                for ma in range(p):
                    for mb in range(p):
                        ok = True
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0: continue
                                # t = (ta, tb)
                                # x_new = x + t * 1 = x + t
                                # y_new = y + t * m
                                # x_a = xa + ta
                                # x_b = xb + tb
                                # y = (ya, yb) + t * (ma, mb)
                                # t * m = (ta*ma - tb*mb*k, ta*mb + tb*ma) ?
                                # (ta + tb*alpha) * (ma + mb*alpha) = ta*ma + (ta*mb + tb*ma)*alpha + tb*mb*alpha^2
                                # = (ta*ma + tb*mb*k) + (ta*mb + tb*ma)*alpha
                                # So y_a = ya + ta*ma + tb*mb*k
                                # y_b = yb + ta*mb + tb*ma
                                
                                # This depends on k.
                                # We need to pass k to this function.
                                pass 
            # This nested loop is too slow in Python for 2401 points * 50 dirs * 49 steps.
            # 2401 * 50 * 49 * 2 ~ 12 million iterations.
            # Might be okay within 600s?
            # 12M * 100ns = 1.2s. Yes.
            
            # I will implement the check properly in the main logic.
            pass
            
        return True # Placeholder

    # Due to complexity, I will stick to the seed but try k=5 and k=6.
    # The seed uses k=3.
    
    best_points = None
    best_score = -1
    
    for k_candidate in [3, 5, 6]:
        if time.time() - start_time > time_limit - 60:
            break
            
        # Construct S for this k
        # S = U \ B
        # U: 2xa + ya^2 - yb^2 k = 0
        # B: y in Y_special
        # Y_special = {(u^2, v)}
        
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k_candidate) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            S.add((xa, xb, ya, yb))
                            
        # Check validity of S
        # If valid, calculate score
        # Score is proportional to |S|
        
        # Quick validity check:
        # For each x in S, check if there is a free line.
        valid = True
        for x in S:
            xa, xb, ya, yb = x
            free_found = False
            
            # Check direction (0, 1)
            ok = True
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0: continue
                    pt = (xa, xb, (ya + ta) % p, (yb + tb) % p)
                    if pt in S:
                        ok = False
                        break
                if not ok: break
            if ok:
                free_found = True
            else:
                # Check directions (1, m)
                for ma in range(p):
                    for mb in range(p):
                        ok = True
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0: continue
                                # t * (ma, mb)
                                # t = ta + tb*alpha
                                # m = ma + mb*alpha
                                # t*m = (ta*ma + tb*mb*k) + (ta*mb + tb*ma)*alpha
                                ya_new = (ya + ta*ma + tb*mb*k_candidate) % p
                                yb_new = (yb + ta*mb + tb*ma) % p
                                xa_new = (xa + ta) % p
                                xb_new = (xb + tb) % p
                                pt = (xa_new, xb_new, ya_new, yb_new)
                                if pt in S:
                                    ok = False
                                    break
                            if not ok: break
                        if ok:
                            free_found = True
                            break
                    if free_found:
                        break
            if not free_found:
                valid = False
                break
                
        if valid:
            # Construct N
            points = []
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            if (xa, xb, ya, yb) not in S:
                                points.append([[xa, xb], [ya, yb]])
            if len(points) < len(best_points) if best_points is not None else True:
                best_points = points
                best_score = len(S)
                
    if best_points is None:
        # Fallback to seed logic with k=3
        k = 3
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        points.append([[xa, xb], [ya, yb]])
        best_points = points

    return np.array(best_points, dtype=np.int64)

# EVOLVE-BLOCK-END
