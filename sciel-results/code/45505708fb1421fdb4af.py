# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.

    Strategy:
    1. Start with a known valid Nikodym set based on a Hermitian unital
       construction.
    2. Use a randomized greedy algorithm with multiple restarts to minimize
       the set size. At each step, randomly select a point from the set of
       points whose removal does not violate the Nikodym property (cost 0).
    3. This explores different local minima compared to a deterministic
       sequential pass.
    """
    deadline = time.time() + 620  # safety margin before 630s limit

    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    # Find smallest quadratic non-residue mod p
    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3  # fallback

    # Represent elements of F_q as integers 0..48: element (a, b) -> a + b*p
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Field arithmetic in F_q
    def f_add(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        return to_int((xa + ya) % p, (xb + yb) % p)

    def f_sub(x, y):
        return f_add(x, f_neg(y))

    def f_neg(x):
        xa, xb = to_pair(x)
        return to_int((-xa) % p, (-xb) % p)

    def f_mul(x, y):
        xa, xb = to_pair(x)
        ya, yb = to_pair(y)
        # (xa + xb*alpha)(ya + yb*alpha) = xa*ya + (xa*yb + xb*ya)*alpha + xb*yb*alpha^2
        # = xa*ya + xb*yb*k_field + (xa*yb + xb*ya)*alpha
        real = (xa * ya + xb * yb * k_field) % p
        imag = (xa * yb + xb * ya) % p
        return to_int(real, imag)

    def f_inv(x):
        if x == 0:
            return 0
        xa, xb = to_pair(x)
        # Inverse of (xa + xb*alpha) = (xa - xb*alpha)/(xa^2 - xb^2*k)
        denom = (xa * xa - xb * xb * k_field) % p
        if denom == 0:
            return 0
        denom_inv = pow(denom, p - 2, p)
        real = (xa * denom_inv) % p
        imag = (-xb * denom_inv) % p
        return to_int(real, imag)

    def f_div(x, y):
        return f_mul(x, f_inv(y))

    # Directions: v = (1, m) for m in F_q, or v = (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))  # vertical
    n_dir = len(directions)

    # All points in F_q^2 as integer pairs
    all_points = [(x, y) for x in range(q) for y in range(q)]
    point_index = {(x, y): i for i, (x, y) in enumerate(all_points)}
    idx_to_point = {i: pt for pt, i in point_index.items()}

    # Build the base Nikodym set
    # Hermitian unital: U = {(x, y) : Trace(x) + Norm(y) = 0}
    # Trace((a,b)) = 2a, Norm((a,b)) = a^2 - b^2*k
    # Blocking set B = points of U whose y-coordinate has y_a a square mod p
    # Excluded set S = U \ B
    # N = complement of S

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(to_int((u * u) % p, v))

    # Determine which points are excluded (not in N)
    excluded = set()
    for x_idx in range(q):
        xa, xb = to_pair(x_idx)
        trace_x = (2 * xa) % p
        for y_idx in range(q):
            ya, yb = to_pair(y_idx)
            norm_y = (ya * ya - yb * yb * k_field) % p
            if (trace_x + norm_y) % p == 0:
                # In unital
                if y_idx not in y_special:
                    # Excluded
                    excluded.add((x_idx, y_idx))

    # Initialize N: all points except excluded
    in_N = [True] * (q * q)
    for (x_idx, y_idx) in excluded:
        in_N[point_index[(x_idx, y_idx)]] = False

    # Initialize not_in_N_count
    # For each point x and direction v, count points on line(x,v)\{x} not in N
    not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)

    # Compute initial counts
    for d_idx, (vx, vy) in enumerate(directions):
        for x_idx in range(q):
            for y_idx in range(q):
                point = (x_idx, y_idx)
                p_idx = point_index[point]
                count = 0
                for t in range(1, q):
                    tx = f_add(x_idx, f_mul(t, vx))
                    ty = f_add(y_idx, f_mul(t, vy))
                    t_point = (tx, ty)
                    if t_point not in point_index:
                        continue
                    t_idx = point_index[t_point]
                    if not in_N[t_idx]:
                        count += 1
                not_in_N_count[p_idx, d_idx] = count

    # A point is "valid" if min over directions of not_in_N_count[point, d] == 0
    # We want to remove points from N.
    
    # Precompute affects: for each point p, list of (x, d) pairs where p is on line(x, d)\{x}
    affects = [[] for _ in range(q * q)]
    for p_idx in range(q * q):
        px, py = idx_to_point[p_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tx = f_sub(px, f_mul(t, vx))
                ty = f_sub(py, f_mul(t, vy))
                if (tx, ty) in point_index:
                    x_idx = point_index[(tx, ty)]
                    affects[p_idx].append((x_idx, d_idx))

    def run_greedy(seed):
        rng = random.Random(seed)
        # Copy state
        cur_in_N = in_N[:]
        cur_counts = not_in_N_count.copy()
        
        # cover_count[x] = number of directions d where cur_counts[x, d] == 0
        cur_cover_count = np.array([np.sum(cur_counts[i] == 0) for i in range(q * q)], dtype=np.int32)
        
        # Set of points currently in N
        active_points = set([i for i in range(q * q) if cur_in_N[i]])
        
        while active_points:
            if time.time() > deadline - 10:
                break
            
            # Find all points in active_points that have cost 0
            # Cost of removing p: number of x in affects[p] where cur_cover_count[x] == 1 and cur_counts[x, d] == 0
            # If we remove p, and for some (x, d) in affects[p], cur_counts[x, d] was 0, 
            # then cur_counts[x, d] becomes 1.
            # If cur_cover_count[x] was 1, it becomes 0 -> x becomes invalid.
            # So cost = count of such x.
            
            candidates = []
            for p_idx in active_points:
                cost = 0
                for (x, d) in affects[p_idx]:
                    if cur_counts[x, d] == 0:
                        if cur_cover_count[x] == 1:
                            cost += 1
                            # Early exit: if cost > 0, it's not a zero-cost candidate
                            # But we need to know if it's 0 or not.
                            # If we just need "is it 0", we can break early if we find one.
                            break
                if cost == 0:
                    candidates.append(p_idx)
            
            if not candidates:
                break
            
            # Pick a random candidate
            p_idx = rng.choice(candidates)
            
            # Remove p_idx
            cur_in_N[p_idx] = False
            active_points.remove(p_idx)
            
            # Update counts
            for (x, d) in affects[p_idx]:
                old_val = cur_counts[x, d]
                cur_counts[x, d] = old_val + 1
                if old_val == 0:
                    cur_cover_count[x] -= 1
        
        return cur_in_N

    best_in_N = in_N[:]
    best_size = sum(best_in_N)
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        res_in_N = run_greedy(i)
        res_size = sum(res_in_N)
        
        if res_size < best_size:
            best_size = res_size
            best_in_N = res_in_N
    
    # Build result
    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    if not points:
        # Fallback: return the base set
        points = []
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                if (x_idx, y_idx) not in excluded:
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
