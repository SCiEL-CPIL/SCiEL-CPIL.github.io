# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 (q=49, p=7, alpha^2=3).
    
    Strategy: 
    1. Start with a known valid Nikodym set (the Hermitian unital construction).
    2. Use greedy local search with bitsets to iteratively remove points from N
       (adding them to the complement) as long as validity is maintained.
    3. A point x in N is "removable" if for every point y that relies on x to
       satisfy its line condition, there exists another point in N (other than y
       and x) that can serve as the direction for y's line.
    
    Actually, the Nikodym condition is: for every point x in F_q^2, there exists
    a direction v such that the line {x + t*v : t in F_q, t != 0} is in N.
    So x itself doesn't need to be in N, but the other q-1 points on the line do.
    
    Let me re-think: N is a Nikodym set if for every point p in F_q^2, there
    exists a direction v such that all points {p + t*v : t in F_q, t != 0} are in N.
    Note: p itself is NOT required to be in N.
    
    So the complement C = F_q^2 \ N can contain points, but for each c in C, 
    there must be a direction v such that all other points on the line through c
    in direction v are in N.
    
    Greedy approach: Start with a valid N. Try to move each point from N to C.
    A point x can be moved to C if for x's own requirement, there's a direction v
    where all other points on the line are still in N (they were, and we're only
    removing x, which isn't on the "other" part). But also, removing x might break
    other points' requirements: for any point y that used x as part of its line,
    y needs an alternative direction.
    
    This is complex. Let me use a simpler valid construction and then try to
    maximize complement via a different structural approach.
    
    Alternative: Use the fact that for q = p^2, there exist Nikodym sets of size
    q^2 - q + 1 (complement size q-1) from projective plane constructions, but
    that's small complement. The current gives complement ~ q*(q+1)/2 = 49*25/2.
    
    Let me try: start with N = all points (complement 0, valid trivially since 
    every line is fully in N). Then greedily add points to complement.
    A point x can be added to complement if: for x itself, there's a direction v 
    where all other points on line through x in direction v are in N. AND for every
    other point y that might have used x, y still has some valid direction.
    
    This is hard to check incrementally. Let me use a known good construction 
    and then do a local search on the complement.
    
    Actually, let me try a completely different angle: use the minimum size 
    Nikodym set constructions from finite geometry. For q=49, the smallest known
    Nikodym sets have size around q^2 - q = 2352, giving complement 49. But the
    current gives complement ~ 49*25/2 = 612.5, so current is already much better.
    
    Wait, let me recompute: current excludes points in U \ B.
    |U| = q + 1 = 50 (Hermitian unital in F_q^2 has q+1 points? No, that's for 
    the projective case. In affine F_q^2, the Hermitian curve has q+1 points.)
    Actually the unital U = {(x,y) : Tr(x) + Nm(y) = 0} where Tr(x)=2*x_a, 
    Nm(y) = y_a^2 - y_b^2*3.
    For each (y_a, y_b), we need 2*x_a = -(y_a^2 - y_b^2*3) mod 7, so x_a is 
    determined, and x_b is free (7 choices). So |U| = 7*49 = 343.
    
    Y = {(u^2 mod 7, v) : u,v in F_7}. Squares mod 7: {0,1,2,4}. So |Y| = 4*7 = 28.
    B = U points with y in Y. For each y in Y (28 choices), x_a is determined, x_b free (7). |B| = 28*7 = 196.
    S = U \ B, |S| = 343 - 196 = 147.
    N = F_q^2 \ S, |N| = 2401 - 147 = 2254. Complement = 147.
    
    Score = 147 / seed_complement. Seed gives same 147, so score 1.0.
    
    To improve, I need complement > 147. Let me try to enlarge S (the excluded set)
    while maintaining validity.
    
    The validity condition: for every point p in F_q^2, there exists a direction v 
    (either (1,m) for some m, or (0,1)) such that all points p + t*v for t != 0 
    are in N (i.e., not in S).
    
    Equivalently: for every p, NOT(all lines through p have at least one point of S 
    other than p). I.e., for every p, there exists a line through p whose other q-1 
    points are all in N.
    
    Let me try a greedy approach: start with S = empty (N = all, valid). 
    Greedily add points to S. A point x can be added to S if after adding, 
    for every point p, there still exists a line through p with all other points in N.
    
    This is expensive to check naively. With bitsets it might work.
    
    Let me implement: represent N as a bitset of 2401 bits. For each point p, 
    precompute the set of lines through p (there are q+1 = 50 lines). For each 
    line, store the bitset of points on that line (excluding p). A point p is 
    "covered" if there exists a line L through p such that (N & line_points[L]) == line_points[L].
    
    When we remove a point x from N (add to S), we need to check that no point 
    p becomes uncovered. A point p becomes uncovered if for ALL lines through p, 
    the line's points are not all in N. Removing x only affects lines that contain x.
    For a line L through p, removing x from N makes L invalid for p iff x was in 
    line_points[L] AND (N_before & line_points[L]) == line_points[L] but 
    (N_after & line_points[L]) != line_points[L].
    
    So x can be removed iff: for every point p such that x is on a line through p 
    (i.e., x != p and x is on some line through p), p still has at least one valid 
    line. This is complex but let me try a simpler heuristic.
    
    Given time constraints, let me try a randomized greedy: repeatedly pick a 
    random point in N, try to remove it, check validity (with early termination), 
    keep if valid. Use bitsets for speed.
    """
    p = 7
    q = p * p  # 49
    total = q * q  # 2401
    deadline = time.time() + 570  # safety margin
    
    # Field operations: F_49 = F_7[alpha], alpha^2 = 3
    k = 3  # alpha^2 = 3
    
    def field_mul(a, b):
        """Multiply two elements of F_49, each as (a0, a1) representing a0 + a1*alpha."""
        a0, a1 = a
        b0, b1 = b
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + (a0*b1 + a1*b0)*alpha + a1*b1*alpha^2
        # = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    def field_div(a, b):
        """Divide a by b in F_49."""
        # b^{-1} = conjugate(b) / norm(b)
        # conjugate(a + b*alpha) = a - b*alpha
        # norm(a + b*alpha) = (a + b*alpha)(a - b*alpha) = a^2 - b^2*alpha^2 = a^2 - b^2*k
        b0, b1 = b
        norm = (b0 * b0 - b1 * b1 * k) % p
        if norm == 0:
            return None
        norm_inv = pow(norm, p - 2, p)  # Fermat's little theorem
        conj_b = (b0, (-b1) % p)
        return field_mul(a, field_mul(conj_b, (norm_inv, 0)))
    
    # Enumerate all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
    
    point_index = {pt: i for i, pt in enumerate(points)}
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # v = (1, m)
    directions.append(((0, 0), (0, 1)))  # v = (0, 1)
    
    # For each point, precompute the lines through it (as sets of point indices)
    # There are q+1 = 50 lines through each point
    # line_points[p_idx][d_idx] = frozenset of point indices on line through p in direction d, excluding p
    # Actually, let's store as bitsets for efficiency
    
    # For each point p and each direction d, compute the set of points on the line
    # Line: {p + t*v : t in F_49}
    # We need points other than p, so t != 0
    
    # Precompute for each (p_idx, d_idx) the bitmask of points on the line (excluding p)
    # This is 2401 * 50 bitmasks of 2401 bits each = 120,050 * 2401 bits ~ 36 MB, OK
    
    # Let's use numpy uint64 arrays for bitsets
    # 2401 bits = 38 uint64s (38*64=2432 >= 2401)
    num_words = (total + 63) // 64  # 38
    
    # line_mask[p_idx * num_dirs + d_idx] = array of num_words uint64
    # This is 2401 * 50 * 38 * 8 bytes = 36.2 MB, acceptable
    
    num_dirs = len(directions)  # 50
    
    # Precompute line masks
    line_masks = np.zeros((total, num_dirs, num_words), dtype=np.uint64)
    
    for p_idx, p_pt in enumerate(points):
        for d_idx, v in enumerate(directions):
            vx, vy = v
            for t_idx in range(q):
                if t_idx == 0:
                    continue
                # t as field element
                ta = t_idx // p
                tb = t_idx % p
                # p + t*v
                tv_x = field_mul((ta, tb), vx)
                tv_y = field_mul((ta, tb), vy)
                new_pt = field_add(p_pt[0], tv_x)
                new_pt = (new_pt, field_add(p_pt[1], tv_y))
                if new_pt in point_index:
                    ni = point_index[new_pt]
                    word_idx = ni // 64
                    bit_idx = ni % 64
                    line_masks[p_idx, d_idx, word_idx] |= (1 << bit_idx)
    
    # Now start with N = all points (complement = 0)
    # N_mask: bitset of points in N
    N_mask = np.ones((total, num_words), dtype=np.uint64)
    # Set only valid bits
    for i in range(total):
        word_idx = i // 64
        bit_idx = i % 64
        if i < total:
            N_mask[i, word_idx] = (1 << bit_idx)
    # Actually, let's represent N as a single bitset of num_words
    N_bitset = np.zeros(num_words, dtype=np.uint64)
    for i in range(total):
        word_idx = i // 64
        bit_idx = i % 64
        N_bitset[word_idx] |= (1 << bit_idx)
    
    # Check if a point p is covered: exists direction d such that 
    # (N_bitset & line_masks[p_idx, d_idx]) == line_masks[p_idx, d_idx]
    def is_covered(p_idx, N_b):
        for d_idx in range(num_dirs):
            lm = line_masks[p_idx, d_idx]
            # Check if N_b contains all bits in lm
            all_in = True
            for w in range(num_words):
                if (N_b[w] & lm[w]) != lm[w]:
                    all_in = False
                    break
            if all_in:
                return True
        return False
    
    # Try to greedily add points to complement (remove from N)
    # Start with S = empty
    S = set()
    
    # Order of removal: try points that are "least critical" first
    # A point x is critical for p if x is on a line through p that is the ONLY valid line for p
    # Let's just try random order
    
    import random
    random.seed(42)
    order = list(range(total))
    random.shuffle(order)
    
    for p_idx in order:
        if time.time() > deadline - 30:
            break
        
        # Try to remove point p_idx from N
        # Check: after removal, is every point still covered?
        # Only points that might be affected are those for which p_idx is on one of their lines
        # And specifically, only if the line containing p_idx was the only valid line
        
        # Save old N
        old_N = N_bitset.copy()
        
        # Remove p_idx from N
        word_idx = p_idx // 64
        bit_idx = p_idx % 64
        N_bitset[word_idx] &= ~(1 << bit_idx)
        
        # Check validity: for each point, is it still covered?
        # Optimization: only check points for which p_idx could have been on a valid line
        # A point y is affected if p_idx is on some line through y
        # Lines through y contain p_idx if y, p_idx, and some direction are collinear
        # For a given y and p_idx, there's a unique direction (if y != p_idx)
        
        valid = True
        for y_idx in range(total):
            if y_idx == p_idx:
                # y = p_idx: we removed p_idx from N. 
                # For y to be covered, there must be a direction d such that all other 
                # points on line through y in direction d are in N.
                # p_idx is y, so p_idx is not on the "other points" part. 
                # But wait, the line through y in direction d contains y + t*v for t != 0.
                # y itself is not included. So removing y from N doesn't affect y's own coverage
                # directly. But it might affect other points.
                # Actually, for y's coverage, we need a line through y where all OTHER points 
                # are in N. Removing y from N doesn't remove y from any line's "other points" 
                # because y is the base point. So y's coverage is unchanged.
                continue
            
            # Check if y is still covered
            covered = False
            for d_idx in range(num_dirs):
                lm = line_masks[y_idx, d_idx]
                all_in = True
                for w in range(num_words):
                    if (N_bitset[w] & lm[w]) != lm[w]:
                        all_in = False
                        break
                if all_in:
                    covered = True
                    break
            if not covered:
                valid = False
                break
        
        if valid:
            S.add(p_idx)
        else:
            # Restore
            N_bitset[word_idx] |= (1 << bit_idx)
    
    # Build result: N = all points not in S
    result_points = []
    for i in range(total):
        if i not in S:
            xa, xb = points[i][0]
            ya, yb = points[i][1]
            result_points.append([[xa, xb], [ya, yb]])
    
    if len(result_points) == 0:
        # Fallback: return all points
        result_points = [[[xa, xb], [ya, yb]] for (xa, xb), (ya, yb) in points]
    
    return np.array(result_points, dtype=np.int64)


# EVOLVE-BLOCK-END
