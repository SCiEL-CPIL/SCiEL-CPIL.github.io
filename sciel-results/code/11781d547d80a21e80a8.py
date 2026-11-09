# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a structural approach:
    Model as constraint satisfaction where each complement point selects a witness direction.
    Use iterative local search to maximize complement size.
    """
    deadline = time.time() + 620
    p = 7
    q = p * p
    k_field = 3

    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions: (1, m) for m in F_49, plus (0, 1)
    directions = [(1, m) for m in range(q)]
    directions.append((0, 1))
    n_dir = len(directions)

    # Precompute line memberships: for each point and direction, the other points on that line
    # line_points[p_idx, d_idx] = list of indices of other points on line through p_idx in direction d_idx
    # We'll store as a flat structure for efficiency
    # For each direction, precompute the line assignment: which line each point is on
    # A line through point (x,y) in direction (vx,vy) contains points (x + t*vx, y + t*vy) for t in F_49
    
    # Precompute: for each direction d_idx and each point p_idx, the set of other points on the line
    # This is expensive to store fully, so we'll compute on the fly but cache the line structure
    
    # Better approach: for each direction, partition all points into lines
    # Then for each point, we know which line it's on for each direction
    
    # For a given direction (vx, vy), two points (x1,y1) and (x2,y2) are on the same line iff
    # (x2-x1, y2-y1) = t*(vx, vy) for some t in F_49
    
    # Precompute for each direction: line_id for each point
    # line_id[d_idx, p_idx] = integer identifying which line
    # Also: line_members[d_idx, line_id] = list of points on that line
    
    # Actually, for the Nikodym condition: for each point x and direction d, 
    # all OTHER points on line L(x,d) must be in N.
    # So if x is in complement C, then for the chosen witness direction d_x,
    # all other points on L(x, d_x) must be in N.
    
    # This means: if x is in C with witness direction d_x, then no other point y on L(x, d_x) 
    # can be in C (unless y's witness direction is different and doesn't conflict).
    
    # Wait, re-reading the task: "every point x of F_49^2 must lie on a line {x + t*v : t in F_49}, 
    # with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    # So for EVERY point x (whether in N or C), there must exist a direction d such that
    # all other points on L(x, d) are in N.
    
    # If x is in C, then x is NOT in N, so we need some direction d where all OTHER points 
    # on L(x,d) are in N. This is the "witness" direction for x.
    
    # If x is in N, same condition applies.
    
    # So the constraint is: for each point x, there exists at least one direction d such that
    # all other points on L(x,d) are in N.
    
    # Equivalently: for each point x, it is NOT the case that for ALL directions d, 
    # there exists some other point y on L(x,d) that is in C.
    
    # Let's define: a point x is "blocked" in direction d if there exists y != x on L(x,d) with y in C.
    # x is valid if it is NOT blocked in at least one direction.
    # x is invalid if it is blocked in ALL directions.
    
    # We want to maximize |C| subject to: no point is invalid.
    
    # Precompute: for each point x and direction d, the list of other points on L(x,d)
    # Store as: line_other[x, d] = array of point indices (length q-1)
    
    # This is 2401 * 50 * 48 = ~5.7M entries, which is manageable.
    
    line_other = np.zeros((q * q, n_dir, q - 1), dtype=np.int32)
    
    for d_idx, (vx, vy) in enumerate(directions):
        for p_idx in range(q * q):
            px = p_idx // q
            py = p_idx % q
            for i, t in enumerate(range(1, q)):
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = nx * q + ny
                line_other[p_idx, d_idx, i] = n_idx

    # Now, for a given complement set C (boolean array), a point x is valid in direction d
    # if no point in line_other[x, d] is in C.
    
    # We can compute for each point x: the number of directions d where x is valid (i.e., 
    # no other point on L(x,d) is in C). x is overall valid if this count > 0.
    
    # Strategy: start with a small C, try to add points to C one at a time, checking validity.
    # Or: start with all in N, try to move points to C.
    
    # Let's use a greedy approach:
    # 1. Start with C = empty (all in N).
    # 2. Repeatedly try to add a point to C if it doesn't invalidate any point.
    # 3. Use local search / simulated annealing to improve.
    
    # For efficiency, maintain for each point x:
    # - in_C[x]: boolean
    # - valid_dirs[x]: number of directions where x is valid (no other point on L(x,d) in C)
    # x is valid iff valid_dirs[x] > 0
    
    # When we add point y to C:
    # - For each direction d, for each x on L(y, d) (x != y):
    #   - y is now in C, so x's validity in direction d is broken (if it wasn't already)
    #   - valid_dirs[x] decreases by 1 (if x was valid in direction d before)
    #   - If valid_dirs[x] becomes 0, x is now invalid -> this move is not allowed
    
    # When we remove point y from C:
    # - For each direction d, for each x on L(y, d) (x != y):
    #   - y is no longer in C, so x's validity in direction d may be restored
    #   - valid_dirs[x] increases by 1 (if x was not valid in direction d before)
    
    # Precompute: for each point y and direction d, the list of x such that y is on L(x, d)
    # This is the reverse of line_other: affects[y, d] = list of x where y in line_other[x, d]
    
    affects = [[] for _ in range(q * q)]
    for d_idx in range(n_dir):
        for x_idx in range(q * q):
            for y_idx in line_other[x_idx, d_idx]:
                affects[y_idx].append((x_idx, d_idx))
    
    # Convert to arrays for speed
    affects_arr = [np.array(lst, dtype=np.int32).reshape(-1, 2) if lst else np.empty((0, 2), dtype=np.int32) for lst in affects]
    
    # Initialize: all in N, C = empty
    in_C = np.zeros(q * q, dtype=bool)
    # valid_dirs[x] = number of directions where no other point on L(x,d) is in C
    # Initially C is empty, so all points are valid in all directions
    valid_dirs = np.full(q * q, n_dir, dtype=np.int32)
    
    # Greedy phase: try to add points to C
    # Priority: points that are "easiest" to add (those whose addition breaks the fewest valid_dirs)
    
    def try_add(y):
        """Try to add point y to C. Returns True if successful."""
        # Check if adding y would make any point invalid
        for x_idx, d_idx in affects_arr[y]:
            if valid_dirs[x_idx] == 1:
                # This point would become invalid
                return False
        # Safe to add
        in_C[y] = True
        for x_idx, d_idx in affects_arr[y]:
            valid_dirs[x_idx] -= 1
        return True
    
    def try_remove(y):
        """Remove point y from C. Always safe."""
        in_C[y] = False
        for x_idx, d_idx in affects_arr[y]:
            valid_dirs[x_idx] += 1
    
    # Phase 1: Greedy addition
    # Order points by some heuristic and try to add each
    # Use random order for diversity
    
    best_C_size = 0
    best_in_C = in_C.copy()
    best_valid_dirs = valid_dirs.copy()
    
    # Run multiple greedy passes with different random orders
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 120:
            break
        
        in_C[:] = False
        valid_dirs[:] = n_dir
        
        order = list(range(q * q))
        random.shuffle(order)
        
        for y in order:
            if try_add(y):
                pass
        
        C_size = np.sum(in_C)
        if C_size > best_C_size:
            best_C_size = C_size
            best_in_C = in_C.copy()
            best_valid_dirs = valid_dirs.copy()
    
    # Phase 2: Local search on the best solution
    # Try to add points that are currently not in C, or swap operations
    in_C = best_in_C.copy()
    valid_dirs = best_valid_dirs.copy()
    
    # Try to add each point not in C
    not_in_C = np.where(~in_C)[0]
    random.shuffle(not_in_C)
    
    for y in not_in_C:
        if time.time() > deadline - 60:
            break
        if try_add(y):
            best_C_size += 1
    
    # Phase 3: More aggressive local search
    # Try pairs: remove one point, add another
    # Or: try to add points in a different order
    
    # Let's try a simulated annealing-like approach
    current_C_size = np.sum(in_C)
    
    for iteration in range(5000):
        if time.time() > deadline - 30:
            break
        
        # Try to add a random point not in C
        if np.sum(in_C) < q * q:
            y = np.random.choice(not_in_C)
            if try_add(y):
                current_C_size += 1
                continue
        
        # Try to remove a random point from C (to allow adding others later)
        if current_C_size > 0:
            in_C_indices = np.where(in_C)[0]
            y = np.random.choice(in_C_indices)
            try_remove(y)
            current_C_size -= 1
    
    # Final result
    in_N = ~in_C
    
    # Build output
    points = []
    for i in range(q * q):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: return all points
        for x in range(q):
            xa, xb = to_pair(x)
            for y in range(q):
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
