# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Precompute field operations for F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) representing a + b*alpha
    # alpha^2 = 3, so (a,b)*(c,d) = (ac + 3bd, ad + bc)
    
    k_const = 3  # alpha^2 = 3
    
    # Build multiplication table: mul[i][j] = i * j in F_49, where i,j are 0..48
    # Map (a,b) -> a + 7*b
    def to_idx(a, b):
        return a + 7 * b
    
    def from_idx(idx):
        return idx % 7, idx // 7
    
    # Precompute all field elements as pairs
    elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = from_idx(i)
        for j in range(q):
            aj, bj = from_idx(j)
            # (ai + bi*alpha) * (aj + bj*alpha) = ai*aj + ai*bj*alpha + bi*aj*alpha + bi*bj*alpha^2
            # = (ai*aj + bi*bj*3) + (ai*bj + bi*aj)*alpha
            res_a = (ai * aj + bi * bj * k_const) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = to_idx(res_a, res_b)
    
    # Addition table (simple)
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = from_idx(i)
        for j in range(q):
            aj, bj = from_idx(j)
            add_table[i, j] = to_idx((ai + aj) % p, (bi + bj) % p)
    
    # Subtraction table
    sub_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = from_idx(i)
        for j in range(q):
            aj, bj = from_idx(j)
            sub_table[i, j] = to_idx((ai - aj) % p, (bi - bj) % p)
    
    # Division table: div[i][j] = i / j (j != 0)
    div_table = np.zeros((q, q), dtype=np.int32)
    # First find inverses
    inv_table = np.zeros(q, dtype=np.int32)
    for i in range(1, q):
        for j in range(1, q):
            if mul_table[i, j] == 0:
                inv_table[i] = j
                break
    for i in range(q):
        for j in range(1, q):
            div_table[i, j] = mul_table[i, inv_table[j]]
    
    # Precompute all lines
    # A line is defined by a point and a direction
    # Directions: (1, m) for m in F_49 (49 directions) and (0, 1) (vertical)
    # Total 50 directions
    
    # For each point, precompute which lines pass through it
    # Point index: point_idx = x_idx * q + y_idx, where x_idx, y_idx in 0..48
    
    # For each direction, and each line in that direction, store the set of points on it
    # There are 50 directions, and for each direction, q lines (each with q points)
    # Total: 50 * q * q = 50 * 49 * 49 = 120050 line-point incidences
    
    # directions: 0..48 are (1, m) for m=0..48, direction 49 is (0,1)
    # For direction (1, m): line through (x0, y0) is {(x0 + t, y0 + t*m) : t in F_49}
    # For direction (0, 1): line through (x0, y0) is {(x0, y0 + t) : t in F_49}
    
    # Precompute: for each point, list of (direction, line_id) pairs
    # And for each (direction, line_id), the list of points on that line
    
    # Let's index lines: for direction d, line l (l in 0..q-1)
    # For direction (1, m) where m = d (d in 0..48):
    #   The line with "intercept" c passes through points (x, y) where y - x*m = c
    #   So line_id = c = y - x*m (in F_49)
    # For direction (0, 1):
    #   The line with x-coordinate c passes through points (c, y)
    #   So line_id = c = x
    
    # For each point, for each direction, compute the line it's on
    # point_lines[point_idx][d] = line_id
    
    point_lines = np.zeros((total_points, 50), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            # For each direction d = 0..48: direction (1, d)
            for d in range(q):
                # line_id = y - x * d
                xd = mul_table[x_idx, d]
                line_id = sub_table[y_idx, xd]
                point_lines[point_idx, d] = line_id
            # Direction 49: (0, 1), line_id = x
            point_lines[point_idx, 49] = x_idx
    
    # For each (direction, line_id), compute the set of points on that line
    # line_points[d][l] = list of point indices on line l in direction d
    
    # We'll store this as a 2D array: line_points[d][l] is an array of q point indices
    line_points = np.zeros((50, q, q), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            for d in range(50):
                l = point_lines[point_idx, d]
                # We need to figure out the position of this point on the line
                # For direction (1, d): points are (x0 + t, y0 + t*d) for t in F_49
                # The line is determined by its "base" point. Let's use the point where x=0
                # For direction (1, d): when x=0, y = line_id (since y - 0*d = line_id)
                # So base point is (0, line_id)
                # Point (x_idx, y_idx) is at t = x_idx (since x = 0 + t)
                # For direction (0, 1): line is x = line_id, base point is (line_id, 0)
                # Point (x_idx, y_idx) is at t = y_idx
                
                if d < q:
                    # direction (1, d), t = x_idx
                    t = x_idx
                    line_points[d, l, t] = point_idx
                else:
                    # direction (0, 1), t = y_idx
                    t = y_idx
                    line_points[d, l, t] = point_idx
    
    # Now, for a given set N (represented as a boolean array of size total_points),
    # check if it's a valid Nikodym set:
    # For every point x (whether in N or not), there exists a direction d such that
    # all points on the line through x in direction d, EXCEPT x itself, are in N.
    
    # For efficiency, precompute for each point and direction, the other 48 points on that line
    # other_points[point_idx][d] = array of 48 point indices (the line minus the point)
    
    other_points = np.zeros((total_points, 50, q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            for d in range(50):
                l = point_lines[point_idx, d]
                # The line has q points indexed 0..q-1
                # Find the position of this point on the line
                if d < q:
                    pos = x_idx
                else:
                    pos = y_idx
                # The other points are at positions 0..q-1 except pos
                count = 0
                for t in range(q):
                    if t != pos:
                        other_points[point_idx, d, count] = line_points[d, l, t]
                        count += 1
    
    # Greedy algorithm: start with all points in N, try to remove points
    # A point can be removed if after removal, it still has a valid line
    # (i.e., some direction where all 48 other points are still in N)
    
    def is_valid_nikodym(N_mask):
        """Check if N_mask is a valid Nikodym set. N_mask[i] = True if point i is in N."""
        for point_idx in range(total_points):
            found = False
            for d in range(50):
                # Check if all other 48 points on this line are in N
                # other_points[point_idx, d] is an array of 48 indices
                others = other_points[point_idx, d]
                if np.all(N_mask[others]):
                    found = True
                    break
            if not found:
                return False
        return True
    
    def can_remove(N_mask, point_idx):
        """Check if point_idx can be removed from N and still be a valid Nikodym set.
        After removal, point_idx is not in N, so it needs a line where all others are in N.
        Also, for other points, removing point_idx might break their validity if they relied on this line.
        This is complex, so we use a simplified check: just verify the removed point still has a valid line.
        The full validity check is done periodically."""
        if not N_mask[point_idx]:
            return False
        # After removal, point_idx needs a direction where all 48 others are in N
        for d in range(50):
            others = other_points[point_idx, d]
            if np.all(N_mask[others]):
                return True
        return False
    
    # Strategy: Start with all points in N. Greedily remove points in random order.
    # Use multiple restarts with different random seeds.
    
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget, but we want safety margin
    
    best_complement = 0
    best_mask = None
    
    # Initial mask: all True
    initial_mask = np.ones(total_points, dtype=bool)
    
    # Try multiple random greedy removals
    np.random.seed(42)
    random.seed(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        mask = initial_mask.copy()
        # Shuffle point order
        order = np.random.permutation(total_points)
        
        removed = 0
        for point_idx in order:
            if time.time() > deadline - 30:
                break
            if mask[point_idx]:
                if can_remove(mask, point_idx):
                    mask[point_idx] = False
                    removed += 1
        
        # Verify full validity
        if is_valid_nikodym(mask):
            if removed > best_complement:
                best_complement = removed
                best_mask = mask.copy()
        
        # Try local search: try to remove more points
        # Actually, let's also try adding back points that were kept but could be removed
        # in a different order - this is complex, skip for now
    
    # Also try a second phase: start from the best found, try to remove more
    if best_mask is not None:
        if time.time() < deadline - 60:
            mask = best_mask.copy()
            # Try to remove more points in random order
            order = np.random.permutation(total_points)
            for point_idx in order:
                if time.time() > deadline - 30:
                    break
                if mask[point_idx]:
                    if can_remove(mask, point_idx):
                        mask[point_idx] = False
                        best_complement += 1
                        best_mask = mask.copy()
    
    # Fallback: if no valid set found (shouldn't happen since full set is valid),
    # use the full set
    if best_mask is None:
        best_mask = initial_mask.copy()
        best_complement = 0
    
    # Convert mask to list of points
    # Point index = x_idx * q + y_idx
    # x_idx = x_a + 7 * x_b, y_idx = y_a + 7 * y_b
    points = []
    for point_idx in range(total_points):
        if best_mask[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a = x_idx % 7
            x_b = x_idx // 7
            y_a = y_idx % 7
            y_b = y_idx // 7
            points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
