# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2 using a greedy algorithm that
    maximizes the complement (excluded) set.

    The key insight: for each point x, we need a direction v such that all other
    points on the line through x in direction v are in N. If we exclude a set C
    (complement of N), then for each x in C, the line through x in some direction
    must have all its other q-1 points in N (i.e., not in C).

    Greedy approach: iteratively add points to C if possible. A point x can be
    added to C if there exists a direction v such that no other point on the line
    {x + t*v : t != 0} is in C.

    We use a priority queue to prioritize points that are "easiest" to exclude
    (those with fewer conflicts).
    """
    p = 7
    q = p * p  # 49

    # Precompute field operations
    # F_49 = F_7[alpha] where alpha^2 = 3 (3 is a quadratic non-residue mod 7)
    # Element is (a, b) representing a + b*alpha
    
    # Precompute multiplication table for F_49
    # mul[(a1,b1)][(a2,b2)] = (a,b) where (a1+b1*alpha)*(a2+b2*alpha) = a + b*alpha
    # (a1+b1*alpha)*(a2+b2*alpha) = a1*a2 + (a1*b2+a2*b1)*alpha + b1*b2*alpha^2
    # = (a1*a2 + 3*b1*b2) + (a1*b2+a2*b1)*alpha
    mul_table = {}
    for a1 in range(p):
        for b1 in range(p):
            for a2 in range(p):
                for b2 in range(p):
                    a = (a1 * a2 + 3 * b1 * b2) % p
                    b = (a1 * b2 + a2 * b1) % p
                    mul_table[(a1, b1, a2, b2)] = (a, b)

    # Directions: v = (1, m) for m in F_49, and v = (0, 1)
    # Total: q + 1 = 50 directions
    # Each direction is (vx_a, vx_b, vy_a, vy_b)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, 0, ma, mb))  # v = (1, m)
    directions.append((0, 1, 0, 1))  # v = (0, 1)

    n_dirs = len(directions)  # 50

    # For each direction, precompute the lines (parallel classes)
    # A line in direction v through point x: {x + t*v : t in F_49}
    # Points on the same line have the same "quotient" when divided by v
    
    # For v = (1, m): line through (x, y) consists of points (x + t, y + t*m) for t in F_49
    # The invariant is y - x*m (in F_49^2 sense)
    # For v = (0, 1): line through (x, y) consists of points (x, y + t) for t in F_49
    # The invariant is x
    
    # Let's index points as 0..2400
    # Point index: ((xa * p + xb) * p + ya) * p + yb
    def point_to_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    total_points = q * q  # 2401

    # For each direction, assign each point to a line ID
    # line_id[dir_idx][point_idx] = line identifier
    # For v = (1, m) with m = (ma, mb):
    #   Line through (x, y) where x = (xa, xb), y = (ya, yb)
    #   Invariant: y - x*m in F_49
    #   Compute (ya, yb) - (xa, xb)*(ma, mb)
    
    line_ids = []  # line_ids[d][i] = line id for point i in direction d
    
    for d in range(n_dirs):
        vx_a, vx_b, vy_a, vy_b = directions[d]
        lid = np.zeros(total_points, dtype=np.int32)
        
        if vx_a == 0 and vx_b == 0:
            # v = (0, 1): lines are vertical, invariant is x
            for xa in range(p):
                for xb in range(p):
                    line_id = xa * p + xb
                    for ya in range(p):
                        for yb in range(p):
                            idx = point_to_idx(xa, xb, ya, yb)
                            lid[idx] = line_id
        else:
            # v = (1, m) with m = (vy_a, vy_b)
            # Invariant: y - x*m
            # x*m = (xa, xb)*(vy_a, vy_b)
            for xa in range(p):
                for xb in range(p):
                    # Compute x*m
                    xm_a = (xa * vy_a + 3 * xb * vy_b) % p
                    xm_b = (xa * vy_b + xb * vy_a) % p
                    for ya in range(p):
                        for yb in range(p):
                            # y - x*m
                            inv_a = (ya - xm_a) % p
                            inv_b = (yb - xm_b) % p
                            line_id = inv_a * p + inv_b
                            idx = point_to_idx(xa, xb, ya, yb)
                            lid[idx] = line_id
        line_ids.append(lid)

    # Greedy algorithm with priority queue
    # For each point, track which directions are "available" (all other points on that line are not in C)
    
    import heapq
    
    # conflicts[dir_idx][line_id] = count of points in C on that line
    conflicts = [np.zeros(q, dtype=np.int32) for _ in range(n_dirs)]
    
    # For each point, available_dirs = set of directions where the line has no other C points
    # Initially all directions are available for all points
    
    # Priority: number of available directions (fewer first = harder to place, do first)
    # Actually, we want to exclude as many points as possible.
    # A point can be excluded if at least one direction is available.
    # Greedy: exclude points that have the fewest available directions first (most constrained)
    
    in_C = np.zeros(total_points, dtype=bool)
    # avail_count[i] = number of directions where point i's line has 0 other C points
    avail_count = np.full(total_points, n_dirs, dtype=np.int32)
    
    # For tracking which directions are still available for each point
    # We'll use a bitset or just recompute when needed
    
    # Priority queue: (avail_count, point_idx)
    heap = [(n_dirs, i) for i in range(total_points)]
    heapq.heapify(heap)
    
    # To avoid stale entries, track current avail_count
    current_avail = avail_count.copy()
    
    excluded_count = 0
    deadline = time.time() + 580  # Safety margin
    
    while time.time() < deadline:
        if not heap:
            break
        
        ac, i = heapq.heappop(heap)
        
        if ac != current_avail[i]:
            continue  # Stale entry
        
        if in_C[i]:
            continue
        
        if ac == 0:
            continue  # Cannot exclude this point
        
        # Try to exclude point i
        # Find a direction where the line has 0 other C points
        # We need to check which directions are available
        
        # For efficiency, check directions in order
        placed = False
        for d in range(n_dirs):
            lid = line_ids[d][i]
            if conflicts[d][lid] == 0:
                # This direction is available
                in_C[i] = True
                excluded_count += 1
                
                # Update conflicts for this line in this direction
                conflicts[d][lid] += 1
                
                # Update avail_count for all other points on this line
                # Points j on the same line in direction d now have one less available direction
                # (if they weren't already in C)
                # But we need to be careful: the conflict count went from 0 to 1,
                # so for other points on this line, this direction is no longer available
                # (since there's now 1 C point on the line, which is i itself)
                # Wait - the condition is: line through x in direction v has all OTHER points in N
                # So if x is in C, the other q-1 points on the line must be in N
                # When we add x to C, for any other point y on the same line in direction d,
                # direction d is no longer available for y (because x is on y's line in direction d,
                # and x is in C, so y cannot use direction d)
                
                # Find all points on the same line in direction d
                # We can iterate through all points and check if they're on the same line
                # But that's O(2401) per exclusion, which is 2401 * 2401 = ~5.7M ops total
                # That's fine
                
                for j in range(total_points):
                    if j == i or in_C[j]:
                        continue
                    if line_ids[d][j] == lid:
                        # Point j loses direction d as an option
                        current_avail[j] -= 1
                        heapq.heappush(heap, (current_avail[j], j))
                
                placed = True
                break
        
        if not placed:
            # Should not happen since ac > 0, but just in case
            continue
    
    # Build the result: N = all points not in C
    points = []
    for i in range(total_points):
        if not in_C[i]:
            xa, xb, ya, yb = idx_to_point(i)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: return all points
        for i in range(total_points):
            xa, xb, ya, yb = idx_to_point(i)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
