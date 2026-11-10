# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using constraint propagation.
    Start with a candidate excluded set, then iteratively remove excluded points
    that have no valid witness line (all other points on every line through them
    are also excluded). This is much faster than point-by-point greedy.
    """
    start_time = time.time()
    deadline = start_time + 570
    
    p = 7
    n = p * p * p * p  # 2401
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
    
    # All points indexed: idx = xa*343 + xb*49 + ya*7 + yb
    def idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb
    
    def decode(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb
    
    # Precompute all lines as numpy arrays of point indices
    # Vertical lines: 49 lines, each with 49 points
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            line = np.array([idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)], dtype=np.int32)
            vert_lines.append(line)
    vert_lines = np.array(vert_lines, dtype=np.int32)  # shape (49, 49)
    
    # Slope lines: 48 slopes, each with 49 lines of 49 points
    # For slope m=(ma,mb), line through point (xa,xb,ya,yb) has intercept c = ya - m*xa
    slope_lines_all = {}
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines = []
            for xa in range(p):
                for xb in range(p):
                    mx = mul_f49((ma, mb), (xa, xb))
                    for ya in range(p):
                        for yb in range(p):
                            ca = (ya - mx[0]) % p
                            cb = (yb - mx[1]) % p
                            # This point is on the line with intercept (ca, cb)
                            pass
            # Better: for each intercept (ca, cb), collect all points
            intercept_lines = defaultdict(list)
            for xa in range(p):
                for xb in range(p):
                    mx = mul_f49((ma, mb), (xa, xb))
                    for ya in range(p):
                        for yb in range(p):
                            ca = (ya - mx[0]) % p
                            cb = (yb - mx[1]) % p
                            intercept_lines[(ca, cb)].append(idx(xa, xb, ya, yb))
            # Convert to array
            for ca in range(p):
                for cb in range(p):
                    line = np.array(intercept_lines.get((ca, cb), []), dtype=np.int32)
                    if len(line) == p:
                        lines.append(line)
            slope_lines_all[(ma, mb)] = np.array(lines, dtype=np.int32)  # shape (49, 49)
    
    # All lines combined: vertical (49) + slope (48*49 = 2352) = 2401 lines
    # Each line has 49 points
    all_lines = [vert_lines]
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            all_lines.append(slope_lines_all[(ma, mb)])
    
    # For each point, precompute which lines contain it
    # point_lines[point_idx] = list of line indices
    point_lines = defaultdict(list)
    line_id = 0
    line_arrays = []
    for xa in range(p):
        for xb in range(p):
            line = np.array([idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)], dtype=np.int32)
            line_arrays.append(line)
            for pt in line:
                point_lines[pt].append(line_id)
            line_id += 1
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            for line in slope_lines_all[(ma, mb)]:
                line_arrays.append(line)
                for pt in line:
                    point_lines[pt].append(line_id)
                line_id += 1
    
    total_lines = line_id  # should be 2401
    
    # Initial excluded set: points on Hermitian unital with y_a in QR
    # Unital: 2*xa + ya^2 - 3*yb^2 = 0 in F_7
    # y_a in QR means ya is a quadratic residue mod 7: {0, 1, 2, 4}
    excluded = np.zeros(n, dtype=bool)
    for i in range(n):
        xa, xb, ya, yb = decode(i)
        if (2*xa + ya*ya - 3*yb*yb) % p == 0:
            # ya in QR: ya in {0,1,2,4}
            if ya in (0, 1, 2, 4):
                excluded[i] = True
    
    # Constraint propagation: iteratively remove excluded points that have no witness
    # A point i (excluded) has a witness line L if all other points on L are NOT excluded
    # If no such line exists, point i cannot be excluded -> must be in N
    
    def propagate(excluded):
        """Remove excluded points that have no valid witness line. Returns new excluded array."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not excluded[i]:
                    continue
                # Check if point i has a witness line
                found = False
                for lid in point_lines[i]:
                    line = line_arrays[lid]
                    # Check if all other points on this line are NOT excluded
                    # i.e., excluded[line] has False for all j != i
                    # Count excluded points on this line
                    cnt = np.sum(excluded[line])
                    if cnt == 1:  # only point i is excluded on this line
                        found = True
                        break
                if not found:
                    excluded[i] = False
                    changed = True
        return excluded
    
    # Run propagation
    if time.time() < deadline - 30:
        excluded = propagate(excluded)
    
    # Try to add more excluded points: greedily try to exclude additional points
    # that are on the unital but not yet excluded, using propagation to check validity
    if time.time() < deadline - 30:
        # Get all unital points not yet excluded
        unital_candidates = []
        for i in range(n):
            if excluded[i]:
                continue
            xa, xb, ya, yb = decode(i)
            if (2*xa + ya*ya - 3*yb*yb) % p == 0:
                unital_candidates.append(i)
        
        # Try adding them in random order, using propagation to check
        np.random.seed(42)
        np.random.shuffle(unital_candidates)
        
        for i in unital_candidates:
            if time.time() > deadline - 20:
                break
            if excluded[i]:
                continue
            # Try excluding point i
            old_excluded = excluded.copy()
            excluded[i] = True
            # Propagate: if this causes any point to lose its witness, revert
            test_excluded = excluded.copy()
            # Quick check: does point i itself have a witness?
            found = False
            for lid in point_lines[i]:
                line = line_arrays[lid]
                cnt = np.sum(test_excluded[line])
                if cnt == 1:
                    found = True
                    break
            if not found:
                excluded[i] = False
                continue
            # Check if any other excluded point lost its witness
            valid = True
            for j in range(n):
                if not test_excluded[j]:
                    continue
                if j == i:
                    continue
                has_w = False
                for lid in point_lines[j]:
                    line = line_arrays[lid]
                    cnt = np.sum(test_excluded[line])
                    if cnt == 1:
                        has_w = True
                        break
                if not has_w:
                    valid = False
                    break
            if not valid:
                excluded[i] = False
    
    # Final propagation to clean up
    if time.time() < deadline - 10:
        excluded = propagate(excluded)
    
    # Build result
    in_N = ~excluded
    pts = []
    for i in range(n):
        if in_N[i]:
            xa, xb, ya, yb = decode(i)
            pts.append([xa, xb, ya, yb])
    
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
