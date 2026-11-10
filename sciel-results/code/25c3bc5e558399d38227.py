# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    total_points = p * p * p * p  # 2401
    deadline = time.time() + 580  # safety margin before 600s limit

    # Field arithmetic helpers (mod 7)
    # Point representation: (xa, xb, ya, yb) -> flat index = xa*343 + xb*49 + ya*7 + yb
    def point_to_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_point(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return (xa, xb, ya, yb)

    # Precompute all points
    all_points = np.zeros((total_points, 4), dtype=np.int64)
    for i in range(total_points):
        xa, xb, ya, yb = idx_to_point(i)
        all_points[i] = [xa, xb, ya, yb]

    # Precompute lines: for each slope m (including vertical), for each intercept c
    # Line with slope m: y = m*x + c, where m, c in F_49
    # Vertical line: x = c
    # Represent lines as (type, param1, param2)
    # type=0: y = m*x + c (m in F_49, c in F_49)
    # type=1: x = c (c in F_49)
    
    # For each point, precompute which lines pass through it
    # A point (xa,xb,ya,yb) corresponds to field element x = xa + xb*alpha, y = ya + yb*alpha
    
    def field_mul(a, b):
        # Multiply two field elements represented as (a0, a1) * (b0, b1)
        # (a0 + a1*alpha)(b0 + b1*alpha) = a0*b0 + a0*b1*alpha + a1*b0*alpha + a1*b1*alpha^2
        # alpha^2 = 3
        # = (a0*b0 + 3*a1*b1) + (a0*b1 + a1*b0)*alpha
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + 3 * a1 * b1) % p, (a0 * b1 + a1 * b0) % p)

    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    # Precompute field multiplication table
    # Represent field elements as indices 0..48: idx = a0*7 + a1
    field_mul_table = np.zeros((q, q), dtype=np.int64)
    field_add_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        a0, a1 = i // 7, i % 7
        for j in range(q):
            b0, b1 = j // 7, j % 7
            # Addition
            field_add_table[i, j] = ((a0 + b0) % p) * 7 + ((a1 + b1) % p)
            # Multiplication
            m0 = (a0 * b0 + 3 * a1 * b1) % p
            m1 = (a0 * b1 + a1 * b0) % p
            field_mul_table[i, j] = m0 * 7 + m1

    # Line representation:
    # For non-vertical lines: y = m*x + c, where m, c are field elements (indices 0..48)
    # For vertical lines: x = c, where c is a field element (index 0..48)
    # Total lines: 49 (vertical) + 49*49 (non-vertical) = 49 + 2401 = 2450 lines
    
    # Precompute: for each line, the set of points on it (as a boolean array)
    # This is 2450 * 2401 booleans = ~5.9M booleans, manageable
    
    # Line indices:
    # 0..48: vertical lines x = c (c = 0..48)
    # 49..2450: non-vertical lines y = m*x + c
    #   line_idx = 49 + m*49 + c
    
    num_lines = q + q * q  # 49 + 2401 = 2450
    line_points = np.zeros((num_lines, total_points), dtype=bool)
    
    # For each point, compute its field coordinates
    # Point i has (xa, xb, ya, yb) -> x_field = xa*7+xb, y_field = ya*7+yb
    x_field = all_points[:, 0] * 7 + all_points[:, 1]
    y_field = all_points[:, 2] * 7 + all_points[:, 3]
    
    # Vertical lines: x = c
    for c in range(q):
        line_idx = c
        mask = (x_field == c)
        line_points[line_idx, mask] = True
    
    # Non-vertical lines: y = m*x + c
    for m in range(q):
        for c in range(q):
            line_idx = q + m * q + c
            # y = m*x + c for all x
            # For each point, check if y_field == m*x_field + c
            # Use vectorized operations
            rhs = field_add_table[field_mul_table[m, x_field], c]
            mask = (y_field == rhs)
            line_points[line_idx, mask] = True

    # Now implement greedy search to maximize |S|
    # S is the excluded set (complement of N)
    # N = all_points \ S must be a Nikodym set
    
    # Validity check for a given S (boolean array of size total_points):
    # For every point x:
    #   If x in S: there exists a line L through x such that L \ {x} is disjoint from S
    #   If x in N: there exists a line L through x such that L is disjoint from S
    
    # For point x on line L:
    #   If x in S: need (L \ {x}) ∩ S = ∅, i.e., S[L] has at most 1 point (x itself)
    #   If x in N: need L ∩ S = ∅, i.e., S[L] is empty
    
    # Precompute: for each point, which lines pass through it
    point_lines = np.zeros((total_points, num_lines), dtype=bool)
    for li in range(num_lines):
        pts_on_line = np.where(line_points[li])[0]
        point_lines[pts_on_line, li] = True

    def is_valid(S):
        """Check if N = complement of S is a valid Nikodym set."""
        # For each point, check if there's a valid line
        # Vectorized: for each point x, check all lines through x
        
        # For x in S: need a line L through x with S[L].sum() <= 1
        # For x in N: need a line L through x with S[L].sum() == 0
        
        # S_on_lines: for each line, how many points of S are on it
        S_on_lines = (line_points & S[None, :]).sum(axis=1)  # shape (num_lines,)
        
        # For each point x, get the lines through it and check
        # point_lines[x] gives boolean mask of lines through x
        # For x in S: need any line through x with S_on_lines[li] <= 1
        # For x in N: need any line through x with S_on_lines[li] == 0
        
        # Vectorized approach:
        # For each point, compute min S_on_lines over lines through it
        # But we need different criteria for S vs N
        
        # Let's do it in chunks to avoid memory issues
        # Actually, let's just loop over points (2401 is small)
        
        for x in range(total_points):
            lines_through_x = np.where(point_lines[x])[0]
            if len(lines_through_x) == 0:
                return False
            s_on_these_lines = S_on_lines[lines_through_x]
            if S[x]:
                # Need a line with S_on_lines <= 1
                if s_on_these_lines.min() > 1:
                    return False
            else:
                # Need a line with S_on_lines == 0
                if s_on_these_lines.min() > 0:
                    return False
        return True

    # Greedy algorithm: start with S = empty, try to add points one by one
    # Use multiple random orderings and keep the best result
    
    best_S = np.zeros(total_points, dtype=bool)
    best_size = 0
    
    # Initial solution: use the Hermitian curve construction as a starting point
    # Points in U with y_a in {3, 5, 6}
    for i in range(total_points):
        xa, xb, ya, yb = idx_to_point(i)
        val = (2 * xa + ya * ya - yb * yb * k) % p
        if val == 0 and ya in {3, 5, 6}:
            best_S[i] = True
    
    best_size = best_S.sum()
    
    # Try to improve by adding points
    # Candidate points: all points not in S
    candidates = np.where(~best_S)[0]
    
    # Try multiple random orderings
    rng = np.random.default_rng(42)
    
    num_trials = 10
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        S = best_S.copy()
        current_size = S.sum()
        
        # Random ordering of candidates
        order = rng.permutation(candidates)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            # Try adding point idx to S
            S[idx] = True
            
            # Check validity
            if is_valid(S):
                current_size += 1
            else:
                S[idx] = False
        
        if current_size > best_size:
            best_size = current_size
            best_S = S.copy()
    
    # Also try removing points from S to see if we can add more elsewhere
    # (This is a local search step)
    for _ in range(3):
        if time.time() > deadline:
            break
        
        S = best_S.copy()
        current_size = S.sum()
        
        # Try removing each point in S and see if we can add more
        s_points = np.where(S)[0]
        for sp in s_points:
            if time.time() > deadline:
                break
            
            S[sp] = False
            if is_valid(S):
                # Now try to add other points
                remaining_candidates = np.where(~S)[0]
                order = rng.permutation(remaining_candidates)
                for idx in order:
                    if time.time() > deadline:
                        break
                    S[idx] = True
                    if is_valid(S):
                        pass
                    else:
                        S[idx] = False
                
                new_size = S.sum()
                if new_size > best_size:
                    best_size = new_size
                    best_S = S.copy()
            else:
                S[sp] = True
    
    # Construct N = complement of S
    N_indices = np.where(~best_S)[0]
    N = all_points[N_indices]
    
    # Format as (k, 2, 2)
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    result[:, 0, 0] = N[:, 0]  # xa
    result[:, 0, 1] = N[:, 1]  # xb
    result[:, 1, 0] = N[:, 2]  # ya
    result[:, 1, 1] = N[:, 3]  # yb
    
    # Reduce mod 7 (already done, but ensure)
    result = result % 7
    
    return result

# EVOLVE-BLOCK-END
