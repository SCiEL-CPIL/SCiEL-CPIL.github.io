# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    Q2 = q * q  # 2401
    k = 3  # alpha^2 = 3 in F_7

    def get_idx(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    def get_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb

    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua * va + ub * vb * k) % p, (ua * vb + ub * va) % p)

    def add(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)

    lines = []
    line_membership = [[] for _ in range(Q2)]

    for xa in range(p):
        for xb in range(p):
            pts = np.array([get_idx(xa, xb, ya, yb) for ya in range(p) for yb in range(p)], dtype=np.int32)
            li = len(lines)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(li)

    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    b = (ba, bb)
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, b)
                            pts.append(get_idx(xa, xb, y[0], y[1]))
                    pts = np.array(pts, dtype=np.int32)
                    li = len(lines)
                    lines.append(pts)
                    for pt in pts:
                        line_membership[pt].append(li)

    num_lines = len(lines)
    line_membership_arr = [np.array(lm, dtype=np.int32) for lm in line_membership]
    
    # Precompute affected points for each point y
    # affected[y] is a numpy array of unique points on lines through y
    affected_points = [np.unique(np.concatenate([lines[li] for li in line_membership_arr[y]])) for y in range(Q2)]

    # Initialize S
    # Start with a random small S or empty
    # Let's start with empty S to allow SA to build it up
    in_S = np.zeros(Q2, dtype=bool)
    
    line_count = np.zeros(num_lines, dtype=np.int32)
    witness_count = np.zeros(Q2, dtype=np.int32)
    zero_count = np.zeros(Q2, dtype=np.int32)
    
    # Initial state: S is empty
    # All lines have count 0
    # All points in N (all points) have zero_count = 50 (all lines through them are zero lines)
    for y in range(Q2):
        zero_count[y] = len(line_membership_arr[y]) # 50
        
    total_penalty = 0 # All N points are covered (zero_count >= 1), no S points.
    current_score = 0 # |S| = 0

    deadline = time.time() + 580
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    lambda_pen = 1000.0
    
    best_S = in_S.copy()
    best_score = 0

    while time.time() < deadline and T > min_T:
        # Pick a random point y
        y = random.randint(0, Q2 - 1)
        
        if in_S[y]:
            # Try to remove y
            # Delta |S| = -1
            delta_size = -1
            # Calculate penalty change
            A = affected_points[y]
            
            # Store old counts for A
            old_wit = witness_count[A].copy()
            old_zero = zero_count[A].copy()
            old_in_S = in_S[A].copy()
            
            # Calculate old penalty for A
            # z in A is invalid if:
            # (z in S and old_wit[z] == 0) OR (z not in S and old_zero[z] == 0)
            old_pen_A = 0
            for i, z in enumerate(A):
                if old_in_S[i]:
                    if old_wit[i] == 0:
                        old_pen_A += 1
                else:
                    if old_zero[i] == 0:
                        old_pen_A += 1

            # Apply removal
            in_S[y] = False
            for li in line_membership_arr[y]:
                line_count[li] -= 1
                lc = line_count[li]
                if lc == 0:
                    # Line becomes zero line
                    # All N points on L gain a zero line
                    for z in lines[li]:
                        if not in_S[z]:
                            zero_count[z] += 1
                elif lc == 1:
                    # Line becomes witness line for the remaining S point
                    # Find the S point
                    # We can just iterate
                    for z in lines[li]:
                        if in_S[z]:
                            witness_count[z] += 1
                            break

            # Calculate new penalty for A
            new_pen_A = 0
            for i, z in enumerate(A):
                if in_S[z]:
                    if witness_count[z] == 0:
                        new_pen_A += 1
                else:
                    if zero_count[z] == 0:
                        new_pen_A += 1

            delta_pen = new_pen_A - old_pen_A
            delta_E = -delta_size + lambda_pen * delta_pen
            
            # Accept?
            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                # Accept
                current_score += delta_size
                total_penalty += delta_pen
                if total_penalty == 0 and current_score > best_score:
                    best_score = current_score
                    best_S = in_S.copy()
            else:
                # Reject: Rollback
                in_S[y] = True
                for li in line_membership_arr[y]:
                    line_count[li] += 1
                    lc = line_count[li]
                    if lc == 1:
                        # Line was 0, now 1. y is the S point.
                        # Zero lines for N points on L are lost
                        for z in lines[li]:
                            if not in_S[z]:
                                zero_count[z] -= 1
                        # y gains a witness line
                        witness_count[y] += 1
                    elif lc == 2:
                        # Line was 1, now 2. Witness line lost for the other S point
                        for z in lines[li]:
                            if in_S[z] and z != y:
                                witness_count[z] -= 1
                                break

        else:
            # Try to add y
            delta_size = 1
            A = affected_points[y]
            
            old_wit = witness_count[A].copy()
            old_zero = zero_count[A].copy()
            old_in_S = in_S[A].copy()
            
            old_pen_A = 0
            for i, z in enumerate(A):
                if old_in_S[i]:
                    if old_wit[i] == 0:
                        old_pen_A += 1
                else:
                    if old_zero[i] == 0:
                        old_pen_A += 1

            # Apply addition
            in_S[y] = True
            for li in line_membership_arr[y]:
                line_count[li] += 1
                lc = line_count[li]
                if lc == 1:
                    # Line was 0, now 1. y is the S point.
                    # Zero lines for N points on L are lost
                    for z in lines[li]:
                        if not in_S[z]:
                            zero_count[z] -= 1
                    # y gains a witness line
                    witness_count[y] += 1
                elif lc == 2:
                    # Line was 1, now 2. Witness line lost for the other S point
                    for z in lines[li]:
                        if in_S[z] and z != y:
                            witness_count[z] -= 1
                            break

            new_pen_A = 0
            for i, z in enumerate(A):
                if in_S[z]:
                    if witness_count[z] == 0:
                        new_pen_A += 1
                else:
                    if zero_count[z] == 0:
                        new_pen_A += 1

            delta_pen = new_pen_A - old_pen_A
            delta_E = -delta_size + lambda_pen * delta_pen

            if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                current_score += delta_size
                total_penalty += delta_pen
                if total_penalty == 0 and current_score > best_score:
                    best_score = current_score
                    best_S = in_S.copy()
            else:
                # Reject: Rollback
                in_S[y] = False
                for li in line_membership_arr[y]:
                    line_count[li] -= 1
                    lc = line_count[li]
                    if lc == 0:
                        for z in lines[li]:
                            if not in_S[z]:
                                zero_count[z] += 1
                    elif lc == 1:
                        for z in lines[li]:
                            if in_S[z]:
                                witness_count[z] += 1
                                break

        T *= cooling

    # Reconstruct N from best_S
    N_points = []
    for i in range(Q2):
        if not best_S[i]:
            xa, xb, ya, yb = get_coords(i)
            N_points.append([[xa, xb], [ya, yb]])

    if not N_points:
        # Fallback
        N_points = [[[0, 0], [0, 0]]]

    return np.array(N_points, dtype=np.int64)

# EVOLVE-BLOCK-END
