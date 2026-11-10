# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49 (p=7).
    Uses greedy construction with incremental validation and random restarts.
    """
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 590.0  # Safety margin for 630s limit

    # Field arithmetic setup: F_49 = F_7[alpha], alpha^2 = 3
    k = 3

    # Helper functions for field arithmetic
    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
        
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
        
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute lines
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # Generate lines for v = (0, 1) (vertical lines)
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    pts.append(idx)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(len(lines) - 1)
                
    # Generate lines for v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    pts = []
                    for ta in range(p):
                        for tb in range(p):
                            t = (ta, tb)
                            x = t
                            y = add((ba, bb), mul(m, t))
                            idx = get_idx(x[0], x[1], y[0], y[1])
                            pts.append(idx)
                    lines.append(pts)
                    for pt in pts:
                        line_membership[pt].append(len(lines) - 1)

    num_lines = len(lines)
    
    # Precompute affected points for each point (union of points on all lines through it)
    # For efficiency, we'll compute this on the fly but cache the line membership

    # State variables
    in_S = np.zeros(total_points, dtype=bool)
    line_count = np.zeros(num_lines, dtype=np.int32)
    zero_count_lines = np.zeros(total_points, dtype=np.int32)
    one_count_lines = np.zeros(total_points, dtype=np.int32)

    def init_state(seed_S=None):
        nonlocal in_S, line_count, zero_count_lines, one_count_lines
        in_S.fill(False)
        line_count.fill(0)
        zero_count_lines.fill(0)
        one_count_lines.fill(0)
        
        if seed_S is not None:
            for x in seed_S:
                in_S[x] = True
                for l_idx in line_membership[x]:
                    line_count[l_idx] += 1
        
        for x in range(total_points):
            zc = 0
            oc = 0
            for l_idx in line_membership[x]:
                c = line_count[l_idx]
                if c == 0:
                    zc += 1
                elif c == 1:
                    oc += 1
            zero_count_lines[x] = zc
            one_count_lines[x] = oc

    def try_add_point(x):
        """Try to add point x to S. Returns True if valid."""
        if in_S[x]:
            return False
        
        lines_through_x = line_membership[x]
        
        # Save old state for affected points
        old_line_counts = {}
        for l_idx in lines_through_x:
            old_line_counts[l_idx] = line_count[l_idx]
        
        # Find affected points
        affected = set()
        for l_idx in lines_through_x:
            for pt in lines[l_idx]:
                affected.add(pt)
        
        old_zero = {}
        old_one = {}
        for pt in affected:
            old_zero[pt] = zero_count_lines[pt]
            old_one[pt] = one_count_lines[pt]
        
        # Apply: add x to S
        in_S[x] = True
        for l_idx in lines_through_x:
            line_count[l_idx] += 1
        
        # Update zero/one counts for affected points
        for pt in affected:
            zc = 0
            oc = 0
            for l_idx in line_membership[pt]:
                c = line_count[l_idx]
                if c == 0:
                    zc += 1
                elif c == 1:
                    oc += 1
            zero_count_lines[pt] = zc
            one_count_lines[pt] = oc
        
        # Check validity of affected points
        valid = True
        for pt in affected:
            if in_S[pt]:
                if one_count_lines[pt] < 1:
                    valid = False
                    break
            else:
                if zero_count_lines[pt] < 1:
                    valid = False
                    break
        
        if not valid:
            # Rollback
            in_S[x] = False
            for l_idx in lines_through_x:
                line_count[l_idx] = old_line_counts[l_idx]
            for pt in affected:
                zero_count_lines[pt] = old_zero[pt]
                one_count_lines[pt] = old_one[pt]
        
        return valid

    def is_valid():
        for x in range(total_points):
            if in_S[x]:
                if one_count_lines[x] < 1:
                    return False
            else:
                if zero_count_lines[x] < 1:
                    return False
        return True

    # Generate a seed using the unital-based approach
    def generate_seed():
        squares = {0, 1, 2, 4}
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if (2*xa + ya*ya - yb*yb*k) % p == 0:
                            if ya not in squares:
                                idx = get_idx(xa, xb, ya, yb)
                                S.add(idx)
        return S

    # Greedy construction from a given seed
    def greedy_construct(seed_S, max_iter=10000):
        nonlocal in_S, line_count, zero_count_lines, one_count_lines
        init_state(seed_S)
        
        if not is_valid():
            # Reset to empty if seed is invalid
            init_state(None)
        
        S_size = np.sum(in_S)
        best_size = S_size
        best_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Get candidates: points not in S
            candidates = np.where(~in_S)[0]
            np.random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if try_add_point(x):
                    improved = True
                    S_size += 1
                    if S_size > best_size:
                        best_size = S_size
                        best_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())
        
        # Restore best state
        in_S, line_count, zero_count_lines, one_count_lines = best_state
        return best_size

    # Random restart with perturbation
    def perturbed_seed(base_seed, perturbation_rate=0.1):
        S = set(base_seed)
        for x in list(S):
            if random.random() < perturbation_rate:
                S.remove(x)
        # Add some random points
        for _ in range(int(perturbation_rate * len(base_seed))):
            x = random.randint(0, total_points - 1)
            S.add(x)
        return S

    # Main loop: try multiple strategies
    best_S_size = 0
    best_state = None

    # Strategy 1: Unital seed + greedy
    seed1 = generate_seed()
    size1 = greedy_construct(seed1)
    if size1 > best_S_size:
        best_S_size = size1
        best_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())

    if time.time() > deadline:
        in_S, line_count, zero_count_lines, one_count_lines = best_state
    else:
        # Strategy 2: Empty seed + greedy
        size2 = greedy_construct(set())
        if size2 > best_S_size:
            best_S_size = size2
            best_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())

    if time.time() > deadline:
        in_S, line_count, zero_count_lines, one_count_lines = best_state
    else:
        # Strategy 3: Multiple random restarts with perturbation
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            pert_rate = random.uniform(0.05, 0.3)
            pert_seed = perturbed_seed(seed1, pert_rate)
            size_r = greedy_construct(pert_seed, max_iter=5000)
            if size_r > best_S_size:
                best_S_size = size_r
                best_state = (in_S.copy(), line_count.copy(), zero_count_lines.copy(), one_count_lines.copy())

    # Restore best state
    if best_state is not None:
        in_S, line_count, zero_count_lines, one_count_lines = best_state
    else:
        init_state(None)

    # Return N = F_q^2 \ S
    points = []
    for x in range(total_points):
        if not in_S[x]:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
