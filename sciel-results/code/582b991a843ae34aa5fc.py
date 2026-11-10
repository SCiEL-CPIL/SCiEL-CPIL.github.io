import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k_const = 3  # alpha^2 = 3
    start_time = time.time()
    deadline = start_time + 580.0

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # Precompute for each point and direction, the 48 other points on that line
    # affected[p_idx, d_idx, 0..47] = indices of other points
    affected = np.zeros((2401, num_dirs, 48), dtype=np.int32)
    for p_idx in range(2401):
        p_pt = idx_to_pt(p_idx)
        px, py = p_pt[0], p_pt[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    n_idx = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    affected[p_idx, d_idx, count] = n_idx
                    count += 1

    # For each point, precompute all 2400 other points grouped by direction
    # neighbors[p_idx] = list of (neighbor_idx, direction_idx)
    # But for efficiency, let's precompute for each point the full neighbor list
    # and for each neighbor, which direction it belongs to

    # Alternative: for each point p, for each direction d, we have 48 neighbors
    # A point p is "covered" by direction d if all 48 neighbors in direction d are in S
    # We want: for every p NOT in S, there exists a direction d such that all 48 neighbors of p in dir d are in S

    # Let's think of it as: S must be such that for every p not in S, at least one of the 50 lines through p
    # (in the 50 directions) has all its other 48 points in S.

    # Simulated annealing approach:
    # State: binary array s[2401] where s[i]=1 means point i is in S
    # Objective: maximize |S| = sum(s)
    # Constraint: for every i with s[i]=0, there must exist a direction d such that
    #   all 48 neighbors of i in direction d have s[neighbor]=1

    # Penalty: for each uncovered point (s[i]=0 and no direction has all 48 neighbors in S), add penalty
    # We want to minimize penalty while maximizing |S|

    # Let's use a different formulation:
    # For each point i, define cover_count[i] = number of directions d where all 48 neighbors of i in dir d are in S
    # A point i is valid if s[i]=1 (in S, no constraint) or cover_count[i] >= 1 (not in S but covered)
    # Invalid if s[i]=0 and cover_count[i] == 0

    # Simulated annealing:
    # Start with a random S, then flip points in/out of S, accepting based on energy change
    # Energy = -|S| + penalty * (number of invalid points)

    # For efficiency, maintain:
    # - s: binary array
    # - for each point i and direction d: count of neighbors of i in dir d that are in S
    #   neighbor_count[i, d] = number of the 48 neighbors of i in direction d that are in S
    #   i is covered by direction d if neighbor_count[i, d] == 48

    # When we flip point j:
    # - For each point i that has j as a neighbor in some direction d, neighbor_count[i, d] changes by +/-1
    # - We need to track which points i are affected and in which direction

    # Precompute: for each point j, for each direction d, the 48 neighbors
    # But we also need the reverse: for each point j, which (i, d) pairs have j as a neighbor of i in direction d
    # Since the line structure is symmetric, if j is a neighbor of i in direction d, then i is a neighbor of j in direction d
    # (same line, same direction)

    # So: for each point j, for each direction d, the 48 points that have j as a neighbor in direction d
    # are exactly the other 48 points on the line through j in direction d
    # which is affected[j, d, :]

    # So when we flip j, for each direction d, we update neighbor_count[affected[j, d, :], d] by +/-1

    # Let's implement this:

    np.random.seed(42)

    # Initial S: try a structured seed
    # Use the unital-based approach as seed
    def make_seed_unital_yblock(y_block_set):
        S = set()
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k_const) % p == 0
                        in_y_block = (ya, yb) in y_block_set
                        if in_unital and not in_y_block:
                            S.add(pt_to_idx([[xa, xb], [ya, yb]]))
        return S

    y_sq = set()
    for u in range(p):
        for v in range(p):
            y_sq.add((u * u % p, v))

    S_init = make_seed_unital_yblock(y_sq)

    # Convert to binary array
    s = np.zeros(2401, dtype=np.int8)
    for idx in S_init:
        s[idx] = 1

    # Initialize neighbor_count[i, d] for all i, d
    # neighbor_count[i, d] = number of the 48 neighbors of i in direction d that are in S
    neighbor_count = np.zeros((2401, num_dirs), dtype=np.int32)

    # For each point i in S, for each direction d, increment neighbor_count for its neighbors
    # Actually, neighbor_count[i, d] counts neighbors of i in dir d that are in S
    # So for each j in S, for each direction d, j is a neighbor of affected[j, d, :] in direction d
    # So neighbor_count[affected[j, d, :], d] += 1 for each j in S

    for j in S_init:
        for d in range(num_dirs):
            neighbor_count[affected[j, d, :], d] += 1

    # Compute cover_count[i] = number of directions d where neighbor_count[i, d] == 48
    def compute_cover_count(neighbor_count):
        cover_count = np.sum(neighbor_count == 48, axis=1)
        return cover_count

    cover_count = compute_cover_count(neighbor_count)

    # A point i is invalid if s[i] == 0 and cover_count[i] == 0
    def compute_invalid(s, cover_count):
        invalid = (s == 0) & (cover_count == 0)
        return invalid

    invalid = compute_invalid(s, cover_count)
    num_invalid = np.sum(invalid)

    # Energy: we want to minimize num_invalid and maximize |S|
    # Let's use energy = num_invalid * 1000 - |S|
    # But for SA, let's just track num_invalid and |S| separately

    best_S = s.copy()
    best_size = np.sum(s)
    best_invalid = num_invalid

    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    steps = 0

    while time.time() < deadline - 10:
        steps += 1

        # Pick a random point to flip
        j = np.random.randint(2401)

        # Compute the effect of flipping j
        # If s[j] == 1, we remove j from S
        # If s[j] == 0, we add j to S

        delta_size = -1 if s[j] == 1 else 1

        # Compute change in invalid count
        # When we flip j, neighbor_count changes for all (i, d) where j is a neighbor of i in dir d
        # i.e., for each d, i in affected[j, d, :]

        # Let's compute the new cover_count for affected points
        # For each direction d, the points affected[j, d, :] have their neighbor_count[i, d] changed by delta_size

        # We need to check if any of these points become newly invalid or newly valid

        # Current state for each affected point i in direction d:
        # old_count = neighbor_count[i, d]
        # new_count = old_count + delta_size
        # old_covered = (old_count == 48)
        # new_covered = (new_count == 48)
        # If s[i] == 0:
        #   old_invalid = (not old_covered)  # but this is for all directions, not just d
        #   Actually, we need the full cover_count[i], not just for direction d

        # This is complex. Let's compute the change in cover_count for each affected point
        # cover_count[i] = sum_d (neighbor_count[i, d] == 48)
        # When we change neighbor_count[i, d] by delta_size, cover_count[i] changes by:
        #   if old_count == 48 and new_count != 48: -1
        #   if old_count != 48 and new_count == 48: +1
        #   else: 0

        delta_cover = np.zeros(2401, dtype=np.int32)
        for d in range(num_dirs):
            affected_pts = affected[j, d, :]
            old_counts = neighbor_count[affected_pts, d]
            new_counts = old_counts + delta_size
            # Change in cover_count for each affected point
            delta = np.zeros(len(affected_pts), dtype=np.int32)
            delta[(old_counts == 48) & (new_counts != 48)] = -1
            delta[(old_counts != 48) & (new_counts == 48)] = 1
            # Scatter add to delta_cover
            for k, pt in enumerate(affected_pts):
                delta_cover[pt] += delta[k]

        # Now compute the change in invalid count
        # A point i is invalid if s[i] == 0 and cover_count[i] == 0
        # After flip:
        #   new_s[i] = s[i] + delta_size if i == j else s[i]
        #   new_cover[i] = cover_count[i] + delta_cover[i]

        # For i != j:
        #   old_invalid = (s[i] == 0) & (cover_count[i] == 0)
        #   new_invalid = (s[i] == 0) & (new_cover[i] == 0)
        #   delta_invalid[i] = new_invalid - old_invalid

        # For i == j:
        #   old_invalid = (s[j] == 0) & (cover_count[j] == 0)
        #   new_s[j] = s[j] + delta_size
        #   new_cover[j] = cover_count[j] + delta_cover[j]
        #   new_invalid = (new_s[j] == 0) & (new_cover[j] == 0)
        #   delta_invalid[j] = new_invalid - old_invalid

        # Compute delta_invalid for all affected points
        # Affected points are j and all points in affected[j, d, :] for all d
        # But delta_cover is 0 for most points, so we only need to check points with delta_cover != 0 and j

        affected_set = set([j])
        for d in range(num_dirs):
            affected_set.update(affected[j, d, :].tolist())

        delta_invalid_total = 0
        for i in affected_set:
            old_inv = (s[i] == 0) and (cover_count[i] == 0)
            new_s_i = s[i] + (delta_size if i == j else 0)
            new_cover_i = cover_count[i] + delta_cover[i]
            new_inv = (new_s_i == 0) and (new_cover_i == 0)
            delta_invalid_total += int(new_inv) - int(old_inv)

        # Energy change: we want to minimize invalid and maximize size
        # Let's use energy = invalid * 100 - size
        # delta_energy = delta_invalid * 100 - delta_size
        delta_energy = delta_invalid_total * 100 - delta_size

        # Accept if delta_energy < 0 or with probability exp(-delta_energy / T)
        accept = False
        if delta_energy < 0:
            accept = True
        else:
            if T > T_min:
                r = np.random.random()
                if r < np.exp(-delta_energy / T):
                    accept = True

        if accept:
            # Apply the flip
            s[j] += delta_size
            for d in range(num_dirs):
                neighbor_count[affected[j, d, :], d] += delta_size
            cover_count += delta_cover

            new_size = np.sum(s)
            new_invalid = num_invalid + delta_invalid_total
            num_invalid = new_invalid

            if new_invalid == 0 and new_size > best_size:
                best_S = s.copy()
                best_size = new_size
                best_invalid = 0

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Use the best valid S found
    s_final = best_S

    # Build output: points NOT in S (the Nikodym set N)
    points = []
    for i in range(2401):
        if s_final[i] == 0:
            points.append(idx_to_pt(i))

    return np.array(points, dtype=np.int64)
