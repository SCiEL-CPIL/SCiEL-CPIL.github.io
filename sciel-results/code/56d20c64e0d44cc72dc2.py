# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses Hermitian Unital structure with simulated annealing to optimize
    which points to remove from the unital, maximizing the complement size.
    """
    start_time = time.time()
    deadline = start_time + 500.0  # Safety margin

    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Field operations
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p

    # Generate all points in F_49^2
    # Index: idx = ((xa * p + xb) * p + ya) * p + yb
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    # Identify points on the Hermitian Unital U
    unital_points = []
    unital_indices = set()
    for i, (x, y) in enumerate(all_points):
        if (trace(x) + norm(y)) % p == 0:
            unital_points.append(i)
            unital_indices.add(i)

    # Precompute lines
    lines = []
    point_to_lines = {i: [] for i in range(q * q)}

    # Directions: (1, m) for m in F_49, and (0, 1).
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)

    seen_lines = set()

    for vx, vy in directions:
        if vx[0] == 1:
            m = vy
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = ((xa * p + xb) * p + y[0]) * p + y[1]
                            line_points.append(idx)

                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        for idx in line_points:
                            point_to_lines[idx].append(len(lines) - 1)
        else:
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_points = []
                    for ya in range(p):
                        for yb in range(p):
                            idx = ((ca * p + cb) * p + ya) * p + yb
                            line_points.append(idx)

                    line_tuple = tuple(sorted(line_points))
                    if line_tuple not in seen_lines:
                        seen_lines.add(line_tuple)
                        line_set = set(line_points)
                        lines.append(line_set)
                        for idx in line_points:
                            point_to_lines[idx].append(len(lines) - 1)

    # For each u in U, find the tangent line.
    tangent_line_for_u = {}
    for u_idx in unital_points:
        candidate_tangents = []
        for line_id in point_to_lines[u_idx]:
            line_set = lines[line_id]
            inter_count = 0
            for p_idx in line_set:
                if p_idx in unital_indices:
                    inter_count += 1
                    if inter_count > 1:
                        break
            if inter_count == 1:
                candidate_tangents.append(line_id)

        if len(candidate_tangents) == 1:
            tangent_line_for_u[u_idx] = candidate_tangents[0]
        elif candidate_tangents:
            tangent_line_for_u[u_idx] = candidate_tangents[0]
        else:
            tangent_line_for_u[u_idx] = -1

    # Universe: Points in N = F^2 \ U
    universe = [i for i in range(q * q) if i not in unital_indices]
    universe_set = set(universe)

    # For each u in U, the set of universe points covered by its tangent line
    sets_for_u = {}
    for u_idx in unital_points:
        line_id = tangent_line_for_u[u_idx]
        if line_id == -1:
            continue
        line_set = lines[line_id]
        s_u = set()
        for p_idx in line_set:
            if p_idx in universe_set:
                s_u.add(p_idx)
        sets_for_u[u_idx] = s_u

    # Simulated Annealing to find optimal subset R of unital_points to remove
    # Objective: maximize |union of sets_for_u[u] for u in R|
    # This maximizes the complement size

    # Initialize with greedy solution
    uncovered = set(universe)
    R_greedy = []
    while uncovered and time.time() < deadline - 100:
        best_u = None
        best_cover = 0
        for u_idx, s_u in sets_for_u.items():
            if u_idx in R_greedy:
                continue
            cover = len(uncovered & s_u)
            if cover > best_cover:
                best_cover = cover
                best_u = u_idx
        if best_u is None:
            break
        R_greedy.append(best_u)
        uncovered -= sets_for_u[best_u]

    # Convert greedy solution to a set for SA
    R_set = set(R_greedy)
    best_R = set(R_greedy)
    best_score = len(universe_set - uncovered)  # number of universe points covered

    # Simulated Annealing
    current_R = set(R_greedy)
    current_score = best_score
    temperature = 50.0
    cooling_rate = 0.995
    min_temperature = 0.1

    # Precompute for faster evaluation
    # For each u, store the set as a frozenset for hashing
    sets_frozen = {u: frozenset(s) for u, s in sets_for_u.items()}

    def evaluate_R(R_set):
        """Compute the number of universe points covered by tangent lines of points in R_set"""
        covered = set()
        for u in R_set:
            if u in sets_frozen:
                covered.update(sets_frozen[u])
        return len(covered & universe_set)

    # SA loop
    sa_start = time.time()
    sa_deadline = deadline - 50  # Leave 50s for finalization

    while time.time() < sa_deadline:
        if temperature < min_temperature:
            break

        # Pick a random move: add or remove a random unital point from R
        if random.random() < 0.5 and len(current_R) < len(unital_points):
            # Remove a random point from R
            if current_R:
                u_remove = random.choice(list(current_R))
                new_R = current_R - {u_remove}
            else:
                continue
        else:
            # Add a random point to R
            candidates = [u for u in unital_points if u not in current_R]
            if not candidates:
                continue
            u_add = random.choice(candidates)
            new_R = current_R | {u_add}

        new_score = evaluate_R(new_R)
        delta = new_score - current_score

        if delta >= 0 or random.random() < np.exp(delta / temperature):
            current_R = new_R
            current_score = new_score

            if current_score > best_score:
                best_score = current_score
                best_R = set(current_R)

        temperature *= cooling_rate

    # Final result: S = U \ best_R, N = F^2 \ S
    S_set = set(unital_points) - best_R
    N = [i for i in range(q * q) if i not in S_set]

    # Convert indices to points
    points = []
    for idx in N:
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
