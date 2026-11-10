# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using the Hermitian unital structure
    and greedy set cover to minimize N.
    
    Strategy:
    - Use Hermitian unital U: Tr(x) + Norm(y) = 0 (343 points)
    - Each point u in U has a unique tangent line T_u meeting U only at u
    - Select minimal R subset U such that tangent lines of R cover all of U
    - N = F_49^2 \\ (U \\ R) = (F_49^2 \\ U) ∪ R
    - Complement S = U \\ R, size = 343 - |R|
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin before 630s limit

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 (smallest quadratic non-residue mod 7)

    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def trace(a):
        return (2 * a[0]) % p

    def norm(a):
        return (a[0] * a[0] - a[1] * a[1] * k) % p

    # Generate all points and identify unital points
    # Point index: ((xa * p + xb) * p + ya) * p + yb
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    unital_indices = set()
    for i, (x, y) in enumerate(all_points):
        if (trace(x) + norm(y)) % p == 0:
            unital_indices.add(i)

    unital_list = sorted(unital_indices)
    n_unital = len(unital_list)  # Should be 343

    # Generate all lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Total q+1 = 50 directions

    lines = []
    point_to_lines = {i: [] for i in range(q * q)}

    # Direction (1, m): lines are y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
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
                    lines.append(line_points)
                    line_id = len(lines) - 1
                    for idx in line_points:
                        point_to_lines[idx].append(line_id)

    # Direction (0, 1): lines are x = c
    for ca in range(p):
        for cb in range(p):
            c = (ca, cb)
            line_points = []
            for ya in range(p):
                for yb in range(p):
                    idx = ((ca * p + cb) * p + ya) * p + yb
                    line_points.append(idx)
            lines.append(line_points)
            line_id = len(lines) - 1
            for idx in line_points:
                point_to_lines[idx].append(line_id)

    # For each u in U, find its tangent line (line meeting U only at u)
    tangent_line_for_u = {}
    for u_idx in unital_list:
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

        if len(candidate_tangents) >= 1:
            tangent_line_for_u[u_idx] = candidate_tangents[0]
        else:
            tangent_line_for_u[u_idx] = -1

    # Set Cover: Universe = U, Sets = tangent lines of each u in U
    # For each u in U, the set is {v in U : v lies on T_u}
    # We need to cover all of U with these sets

    # Precompute coverage sets
    coverage_sets = {}
    for u_idx in unital_list:
        line_id = tangent_line_for_u[u_idx]
        if line_id == -1:
            coverage_sets[u_idx] = set()
        else:
            line_set = lines[line_id]
            cov = set()
            for p_idx in line_set:
                if p_idx in unital_indices:
                    cov.add(p_idx)
            coverage_sets[u_idx] = cov

    # Greedy set cover
    uncovered = set(unital_list)
    R = []

    while uncovered and time.time() < deadline:
        best_u = None
        best_coverage = 0
        for u_idx in unital_list:
            if u_idx in [r for r in R]:
                continue
            cov = coverage_sets[u_idx]
            new_cov = len(cov & uncovered)
            if new_cov > best_coverage:
                best_coverage = new_cov
                best_u = u_idx

        if best_u is None or best_coverage == 0:
            break

        R.append(best_u)
        uncovered -= coverage_sets[best_u]

    # If greedy didn't cover everything, add remaining points to R
    if uncovered:
        for u_idx in uncovered:
            R.append(u_idx)

    # Build the Nikodym set N = (F_49^2 \ U) ∪ R
    R_set = set(R)
    points = []
    for i in range(q * q):
        if i in unital_indices:
            if i in R_set:
                x, y = all_points[i]
                points.append([[x[0], x[1]], [y[0], y[1]]])
        else:
            x, y = all_points[i]
            points.append([[x[0], x[1]], [y[0], y[1]]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
