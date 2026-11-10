import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600
    p = 7
    q = 49
    k = 3

    # Precompute field tables
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)

    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            add_tbl[i, j] = ((a0 + b0) % p) * p + ((a1 + b1) % p)
            sub_tbl[i, j] = ((a0 - b0) % p) * p + ((a1 - b1) % p)
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1

    total_points = q * q

    # Construct Unital U
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)

    all_points_set = set(range(total_points))
    outside_U = all_points_set - U
    outside_U_set = outside_U  # alias for clarity

    # Build all lines
    lines = []
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        lines.append(pts)

    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)

    num_lines = len(lines)

    # Precompute line properties
    line_U_intersections = []
    line_outside_coverage = []
    line_point_sets = []
    line_outside_list = []  # list of outside-U points on each line

    for L in lines:
        I = set(pt for pt in L if pt in U)
        cov = set(pt for pt in L if pt in outside_U)
        line_U_intersections.append(I)
        line_outside_coverage.append(cov)
        line_point_sets.append(set(L))
        line_outside_list.append(list(cov))

    # For each outside-U point, which lines cover it?
    point_to_lines = {}
    for li in range(num_lines):
        for pt in line_outside_list[li]:
            if pt not in point_to_lines:
                point_to_lines[pt] = []
            point_to_lines[pt].append(li)

    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        # uncovered: set of outside-U points not yet covered
        uncovered = set(outside_U)
        # line_free_count[li] = number of points on line li still uncovered
        line_free_count = [0] * num_lines
        for li in range(num_lines):
            line_free_count[li] = len(line_outside_list[li])

        # H: set of unital points chosen
        H = set()
        # u_count[u] = number of chosen lines that pass through unital point u
        u_count = {}
        for u in U:
            u_count[u] = 0

        chosen_lines = []
        
        max_iterations = 50000
        iteration = 0
        
        while uncovered and iteration < max_iterations:
            if time.time() - start_time > DEADLINE - 120:
                break
                
            best_line_idx = -1
            best_score = -1.0
            
            noise_scale = 0.05 if random_seed is not None else 0.0
            
            for i in range(num_lines):
                fc = line_free_count[i]
                if fc == 0:
                    continue

                # new_H_cost = number of unital points on line i not yet in H
                I_L = line_U_intersections[i]
                new_H_cost = 0
                for u in I_L:
                    if u_count[u] == 0:
                        new_H_cost += 1

                score = fc / (new_H_cost + 1)
                if noise_scale > 0:
                    score += random.uniform(0, noise_scale)
                
                if score > best_score:
                    best_score = score
                    best_line_idx = i

            if best_line_idx == -1:
                break

            li = best_line_idx
            chosen_lines.append(li)

            # Update H and u_count
            for u in line_U_intersections[li]:
                if u_count[u] == 0:
                    H.add(u)
                u_count[u] += 1

            # Update uncovered and line_free_count
            for pt in line_outside_list[li]:
                if pt in uncovered:
                    uncovered.remove(pt)
                    # Decrement free count for all lines covering pt
                    for lj in point_to_lines[pt]:
                        line_free_count[lj] -= 1

            iteration += 1
            
        return H, chosen_lines

    def local_search(H, chosen_lines):
        # Build coverage counts for outside-U points
        # cov_count[pt] = number of chosen lines covering pt
        cov_count = {}
        for li in chosen_lines:
            for pt in line_outside_list[li]:
                cov_count[pt] = cov_count.get(pt, 0) + 1

        # u_count[u] = number of chosen lines through unital point u
        u_count = {}
        for u in U:
            u_count[u] = 0
        for li in chosen_lines:
            for u in line_U_intersections[li]:
                u_count[u] += 1

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE - 60:
                break
            for i in range(len(chosen_lines)):
                li = chosen_lines[i]
                
                # Check if all outside-U points on this line are covered by >= 2 lines
                can_remove = True
                for pt in line_outside_list[li]:
                    if cov_count.get(pt, 0) <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove line li
                    chosen_lines.pop(i)
                    
                    # Update cov_count
                    for pt in line_outside_list[li]:
                        cov_count[pt] -= 1
                        if cov_count[pt] == 0:
                            del cov_count[pt]
                    
                    # Update u_count and H
                    for u in line_U_intersections[li]:
                        u_count[u] -= 1
                        if u_count[u] == 0:
                            H.discard(u)
                    
                    improved = True
                    break
        return H, chosen_lines

    best_H = None
    best_size = float('inf')
    best_chosen_lines = None
    
    # Run multiple greedy passes with different seeds
    num_runs = 20
    for r in range(num_runs):
        if time.time() - start_time > DEADLINE - 200:
            break
            
        H, chosen_lines = run_greedy(r)
        if len(H) < best_size:
            best_H = set(H)
            best_chosen_lines = list(chosen_lines)
            best_size = len(H)
            
    if best_H is None:
        H, chosen_lines = run_greedy(0)
        best_H = set(H)
        best_chosen_lines = list(chosen_lines)
        best_size = len(H)

    # Local search on best
    best_H, best_chosen_lines = local_search(best_H, best_chosen_lines)
    
    S = U - best_H
    N = all_points_set - S

    points = []
    for pt in N:
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
