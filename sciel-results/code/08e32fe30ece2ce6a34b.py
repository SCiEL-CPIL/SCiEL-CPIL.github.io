import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s. Keep a hard safety margin.
    deadline = start_time + 140.0

    total_space = q ** n

    # Digit weights for base-q expansion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Precompute the neighbor set (ball of given radius) for every word.
    # For the target task (q=3, n=7, radius=1) this is 1+7*2 = 15 neighbors.
    # We build an explicit list of neighbor indices for each word.
    neighbor_lists = []
    for i in range(total_space):
        nbrs = {i}
        # Generate all words within Hamming distance `radius`.
        # For small radius and n, a recursive/iterative expansion is fine.
        # Here we use a BFS/DFS up to depth `radius` in the Hamming graph.
        # Given the small space (2187) and small radius (1), direct enumeration is fast.
        if radius == 1:
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val:
                        continue
                    nb = i + (new_val - digit_val) * powers[pos]
                    nbrs.add(nb)
        else:
            # General radius: use BFS
            from collections import deque
            visited = {i}
            dq = deque([i])
            dist = {i: 0}
            while dq:
                cur = dq.popleft()
                d = dist[cur]
                if d >= radius:
                    continue
                for pos in range(n):
                    digit_val = (cur // powers[pos]) % q
                    for new_val in range(q):
                        if new_val == digit_val:
                            continue
                        nb = cur + (new_val - digit_val) * powers[pos]
                        if nb not in visited:
                            visited.add(nb)
                            dist[nb] = d + 1
                            dq.append(nb)
            nbrs = visited
        neighbor_lists.append(sorted(nbrs))

    # Build the incidence matrix A (total_space x total_space) where
    # A[word, center] = 1 if `word` is within radius of `center`.
    # This is the transpose of the "center covers word" relation.
    # We want: for each word w, sum_{c} A[w, c] * x[c] >= 1, minimize sum x[c].
    A = np.zeros((total_space, total_space), dtype=np.int8)
    for center in range(total_space):
        for word in neighbor_lists[center]:
            A[word, center] = 1

    # ILP formulation:
    # Minimize: sum(x)
    # Subject to: A @ x >= 1  (each word covered)
    #             x in {0, 1}
    c = np.ones(total_space, dtype=np.float64)

    # Constraints: A @ x >= 1  =>  -A @ x <= -1
    # LinearConstraint form: lb <= A_mat @ x <= ub
    # We use: A @ x >= 1  =>  A @ x >= ones
    # scipy milp uses LinearConstraint(A_mat, lb, ub)
    lb = np.ones(total_space, dtype=np.float64)
    ub = np.full(total_space, np.inf, dtype=np.float64)
    constraints = LinearConstraint(A.astype(np.float64), lb, ub)

    integrality = np.ones(total_space, dtype=int)  # all binary
    bounds = Bounds(lb=np.zeros(total_space), ub=np.ones(total_space))

    # Time limit: leave margin for result extraction and safety.
    time_limit = max(5.0, deadline - start_time - 10.0)

    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0}
        )
    except Exception:
        # Fallback: return a trivial valid (but large) solution
        # Every word is a codeword -> covers everything
        return [[(i // powers[pos]) % q for pos in range(n)] for i in range(total_space)]

    # Extract solution
    if res.x is not None:
        # res.x may have small floating point errors; round
        x = np.round(res.x).astype(int)
        # Ensure binary
        x = (x > 0.5).astype(int)
        selected = [i for i in range(total_space) if x[i] == 1]
    else:
        selected = list(range(total_space))

    # Validate and repair if necessary (in case solver returned infeasible or partial)
    # Check coverage
    covered = np.zeros(total_space, dtype=np.int8)
    for c_idx in selected:
        for w in neighbor_lists[c_idx]:
            covered[w] = 1

    if not np.all(covered == 1):
        # Repair: add uncovered words as codewords
        uncovered = np.where(covered == 0)[0]
        selected.extend(uncovered.tolist())

    # Remove duplicates (shouldn't happen but safe)
    selected = list(set(selected))

    # Quick greedy reduction to try to improve if ILP didn't find optimum
    # (ILP with time limit may return a suboptimal but valid solution)
    # Do a simple greedy removal of redundant codewords
    if len(selected) > 1 and time.time() < deadline - 2:
        # Build coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for c_idx in selected:
            for w in neighbor_lists[c_idx]:
                counts[w] += 1

        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(selected)):
                c_idx = selected[i]
                # Check if removing c_idx leaves any word uncovered
                removable = True
                for w in neighbor_lists[c_idx]:
                    if counts[w] == 1:
                        removable = False
                        break
                if removable:
                    for w in neighbor_lists[c_idx]:
                        counts[w] -= 1
                    selected.pop(i)
                    changed = True
                    break  # restart scan after removal

    # Decode to tuples
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    best_words = [int_to_tuple(v) for v in selected]

    return best_words
