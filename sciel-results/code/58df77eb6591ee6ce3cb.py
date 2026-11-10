import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    candidates = list(range(20, 201))
    n = len(candidates)

    # Precompute gcd matrix for speed
    gcd_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(i, n):
            g = gcd(candidates[i], candidates[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    selected = []  # List of (m, r)
    selected_indices = []  # Index in candidates for each selected

    for i, m in enumerate(candidates):
        if time.time() > deadline:
            break

        best_r = None
        best_score = -float('inf')

        for r in range(m):
            # Check validity against already selected
            valid = True
            for k, (m2, r2) in enumerate(selected):
                g = gcd_matrix[i][selected_indices[k]]
                if r % g == r2 % g:
                    valid = False
                    break
            if not valid:
                continue

            # Compute weighted cost: prioritize small moduli (high 1/m)
            # Cost = sum over future moduli of (1/m_future) * (number of blocked residues)
            cost = 0.0
            for j in range(i + 1, n):
                m2 = candidates[j]
                g = gcd_matrix[i][j]
                if g > 1:
                    # Count of x in [0, m2) with x ≡ r (mod g)
                    blocked = (m2 - 1 - r) // g + 1
                    cost += blocked / m2

            # Score = 1/m - cost (we want to maximize)
            score = 1.0 / m - cost

            if score > best_score:
                best_score = score
                best_r = r

        if best_r is not None:
            selected.append((m, best_r))
            selected_indices.append(i)

    if not selected:
        return [[20, 0]]

    # Local search: try to swap residues to improve compatibility
    improved = True
    while improved and time.time() < deadline - 20.0:
        improved = False
        for idx1 in range(len(selected)):
            if time.time() > deadline - 20.0:
                break
            m1, r1 = selected[idx1]
            for idx2 in range(idx1 + 1, len(selected)):
                m2, r2 = selected[idx2]
                g = gcd(m1, m2)
                if g == 1:
                    continue  # Already compatible

                # Try swapping r1 to a new value that doesn't conflict
                for new_r1 in range(m1):
                    if new_r1 == r1:
                        continue
                    # Check if new_r1 conflicts with any other selected
                    valid = True
                    for idx3 in range(len(selected)):
                        if idx3 == idx1:
                            continue
                        m3, r3 = selected[idx3]
                        g3 = gcd(m1, m3)
                        if new_r1 % g3 == r3 % g3:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Check if this improves compatibility with m2
                    # (new_r1 % g != r2 % g)
                    if new_r1 % g != r2 % g:
                        # This swap resolves the conflict
                        selected[idx1] = (m1, new_r1)
                        improved = True
                        break

    if not selected:
        return [[20, 0]]

    return [[m, r] for m, r in selected]
