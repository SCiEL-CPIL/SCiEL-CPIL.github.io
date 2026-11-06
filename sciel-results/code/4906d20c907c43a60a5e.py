def solve(instance):
    import time
    import random

    n = instance.get("n", 0)
    raw_clauses = instance.get("clauses", [])

    # Robust parsing
    clauses = []
    if isinstance(raw_clauses, (list, tuple)):
        if len(raw_clauses) > 0:
            if isinstance(raw_clauses[0], (list, tuple)):
                clauses = [list(c) for c in raw_clauses]
            else:
                clauses = [list(raw_clauses[i:i+3]) for i in range(0, len(raw_clauses), 3)]
        else:
            clauses = []
    elif isinstance(raw_clauses, int):
        clauses = []
    else:
        clauses = []

    if n == 0 or not clauses:
        return [False] * n

    deadline = time.time() + 290  # Safety margin

    # Pre-compute clauses as 0-based (idx, pos) tuples
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)

    # Build variable -> clauses index for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for idx, _pos in c:
            if 0 <= idx < n:
                var_to_clauses[idx].append(ci)

    def sat_mask(assignment, ci):
        c = clauses_idx[ci]
        mask = 0
        for k, (idx, pos) in enumerate(c):
            if 0 <= idx < n and assignment[idx] == pos:
                mask |= (1 << k)
        return mask

    def evaluate(assignment):
        unsat = []
        for ci in range(m):
            if sat_mask(assignment, ci) == 0:
                unsat.append(ci)
        return unsat

    # 1. Try Gaussian Elimination (Linear/XOR assumption)
    try:
        if time.time() < deadline:
            A = [[0] * n for _ in range(m)]
            for i, clause in enumerate(clauses_idx):
                for idx, _pos in clause:
                    if 0 <= idx < n:
                        A[i][idx] = 1

            aug = [row[:] + [1] for row in A]
            rank = 0
            for col in range(n):
                pivot = -1
                for r in range(rank, m):
                    if aug[r][col] == 1:
                        pivot = r
                        break
                if pivot == -1:
                    continue
                aug[rank], aug[pivot] = aug[pivot], aug[rank]
                for r in range(m):
                    if r != rank and aug[r][col] == 1:
                        for c in range(n + 1):
                            aug[r][c] ^= aug[rank][c]
                rank += 1
                if rank == n:
                    break

            consistent = True
            for r in range(rank, m):
                if aug[r][n] == 1:
                    consistent = False
                    break

            if consistent:
                assignment = [False] * n
                for r in range(rank):
                    pivot_col = -1
                    for c in range(n):
                        if aug[r][c] == 1:
                            pivot_col = c
                            break
                    if pivot_col != -1:
                        assignment[pivot_col] = bool(aug[r][n])

                if not evaluate(assignment):
                    return assignment
    except Exception:
        pass

    # 2. WalkSAT with incremental updates, adaptive noise, and restarts
    best_assignment = None
    best_unsat = m + 1

    for attempt in range(200):
        if time.time() > deadline:
            break

        rng = random.Random(attempt * 7919 + 13)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        masks = [sat_mask(assignment, ci) for ci in range(m)]
        unsat = set(ci for ci in range(m) if masks[ci] == 0)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            best_assignment = assignment[:]

        noise = 0.2
        max_flips = 4000
        last_unsat_count = len(unsat)
        no_improve = 0

        for _flip in range(max_flips):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            cur_unsat = len(unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            if cur_unsat < last_unsat_count:
                no_improve = 0
                last_unsat_count = cur_unsat
            else:
                no_improve += 1

            if no_improve > 100:
                noise = min(0.7, noise + 0.05)
                no_improve = 0

            # Pick random unsatisfied clause
            ci = rng.choice(list(unsat))
            c = clauses_idx[ci]

            # Decide which literal to flip
            unsat_positions = [k for k, (idx, pos) in enumerate(c)
                              if 0 <= idx < n and not (assignment[idx] == pos)]
            if unsat_positions and rng.random() > noise:
                k = rng.choice(unsat_positions)
            else:
                k = rng.randrange(3)
            idx, pos = c[k]

            if not (0 <= idx < n):
                continue

            assignment[idx] = not assignment[idx]

            # Incrementally update masks for clauses touching idx
            for cj in var_to_clauses[idx]:
                new_mask = sat_mask(assignment, cj)
                old_mask = masks[cj]
                if old_mask == 0 and new_mask != 0:
                    unsat.discard(cj)
                elif old_mask != 0 and new_mask == 0:
                    unsat.add(cj)
                masks[cj] = new_mask

    if best_assignment is not None:
        return best_assignment

    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
