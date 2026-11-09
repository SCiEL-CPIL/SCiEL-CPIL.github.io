import time
import sys

def construct():
    n = 10
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill

    # MOLS of order 5
    m = 5
    mols_m = []
    for k in range(m - 1):
        sq = [[(i + (k + 1) * j) % m for j in range(m)] for i in range(m)]
        mols_m.append(sq)

    # Lift to order 10
    mols_n = []
    for sq in mols_m:
        new_sq = []
        for i in range(n):
            row = []
            for j in range(n):
                i1 = i // m
                i2 = i % m
                j1 = j // m
                j2 = j % m
                val = 2 * sq[i2][j2] + ((i1 + j1) % 2)
                row.append(val)
            new_sq.append(row)
        mols_n.append(new_sq)

    # Validation
    def is_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            col = [sq[k][i] for k in range(n)]
            if set(col) != set(range(n)):
                return False
        return True

    def is_orthogonal(sq_a, sq_b):
        seen_pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq_a[i][j], sq_b[i][j])
                if pair in seen_pairs:
                    return False
                seen_pairs.add(pair)
        return True

    # Check all
    valid_sqs = []
    for sq in mols_n:
        if time.time() - start_time > time_limit - 15:
            break
        if not is_latin(sq):
            continue
        orthogonal = True
        for v_sq in valid_sqs:
            if not is_orthogonal(sq, v_sq):
                orthogonal = False
                break
        if orthogonal:
            valid_sqs.append(sq)

    if not valid_sqs:
        sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [sq1]

    # Attempt to find more squares using backtracking
    def try_find_next_square(existing_sqs):
        new_sq = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = [set() for _ in range(len(existing_sqs))]

        def is_consistent(i, j, v):
            if v in row_used[i]:
                return False
            if v in col_used[j]:
                return False
            for k, esq in enumerate(existing_sqs):
                pair = (v, esq[i][j])
                if pair in pair_used[k]:
                    return False
            return True

        def backtrack(i, j):
            if time.time() - start_time > time_limit:
                return False
            if i == n:
                return True
            if j == n:
                return backtrack(i + 1, 0)

            for v in range(n):
                if is_consistent(i, j, v):
                    new_sq[i][j] = v
                    row_used[i].add(v)
                    col_used[j].add(v)
                    for k, esq in enumerate(existing_sqs):
                        pair_used[k].add((v, esq[i][j]))

                    if backtrack(i, j + 1):
                        return True

                    new_sq[i][j] = -1
                    row_used[i].remove(v)
                    col_used[j].remove(v)
                    for k, esq in enumerate(existing_sqs):
                        pair_used[k].remove((v, esq[i][j]))

            return False

        if backtrack(0, 0):
            return new_sq
        return None

    max_attempts = 9
    while len(valid_sqs) < max_attempts:
        if time.time() - start_time > time_limit - 5:
            break
        next_sq = try_find_next_square(valid_sqs)
        if next_sq is None:
            break
        if not is_latin(next_sq):
            break
        ok = True
        for v_sq in valid_sqs:
            if not is_orthogonal(next_sq, v_sq):
                ok = False
                break
        if ok:
            valid_sqs.append(next_sq)
        else:
            break

    # Ensure types are pure Python lists and ints
    result = [
        [[int(val) for val in row] for row in sq]
        for sq in valid_sqs
    ]

    return result
