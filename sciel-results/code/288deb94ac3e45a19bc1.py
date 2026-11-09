"""XOR-circuit synthesis with shared sub-expression elimination."""
import time
import numpy as np


def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    matrix = np.asarray(matrix, dtype=np.uint8)
    n_rows, n_cols = matrix.shape
    n = n_cols

    # Convert rows to integer bit-vectors
    targets = []
    for r in range(n_rows):
        val = 0
        for c in range(n):
            if matrix[r, c]:
                val |= (1 << c)
        targets.append(val)

    zero = 0
    available = {}  # vector_int -> wire_index
    for c in range(n):
        available[(1 << c)] = c
    available[zero] = n  # constant-zero wire

    next_wire = n + 1
    gates = []
    pending = set(t for t in targets if t not in available)

    if not pending:
        return {'gates': gates, 'outputs': [available[t] for t in targets]}

    def popcount(x):
        return x.bit_count()

    max_iter = 10000
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        if not pending:
            break

        # Generate candidate intermediates
        candidates = {}  # cand_int -> (score, vec_a, vec_b)
        avail_items = list(available.items())

        for T in pending:
            for A, _ in avail_items:
                if A == 0 or A == T:
                    continue
                cand = T ^ A
                if cand == 0 or cand in available:
                    continue
                # Score: sharing potential + overlap bonus
                score = popcount(A & T)
                for T2 in pending:
                    if T2 != cand and (cand & T2) == cand:
                        score += 1
                if cand not in candidates or candidates[cand][0] < score:
                    candidates[cand] = (score, A, T)

        if not candidates:
            # Fallback: synthesize one target from inputs
            T = next(iter(pending))
            bits = [c for c in range(n) if (T >> c) & 1]
            if len(bits) <= 1:
                if T in available:
                    pending.discard(T)
                continue
            a = (1 << bits[0])
            b = (1 << bits[1])
            if a in available and b in available:
                cand = a ^ b
                if cand not in available:
                    candidates[cand] = (1, a, b)
            if not candidates:
                break
            cand = next(iter(candidates))
            _, wa, wb = candidates[cand]
            gates.append([available[wa], available[wb]])
            available[cand] = next_wire
            next_wire += 1
            for T2 in list(pending):
                if T2 in available:
                    pending.discard(T2)
            continue

        # Pick best candidate
        best_cand = max(candidates, key=lambda c: candidates[c][0])
        _, wa, wb = candidates[best_cand]
        gates.append([available[wa], available[wb]])
        available[best_cand] = next_wire
        next_wire += 1

        for T in list(pending):
            if T in available:
                pending.discard(T)

    # Fallback for any remaining pending targets
    for T in list(pending):
        if time.time() > DEADLINE:
            available[T] = next_wire
            next_wire += 1
            continue
        bits = [c for c in range(n) if (T >> c) & 1]
        if len(bits) == 0:
            available[T] = n
        elif len(bits) == 1:
            available[T] = bits[0]
        else:
            current = (1 << bits[0])
            current_wire = bits[0]
            for i in range(1, len(bits)):
                b = (1 << bits[i])
                new_val = current ^ b
                gates.append([current_wire, bits[i]])
                available[new_val] = next_wire
                current = new_val
                current_wire = next_wire
                next_wire += 1
            available[T] = current_wire

    outputs = [available[t] for t in targets]
    return {'gates': gates, 'outputs': outputs}
