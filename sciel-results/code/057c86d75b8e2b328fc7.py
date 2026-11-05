import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Try multiple partition strategies and pick the best (lowest rank)
    # Strategy: partition into blocks, use Strassen (rank 7) for 2x2x2, schoolbook otherwise

    def compute_rank(partition):
        """Compute total rank for a given partition."""
        total = 0
        offsets = [0]
        for s in partition:
            offsets.append(offsets[-1] + s)
        for i in range(len(partition)):
            for j in range(len(partition)):
                for k in range(len(partition)):
                    ni = partition[i]
                    mj = partition[j]
                    pk = partition[k]
                    if ni == 2 and mj == 2 and pk == 2:
                        total += 7
                    else:
                        total += ni * mj * pk
        return total

    # Find the partition of 5 that minimizes rank
    # Possible partitions of 5 (as unordered multisets, but order matters for blocks)
    # We'll try all compositions of 5
    def compositions(total, max_part=5):
        if total == 0:
            yield []
            return
        for first in range(1, min(total, max_part) + 1):
            for rest in compositions(total - first, max_part):
                yield [first] + rest

    best_partition = None
    best_rank = 125  # schoolbook

    for comp in compositions(5):
        r = compute_rank(comp)
        if r < best_rank:
            best_rank = r
            best_partition = comp

    # Also try specific known good partitions
    test_partitions = [
        [2, 2, 1],
        [1, 2, 2],
        [2, 1, 2],
        [1, 1, 3],
        [3, 1, 1],
        [1, 3, 1],
        [1, 1, 1, 2],
        [2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 1, 1],
    ]

    for part in test_partitions:
        if sum(part) == 5:
            r = compute_rank(part)
            if r < best_rank:
                best_rank = r
                best_partition = part

    if best_partition is None:
        best_partition = [5]
        best_rank = 125

    block_sizes = best_partition
    num_blocks = len(block_sizes)

    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Strassen decomposition for 2x2x2
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()

    # Calculate total rank
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                if time.time() > deadline:
                    break

                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]

                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]

                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"

    return {"U": U, "V": V, "W": W}
