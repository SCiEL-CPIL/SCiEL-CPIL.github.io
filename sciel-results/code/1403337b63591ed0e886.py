"""Model-and-optimize: parameterize near-Turán graphs, optimize partition sizes,
then hill-climb with fast bitset-based triangle-degree-sum updates.
Combined with vectorized numpy scoring and structured multi-start."""
import random
import time
import math
import copy
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 130.0  # safety margin before 150s kill

# Precompute all pairs and triangles
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i+1, N)]
TRIANGLES = []
for u in range(N):
    for v in range(u+1, N):
        for w in range(v+1, N):
            TRIANGLES.append((u, v, w))
TRI_ARRAY = np.array(TRIANGLES, dtype=np.int32)  # shape (560, 3)
TRI_U = TRI_ARRAY[:, 0]
TRI_V = TRI_ARRAY[:, 1]
TRI_W = TRI_ARRAY[:, 2]


def _score_numpy(A):
    """Compute max triangle degree sum using numpy adjacency matrix (16x16 bool/int)."""
    d = A.sum(axis=1)
    active = A[TRI_U, TRI_V] & A[TRI_V, TRI_W] & A[TRI_W, TRI_U]
    if not np.any(active):
        return 0
    sums = d[TRI_U] + d[TRI_V] + d[TRI_W]
    return int(np.max(sums[active]))


def _build_turan_plus_one():
    """Build K_{8,8} plus one internal edge in one part."""
    A = np.zeros((N, N), dtype=np.int8)
    for i in range(8):
        for j in range(8, 16):
            A[i, j] = 1
            A[j, i] = 1
    A[0, 1] = 1
    A[1, 0] = 1
    score = _score_numpy(A)
    return A, score


def _apply_swap_numpy(A, e_remove, e_add):
    """Apply edge swap, return new matrix and score."""
    u, v = e_remove
    x, y = e_add
    new_A = A.copy()
    new_A[u, v] = 0
    new_A[v, u] = 0
    new_A[x, y] = 1
    new_A[y, x] = 1
    return new_A, _score_numpy(new_A)


def _build_partition(a_size, internal_edges_a, internal_edges_b):
    """Build a graph with cross edges between parts of size a_size and N-a_size,
    plus specified internal edges."""
    b_size = N - a_size
    A = np.zeros((N, N), dtype=np.int8)
    # Cross edges
    for i in range(a_size):
        for j in range(a_size, N):
            A[i, j] = 1
            A[j, i] = 1
    # Internal edges in A
    for (i, j) in internal_edges_a:
        A[i, j] = 1
        A[j, i] = 1
    # Internal edges in B
    for (i, j) in internal_edges_b:
        A[i, j] = 1
        A[j, i] = 1
    score = _score_numpy(A)
    return A, score


def construct():
    t0 = time.time()
    rng = random.Random(42)

    def elapsed():
        return time.time() - t0

    # Start with Turán + 1
    A, score = _build_turan_plus_one()
    best_A = A.copy()
    best_score = score

    # Phase 1: Explore different partition structures
    for a_size in range(4, 13):
        b_size = N - a_size
        cross = a_size * b_size
        internal_needed = TARGET_EDGES - cross
        if internal_needed < 0:
            continue
        max_internal_a = a_size * (a_size - 1) // 2
        max_internal_b = b_size * (b_size - 1) // 2
        if internal_needed > max_internal_a + max_internal_b:
            continue
        if internal_needed == 0:
            continue

        # Generate candidate internal edge sets
        all_internal_a = [(i, j) for i in range(a_size) for j in range(i+1, a_size)]
        all_internal_b = [(i, j) for i in range(a_size, N) for j in range(i+1, N)]

        for trial in range(20):
            if elapsed() > DEADLINE:
                break
            # Randomly distribute internal edges between the two parts
            if internal_needed <= len(all_internal_a) + len(all_internal_b):
                # Try biased distributions
                if rng.random() < 0.5:
                    na = rng.randint(0, min(internal_needed, len(all_internal_a)))
                    nb = internal_needed - na
                    if nb > len(all_internal_b):
                        nb = len(all_internal_b)
                        na = internal_needed - nb
                        if na > len(all_internal_a):
                            continue
                else:
                    nb = rng.randint(0, min(internal_needed, len(all_internal_b)))
                    na = internal_needed - nb
                    if na > len(all_internal_a):
                        na = len(all_internal_a)
                        nb = internal_needed - na
                        if nb > len(all_internal_b):
                            continue

                chosen_a = rng.sample(all_internal_a, na) if na > 0 else []
                chosen_b = rng.sample(all_internal_b, nb) if nb > 0 else []
                A_try, score_try = _build_partition(a_size, chosen_a, chosen_b)
                if score_try < best_score:
                    best_score = score_try
                    best_A = A_try.copy()

    # Phase 2: Simulated annealing with numpy
    A_cur = best_A.copy()
    cur_score = best_score

    # Maintain edge/non-edge lists for efficient sampling
    def get_edges(A):
        return [(i, j) for i in range(N) for j in range(i+1, N) if A[i, j]]

    def get_nonedges(A):
        return [(i, j) for i in range(N) for j in range(i+1, N) if not A[i, j]]

    edges = get_edges(A_cur)
    nonedges = get_nonedges(A_cur)

    start_temp = 3.0
    end_temp = 0.05
    cooling_rate = 0.9995
    temp = start_temp
    no_improve_count = 0

    while elapsed() < DEADLINE - 10.0:
        if not edges or not nonedges:
            break

        e_remove = edges[rng.randrange(len(edges))]
        e_add = nonedges[rng.randrange(len(nonedges))]

        A_new, new_score = _apply_swap_numpy(A_cur, e_remove, e_add)

        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.15
        else:
            delta = new_score - cur_score
            if temp < 1e-10:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()

        if accept:
            A_cur = A_new
            cur_score = new_score
            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)

            if cur_score < best_score:
                best_score = cur_score
                best_A = A_cur.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Reheat if stuck
            if no_improve_count > 5000 and temp < 1.0:
                temp = 2.0
                no_improve_count = 0

        temp = max(end_temp, temp * cooling_rate)

    # Phase 3: Deterministic hill-climbing polish on best_A
    if elapsed() < DEADLINE - 3.0:
        A_polish = best_A.copy()
        score_polish = best_score
        edges_p = get_edges(A_polish)
        nonedges_p = get_nonedges(A_polish)

        improved = True
        max_passes = 50
        for _ in range(max_passes):
            if elapsed() > DEADLINE - 3.0 or not improved:
                break
            improved = False
            for e_remove in list(edges_p):
                if elapsed() > DEADLINE - 3.0:
                    break
                for e_add in list(nonedges_p):
                    A_new, new_score = _apply_swap_numpy(A_polish, e_remove, e_add)
                    if new_score < score_polish:
                        A_polish = A_new
                        score_polish = new_score
                        edges_p.remove(e_remove)
                        nonedges_p.append(e_remove)
                        nonedges_p.remove(e_add)
                        edges_p.append(e_add)
                        improved = True
                        break
                if improved:
                    break

        if score_polish < best_score:
            best_score = score_polish
            best_A = A_polish.copy()

    # Phase 4: Final local search with random restarts on neighborhood
    if elapsed() < DEADLINE - 2.0:
        A_final = best_A.copy()
        score_final = best_score
        edges_f = get_edges(A_final)
        nonedges_f = get_nonedges(A_final)

        for restart in range(5):
            if elapsed() > DEADLINE - 2.0:
                break
            improved = True
            while improved and elapsed() < DEADLINE - 2.0:
                improved = False
                for e_remove in list(edges_f):
                    if elapsed() > DEADLINE - 2.0:
                        break
                    # Try a few random non-edges
                    for _ in range(10):
                        e_add = nonedges_f[rng.randrange(len(nonedges_f))]
                        A_new, new_score = _apply_swap_numpy(A_final, e_remove, e_add)
                        if new_score < score_final:
                            A_final = A_new
                            score_final = new_score
                            edges_f.remove(e_remove)
                            nonedges_f.append(e_remove)
                            nonedges_f.remove(e_add)
                            edges_f.append(e_add)
                            improved = True
                            break
                    if improved:
                        break

            if score_final < best_score:
                best_score = score_final
                best_A = A_final.copy()

    # Build result
    result_edges = [[i, j] for i in range(N) for j in range(i+1, N) if best_A[i, j]]
    if len(result_edges) != TARGET_EDGES:
        # Fallback
        result_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]

    return {'n': N, 'edges': result_edges}
