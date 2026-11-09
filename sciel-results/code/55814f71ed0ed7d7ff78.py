import time
import numpy as np

N = 40
M = 401
DEADLINE = 140.0  # wall-clock guard, safety margin under 150s kill


def _book(A):
    """book[e] = number of triangles containing edge e (e is a flat index)."""
    T = A @ A  # T[i,j] = number of common neighbors of i and j
    return T  # book size of edge (i,j) is T[i,j]


def _max_book(T):
    iu = np.triu_indices(N, k=1)
    return int(T[iu].max())


def _max_book_val(A):
    T = A @ A
    iu = np.triu_indices(N, k=1)
    return int(T[iu].max())


def _init_edges():
    edges = [(a, b) for a in range(20) for b in range(20, 40)]
    edges.append((0, 1))
    return edges


def construct():
    t0 = time.time()

    edges = _init_edges()
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    A = np.zeros((N, N), dtype=np.int32)
    for u, v in edge_set:
        A[u, v] = 1
        A[v, u] = 1

    iu = np.triu_indices(N, k=1)
    all_pairs = np.array(iu)  # shape (2, 780)
    cur_T = A @ A
    cur_max = int(cur_T[iu].max())

    # Precompute edge flat indices for all candidate pairs
    # candidate pool: pairs not currently edges (for add), edges (for remove)
    # We will do random swaps: pick a random edge to remove and a random non-edge to add.

    # Map flat index -> (u,v)
    # iu[0], iu[1] arrays
    u_arr = iu[0]
    v_arr = iu[1]

    # current edge flat indices
    cur_edge_flat = []
    for u, v in edge_set:
        # find flat index: since iu is row-major, flat = u*N + v - u*(u+1)//2
        flat = u * N + v - u * (u + 1) // 2
        cur_edge_flat.append(flat)
    cur_edge_flat = np.array(cur_edge_flat, dtype=np.int64)

    # non-edge flat indices
    is_edge = np.zeros(780, dtype=bool)
    is_edge[cur_edge_flat] = True
    nonedge_flat = np.array(np.where(~is_edge)[0], dtype=np.int64)

    best_max = cur_max
    best_edge_set = set(edge_set)

    rng = np.random.default_rng(12345)

    while time.time() < DEADLINE:
        # pick random edge to remove
        ri = rng.integers(len(cur_edge_flat))
        rem_flat = int(cur_edge_flat[ri])
        ru = int(u_arr[rem_flat])
        rv = int(v_arr[rem_flat])

        # pick random non-edge to add
        ai = rng.integers(len(nonedge_flat))
        add_flat = int(nonedge_flat[ai])
        au = int(u_arr[add_flat])
        av = int(v_arr[add_flat])

        # apply swap
        A[ru, rv] = 0
        A[rv, ru] = 0
        A[au, av] = 1
        A[av, au] = 1

        new_T = A @ A
        new_max = int(new_T[iu].max())

        if new_max < best_max:
            best_max = new_max
            best_edge_set = set(edge_set)
            # commit
            edge_set.discard((ru, rv))
            edge_set.add((au, av))
            # update flat arrays
            cur_edge_flat[ri] = add_flat
            nonedge_flat[ai] = rem_flat
            is_edge[rem_flat] = False
            is_edge[add_flat] = True
            cur_T = new_T
            if best_max <= 7:
                # cannot do better than theoretical bound floor; 7 is the theorem bound
                # but we can't go below ceil(40/6)=7, so 7 is optimal
                break
        else:
            # revert
            A[ru, rv] = 1
            A[rv, ru] = 1
            A[au, av] = 0
            A[av, au] = 0
            cur_T = A @ A

    # build final edge list
    out = []
    for u, v in best_edge_set:
        out.append([u, v])
    return out


# quick self-test (not used by evaluator)
if __name__ == "__main__":
    e = construct()
    assert len(e) == 401
    s = set()
    for u, v in e:
        assert 0 <= u < 40 and 0 <= v < 40 and u != v
        s.add(frozenset((u, v)))
    assert len(s) == 401
    # compute score
    A = np.zeros((40, 40), dtype=np.int32)
    for u, v in e:
        A[u, v] = A[v, u] = 1
    T = A @ A
    iu = np.triu_indices(40, k=1)
    print("max book:", int(T[iu].max()))
