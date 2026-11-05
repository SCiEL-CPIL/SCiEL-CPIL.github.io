import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    # Perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Diagonal constraint: 2*v must not be a perfect square
    forbidden = set()
    for v in range(1, N + 1):
        if 2 * v in squares:
            forbidden.add(v)

    valid_vertices = [v for v in range(1, N + 1) if v not in forbidden]
    M = len(valid_vertices)
    vertex_map = valid_vertices  # index -> value

    # Build adjacency (conflict) as bitsets for speed
    # adj_mask[i] is a Python int whose j-th bit is set iff i conflicts with j
    adj_mask = [0] * M
    for i in range(M):
        vi = vertex_map[i]
        for j in range(i + 1, M):
            vj = vertex_map[j]
            if vi + vj in squares:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def can_add(selected_mask, v):
        # v can be added if it conflicts with nothing in selected
        return (selected_mask & adj_mask[v]) == 0

    # ---------- LP relaxation (vertex packing) ----------
    # max sum x_i  s.t. x_i + x_j <= 1 for each edge, 0<=x_i<=1
    # Solve via scipy linprog (HiGHS) on the constraint matrix.
    lp_bound = None
    lp_solution = None
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

        # Build edge list
        edges = []
        for i in range(M):
            m = adj_mask[i]
            j = i + 1
            while m:
                if m & 1:
                    edges.append((i, j))
                m >>= 1
                j += 1

        E = len(edges)
        # c = -1 (maximize sum x)
        c = [-1.0] * M
        # A_ub: each edge row has 1 at i and 1 at j
        A_ub = []
        for (i, j) in edges:
            row = [0.0] * M
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
        A_ub = csr_matrix(A_ub)
        b_ub = [1.0] * E
        bounds = [(0.0, 1.0)] * M

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            lp_bound = float(res.fun)  # = -max sum x
            lp_bound = -lp_bound
            lp_solution = res.x
    except Exception:
        lp_bound = None
        lp_solution = None

    # ---------- Fast bitset local search ----------
    def local_search(initial_mask, budget=5.0):
        selected = initial_mask
        improved = True
        while improved and (elapsed() > 0 and time.time() - start_time < budget + start_time):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Add any non-conflicting vertex
            for v in range(M):
                if not (selected >> v) & 1:
                    if can_add(selected, v):
                        selected |= (1 << v)
                        improved = True
            # 1-for-1 swap: remove v, add a better u
            sel_list = [v for v in range(M) if (selected >> v) & 1]
            for v in sel_list:
                without = selected & ~(1 << v)
                # find best u not in without that fits
                best_u = -1
                best_gain = 0
                for u in range(M):
                    if (without >> u) & 1:
                        continue
                    if can_add(without, u):
                        # net size change is 0 (remove v, add u) unless u==v
                        # only useful if it enables further adds; do simple swap
                        best_u = u
                        break
                # A pure swap doesn't increase size; skip unless it helps.
                # Instead try: remove v, then greedily add everything that fits.
                candidate = without
                for u in range(M):
                    if not (candidate >> u) & 1 and can_add(candidate, u):
                        candidate |= (1 << u)
                if bin(candidate).count('1') > bin(selected).count('1'):
                    selected = candidate
                    improved = True
        return selected

    def mask_to_set(mask):
        return [vertex_map[i] for i in range(M) if (mask >> i) & 1]

    # ---------- Seed solutions ----------
    best_mask = 0
    best_size = 0

    def consider(mask):
        nonlocal best_mask, best_size
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Seed 1: LP-guided greedy (order vertices by LP value descending)
    if lp_solution is not None:
        order = sorted(range(M), key=lambda i: -lp_solution[i])
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        consider(mask)
        mask = local_search(mask, budget=10.0)
        consider(mask)

    # Seed 2: degree-based greedy (pick low-degree first)
    deg = [bin(adj_mask[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: deg[i])
    mask = 0
    for v in order:
        if can_add(mask, v):
            mask |= (1 << v)
    consider(mask)
    mask = local_search(mask, budget=10.0)
    consider(mask)

    # Seed 3: random restarts
    for r in range(200):
        if elapsed() > DEADLINE - 10:
            break
        random.seed(r * 7919 + 13)
        order = list(range(M))
        random.shuffle(order)
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        mask = local_search(mask, budget=0.5)
        consider(mask)

    # ---------- ILP exact solve (if LP says we're close) ----------
    # If LP bound is small above best, try MIP to certify/optimize.
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        import mip

        edges = []
        for i in range(M):
            m = adj_mask[i]
            j = i + 1
            while m:
                if m & 1:
                    edges.append((i, j))
                m >>= 1
                j += 1

        model = mip.Model(sense=mip.MAXIMIZE)
        x = [model.add_var(name=f"x{i}", low=0, up=1, obj=1) for i in range(M)]
        for (i, j) in edges:
            model += x[i] + x[j] <= 1
        model.optimize(max_time=DEADLINE - elapsed() - 2)
        if model.status == mip.OPTIMAL or model.status == mip.BEST_FOUND:
            new_mask = 0
            for i in range(M):
                if x[i].x is not None and x[i].x > 0.5:
                    new_mask |= (1 << i)
            consider(new_mask)
    except Exception:
        pass

    # ---------- Final polish ----------
    final_mask = local_search(best_mask, budget=DEADLINE - elapsed() - 1)
    consider(final_mask)

    result = mask_to_set(best_mask)
    result.sort()

    # Safety: if somehow empty, return a single valid vertex
    if not result:
        for v in range(1, N + 1):
            if 2 * v not in squares:
                return [v]
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    valid = True
    for a in res:
        if 2 * a in squares:
            valid = False
            print(f"Invalid diagonal: {a}")
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if res[i] + res[j] in squares:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print("Valid" if valid else "Invalid")
