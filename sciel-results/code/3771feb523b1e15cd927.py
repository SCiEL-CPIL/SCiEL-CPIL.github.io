import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

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

    def mask_size(mask):
        return bin(mask).count('1')

    # ---------- LP relaxation (vertex packing) ----------
    lp_bound = None
    lp_solution = None
    try:
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

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
        c = [-1.0] * M
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
            lp_bound = -float(res.fun)
            lp_solution = res.x
    except Exception:
        lp_bound = None
        lp_solution = None

    # ---------- Enhanced local search with 1-2 exchange ----------
    def local_search(initial_mask, budget=5.0):
        selected = initial_mask
        cur_size = mask_size(selected)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 1:
            improved = False
            # Phase 1: Add any non-conflicting vertex
            for v in range(M):
                if not (selected >> v) & 1:
                    if can_add(selected, v):
                        selected |= (1 << v)
                        cur_size += 1
                        improved = True
            # Phase 2: 1-2 exchange - remove one vertex, try to add two
            sel_list = [v for v in range(M) if (selected >> v) & 1]
            for v in sel_list:
                if time.time() - start_time > DEADLINE - 2:
                    break
                without = selected & ~(1 << v)
                # Find all vertices that can be added to 'without'
                addable = []
                for u in range(M):
                    if not (without >> u) & 1 and can_add(without, u):
                        addable.append(u)
                # Try adding 2 from addable (if possible)
                if len(addable) >= 2:
                    best_gain = 0
                    best_pair = None
                    for ai in range(len(addable)):
                        for bi in range(ai + 1, len(addable)):
                            u1 = addable[ai]
                            u2 = addable[bi]
                            # Check if u1 and u2 conflict with each other
                            if not (adj_mask[u1] >> u2) & 1:
                                # Net gain: remove 1, add 2 = +1
                                if best_gain < 1:
                                    best_gain = 1
                                    best_pair = (u1, u2)
                                    break
                        if best_pair:
                            break
                    if best_pair:
                        selected = without | (1 << best_pair[0]) | (1 << best_pair[1])
                        cur_size += 1
                        improved = True
                else:
                    # Try adding 1 (net gain 0, but may enable further adds)
                    if addable:
                        candidate = without
                        for u in addable:
                            if can_add(candidate, u):
                                candidate |= (1 << u)
                        if mask_size(candidate) > cur_size:
                            selected = candidate
                            cur_size = mask_size(selected)
                            improved = True
            # Phase 3: Greedy fill after exchanges
            for v in range(M):
                if not (selected >> v) & 1:
                    if can_add(selected, v):
                        selected |= (1 << v)
                        cur_size += 1
                        improved = True
        return selected

    def mask_to_set(mask):
        return [vertex_map[i] for i in range(M) if (mask >> i) & 1]

    # ---------- Seed solutions ----------
    best_mask = 0
    best_size = 0

    def consider(mask):
        nonlocal best_mask, best_size
        sz = mask_size(mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Seed 1: LP-guided greedy
    if lp_solution is not None:
        order = sorted(range(M), key=lambda i: -lp_solution[i])
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        consider(mask)
        mask = local_search(mask, budget=8.0)
        consider(mask)

    # Seed 2: degree-based greedy
    deg = [bin(adj_mask[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: deg[i])
    mask = 0
    for v in order:
        if can_add(mask, v):
            mask |= (1 << v)
    consider(mask)
    mask = local_search(mask, budget=8.0)
    consider(mask)

    # Seed 3: Modular-residue-based partitioning heuristic
    # Assign vertices to groups by residue mod small numbers,
    # then greedily pick from each group to minimize intra-group conflicts
    for mod in [4, 8, 16, 3, 5, 7, 12]:
        if time.time() - start_time > DEADLINE - 15:
            break
        # Group vertices by residue
        groups = {}
        for i in range(M):
            r = vertex_map[i] % mod
            if r not in groups:
                groups[r] = []
            groups[r].append(i)
        # For each group, sort by degree and greedily select
        mask = 0
        for r in sorted(groups.keys()):
            group = sorted(groups[r], key=lambda i: deg[i])
            for v in group:
                if can_add(mask, v):
                    mask |= (1 << v)
        consider(mask)
        mask = local_search(mask, budget=3.0)
        consider(mask)

    # Seed 4: Parity-based partitioning
    # Split by parity, then by higher-order bits
    for bit_pos in [0, 1, 2, 3]:
        if time.time() - start_time > DEADLINE - 10:
            break
        mask = 0
        # Process vertices in order of their bit at position bit_pos
        # First those with bit=0, then bit=1 (or vice versa)
        for bit_val in [0, 1]:
            subset = [i for i in range(M) if (vertex_map[i] >> bit_pos) & 1 == bit_val]
            subset.sort(key=lambda i: deg[i])
            for v in subset:
                if can_add(mask, v):
                    mask |= (1 << v)
        consider(mask)
        mask = local_search(mask, budget=3.0)
        consider(mask)

    # Seed 5: Random restarts with varied strategies
    for r in range(300):
        if time.time() - start_time > DEADLINE - 8:
            break
        random.seed(r * 7919 + 13)
        if r % 3 == 0:
            # Pure random order
            order = list(range(M))
            random.shuffle(order)
        elif r % 3 == 1:
            # Random + degree perturbation
            order = list(range(M))
            random.shuffle(order)
            order.sort(key=lambda i: deg[i] + random.random() * 5)
        else:
            # Random modular grouping
            mod = random.choice([3, 4, 5, 7, 8, 12, 16])
            groups = {}
            for i in range(M):
                gr = random.randint(0, mod - 1)
                if gr not in groups:
                    groups[gr] = []
                groups[gr].append(i)
            order = []
            for gr in sorted(groups.keys()):
                random.shuffle(groups[gr])
                order.extend(groups[gr])
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        mask = local_search(mask, budget=0.3)
        consider(mask)

    # ---------- ILP exact solve ----------
    try:
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

        remaining_time = DEADLINE - elapsed() - 3
        if remaining_time > 5:
            model = mip.Model(sense=mip.MAXIMIZE)
            x = [model.add_var(name=f"x{i}", low=0, up=1, obj=1) for i in range(M)]
            for (i, j) in edges:
                model += x[i] + x[j] <= 1
            model.optimize(max_time=remaining_time)
            if model.status == mip.OPTIMAL or model.status == mip.BEST_FOUND:
                new_mask = 0
                for i in range(M):
                    if x[i].x is not None and x[i].x > 0.5:
                        new_mask |= (1 << i)
                consider(new_mask)
    except Exception:
        pass

    # ---------- Final polish with enhanced local search ----------
    final_mask = local_search(best_mask, budget=DEADLINE - elapsed() - 1)
    consider(final_mask)

    result = mask_to_set(best_mask)
    result.sort()

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
