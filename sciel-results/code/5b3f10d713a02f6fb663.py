<<<<<<< SEARCH
def _find_orthogonal(S, time_limit, t_start, max_nodes=2000000):
    """Backtrack a new Latin square orthogonal to every square in S."""
    K = len(S)
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    pair_used = [set() for _ in range(K)]
    grid = [[-1] * N for _ in range(N)]
    state = {"count": 0, "found": None}

    # Precompute pair constraints as a 3D boolean array for speed
    # pair_forbidden[k][i][j][s] = True if (S[k][i][j], s) is already used
    pair_forbidden = [[[[False] * N for _ in range(N)] for _ in range(N)] for _ in range(K)]
    for k in range(K):
        for i in range(N):
            for j in range(N):
                for s in range(N):
                    if (S[k][i][j], s) in pair_used[k]:
                        pair_forbidden[k][i][j][s] = True

    def get_candidates(i, j):
        ru = row_used[i]
        cu = col_used[j]
        res = []
        for s in range(N):
            if s in ru or s in cu:
                continue
            ok = True
            for k in range(K):
                if pair_forbidden[k][i][j][s]:
                    ok = False
                    break
            if ok:
                res.append(s)
        return res

    def solve(idx):
        if state["found"] is not None:
            return
        if state["count"] > max_nodes or (time.time() - t_start) > time_limit:
            return
        if idx == N * N:
            state["found"] = [row[:] for row in grid]
            return
        i = idx // N
        j = idx % N
        cands = get_candidates(i, j)
        random.shuffle(cands)
        for s in cands:
            grid[i][j] = s
            row_used[i].add(s)
            col_used[j].add(s)
            for k in range(K):
                pair_forbidden[k][i][j][s] = True
            state["count"] += 1
            solve(idx + 1)
            if state["found"] is not None:
                return
            grid[i][j] = -1
            row_used[i].remove(s)
            col_used[j].remove(s)
            for k in range(K):
                pair_forbidden[k][i][j][s] = False

    solve(0)
    return state["found"]
=======
def _find_orthogonal(S, time_limit, t_start, max_nodes=5000000):
    """Backtrack a new Latin square orthogonal to every square in S using MRV."""
    K = len(S)
    row_used = [set() for _ in range(N)]
    col_used = [set() for _ in range(N)]
    grid = [[-1] * N for _ in range(N)]
    state = {"count": 0, "found": None}

    # Precompute pair constraints: pair_forbidden[k][i][j] = set of symbols s
    # such that (S[k][i][j], s) would be a duplicate pair
    pair_forbidden = [set() for _ in range(K)]
    for k in range(K):
        for i in range(N):
            for j in range(N):
                pair_forbidden[k].add(S[k][i][j])

    def get_candidates(i, j):
        ru = row_used[i]
        cu = col_used[j]
        res = []
        for s in range(N):
            if s in ru or s in cu:
                continue
            ok = True
            for k in range(K):
                if s in pair_forbidden[k]:
                    ok = False
                    break
            if ok:
                res.append(s)
        return res

    def solve(remaining):
        if state["found"] is not None:
            return
        if state["count"] > max_nodes or (time.time() - t_start) > time_limit:
            return
        if not remaining:
            state["found"] = [row[:] for row in grid]
            return

        # MRV: find cell with fewest candidates
        best_i, best_j, best_cands = -1, -1, None
        for i, j in remaining:
            cands = get_candidates(i, j)
            if not cands:
                return  # dead end
            if best_cands is None or len(cands) < len(best_cands):
                best_i, best_j, best_cands = i, j, cands
                if len(best_cands) == 1:
                    break

        random.shuffle(best_cands)
        remaining.remove((best_i, best_j))
        for s in best_cands:
            grid[best_i][best_j] = s
            row_used[best_i].add(s)
            col_used[best_j].add(s)
            # Update pair_forbidden: for each k, s is now used at position (best_i, best_j)
            # Actually pair_forbidden tracks which symbol values are forbidden at each position
            # We need a different approach: track used pairs per k
            state["count"] += 1
            solve(remaining)
            if state["found"] is not None:
                return
            grid[best_i][best_j] = -1
            row_used[best_i].remove(s)
            col_used[best_j].remove(s)
        remaining.add((best_i, best_j))

    all_cells = [(i, j) for i in range(N) for j in range(N)]
    solve(all_cells)
    return state["found"]
>>>>>>> REPLACE

Wait, I realize the `pair_forbidden` logic is wrong in my replacement. Let me fix it properly. The pair constraint is: for each existing square k, at position (i,j), the symbol s in the new square must not equal any symbol that already appears paired with S[k][i][j] in another position. Actually, the constraint is simpler: for each k, the set of pairs {(S[k][i][j], new[i][j])} over all (i,j) must be all distinct. This means: for a fixed k and fixed symbol value v = S[k][i][j], the new square can only use each symbol s at most once across all positions where S[k] has value v.

Let me rewrite this more carefully:

