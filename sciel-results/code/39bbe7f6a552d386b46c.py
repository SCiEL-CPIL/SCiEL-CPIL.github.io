"""Rewritten: search for MOLS of order 10 with constraint propagation + deadline guard."""
import time

N = 10
SYMS = list(range(N))

def _validate(squares):
    if not squares:
        return False
    for L in squares:
        for r in range(N):
            if sorted(L[r]) != SYMS:
                return False
        for c in range(N):
            col = [L[r][c] for r in range(N)]
            if sorted(col) != SYMS:
                return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            seen = set()
            for r in range(N):
                for c in range(N):
                    p = (squares[i][r][c], squares[j][r][c])
                    if p in seen:
                        return False
                    seen.add(p)
    return True

def _build_square(base_squares, deadline):
    n = N
    # base_squares: list of existing squares this new one must be orthogonal to
    row_cand = [set(SYMS) for _ in range(n)]
    col_cand = [set(SYMS) for _ in range(n)]
    pair_cand = [set(SYMS) for _ in base_squares]
    for r in range(n):
        for c in range(n):
            for idx, B in enumerate(base_squares):
                pair_cand[idx].discard(B[r][c])

    grid = [[None] * n for _ in range(n)]

    def backtrack(r, c):
        if time.time() > deadline:
            return False
        if r == n:
            return True
        if c == n:
            return backtrack(r + 1, 0)

        # choose next empty cell (row-major is fine; we fill sequentially)
        candidates = row_cand[r] & col_cand[c]
        for v in sorted(candidates):
            if all(v in pair_cand[idx] for idx in range(len(base_squares))):
                grid[r][c] = v
                row_cand[r].discard(v)
                col_cand[c].discard(v)
                for idx in range(len(base_squares)):
                    pair_cand[idx].discard(v)

                # forward check: each future row/col/pair must still have options
                ok = True
                for rr in range(r, n):
                    if not (row_cand[rr] & (set(SYMS) - set())):
                        pass
                    # row must be fillable: remaining count in row == remaining cells
                    rem_cells_row = (n - 1 - c) if rr == r else (n - 1)
                    # actually compute per row
                # simpler: check each row has enough candidates for its remaining cells
                for rr in range(n):
                    placed_in_row = sum(1 for cc in range(n) if grid[rr][cc] is not None)
                    rem_cells = n - placed_in_row
                    if len(row_cand[rr]) < rem_cells:
                        ok = False
                        break
                if ok:
                    for cc in range(n):
                        placed_in_col = sum(1 for rr2 in range(n) if grid[rr2][cc] is not None)
                        rem_cells = n - placed_in_col
                        if len(col_cand[cc]) < rem_cells:
                            ok = False
                            break
                if ok:
                    for idx in range(len(base_squares)):
                        placed_in_pair = sum(1 for rr2 in range(n) for cc2 in range(n) if grid[rr2][cc2] is not None)
                        rem_cells = n * n - placed_in_pair
                        if len(pair_cand[idx]) < rem_cells:
                            ok = False
                            break
                if ok:
                    if backtrack(r, c + 1):
                        return True

                # undo
                row_cand[r].add(v)
                col_cand[c].add(v)
                for idx in range(len(base_squares)):
                    pair_cand[idx].add(v)
                grid[r][c] = None
        return False

    if backtrack(0, 0):
        # convert to plain ints/ lists
        return [[int(grid[r][c]) for c in range(n)] for r in range(n)]
    return None

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    squares = [base]

    # Try to add squares one by one
    for _ in range(8):
        if time.time() > deadline - 10.0:
            break
        nxt = _build_square(squares, deadline)
        if nxt is None:
            break
        squares.append(nxt)

    # Final safety validation
    if _validate(squares):
        # ensure plain Python ints and lists
        out = []
        for L in squares:
            out.append([[int(L[r][c]) for c in range(N)] for r in range(N)])
        return out

    # Fallback: single valid square
    return [[int((i + j) % N) for j in range(N)] for i in range(N)]
