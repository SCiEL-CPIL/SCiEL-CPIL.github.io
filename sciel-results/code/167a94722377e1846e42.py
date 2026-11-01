def gauss(A, b, nvar):
    import copy
    A = [row[:] for row in A]
    b = b[:]
    neq = len(A)
    pivot_cols = []
    row = 0
    for col in range(nvar):
        # find pivot
        piv = None
        for rr in range(row, neq):
            if A[rr][col] != 0:
                piv = rr; break
        if piv is None:
            continue
        A[row], A[piv] = A[piv], A[row]
        b[row], b[piv] = b[piv], b[row]
        pivcol = A[row][col]
        # normalize
        for cc in range(col, nvar):
            A[row][cc] = A[row][cc] / pivcol
        b[row] = b[row] / pivcol
        for rr in range(neq):
            if rr != row and A[rr][col] != 0:
                f = A[rr][col]
                for cc in range(col, nvar):
                    A[rr][cc] = A[rr][cc] - f*A[row][cc]
                b[rr] = b[rr] - f*b[row]
        pivot_cols.append(col)
        row += 1
        if row == neq:
            break
    # check consistency
    for rr in range(row, neq):
        if b[rr] != 0:
            return None
    # solution: free vars = 0
    x = [F(0)]*nvar
    pivot_set = set(pivot_cols)
    for rr in range(row):
        # find the pivot col of this row (first nonzero)
        pc = None
        for cc in range(nvar):
            if A[rr][cc] != 0:
                pc = cc; break
        # x[pc] = b[rr] - sum_{cc>pc, free} ... but free=0, and other pivot cols already handled by reduced form
        # Since we did full Gauss-Jordan (eliminate above and below), each pivot row has 1 at pc and 0 elsewhere in pivot cols; non-pivot cols may be nonzero but those x are 0.
        x[pc] = b[rr]
    return x
