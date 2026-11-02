def _polish(rows, n_rows, k):
    # try to reduce max entry
    rows = [list(r) for r in rows]
    # compute used diffs
    def diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                s.add(row[j]-row[i])
        return s
    improved = True
    while improved and time_left() > 10:
        improved = False
        for r in range(n_rows):
            row = rows[r]
            # try to reduce each mark from last to first
            for p in range(len(row)-1, 0, -1):
                old = row[p]
                # find smallest v in (row[p-1], old) such that row remains Golomb and no global conflict
                for v in range(row[p-1]+1, old):
                    # tentatively set
                    row[p] = v
                    # check row Golomb
                    rd = diffs(row)
                    if len(rd) != len(row)*(len(row)-1)//2:
                        row[p] = old
                        continue
                    # check global: all other rows' diffs + this row's diffs distinct
                    # compute global used excluding this row
                    ok = True
                    all_d = set(rd)
                    for r2 in range(n_rows):
                        if r2 == r: continue
                        d2 = diffs(rows[r2])
                        if all_d & d2:
                            ok = False; break
                    if ok:
                        improved = True
                        break
                    row[p] = old
    return rows
