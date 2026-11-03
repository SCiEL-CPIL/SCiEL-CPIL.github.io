<<<<<<< SEARCH
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows
=======
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        # Try to improve the greedy solution by a simple local search?
        # Or just return it. The backtracker will handle optimization.
        return rows
>>>>>>> REPLACE
