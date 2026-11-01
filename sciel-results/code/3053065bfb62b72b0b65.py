def all_row_maxes(rows):
    if not rows:
        return 0
    return max(row_max(r) for r in rows)
