        is_distinct = True
        for k in range(12):
            if k != i and points[k] == [nx, ny]:
                is_distinct = False
                break
        if not is_distinct:
            continue
