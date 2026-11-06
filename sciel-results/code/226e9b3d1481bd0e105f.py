    j = C_int
    for i in range(n - 1, -1, -1):
        if choice[i, j]:
            return i
        # else: item i is not in the solution, j remains unchanged
