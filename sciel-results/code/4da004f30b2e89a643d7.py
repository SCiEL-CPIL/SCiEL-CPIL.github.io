        current_max = 0
        current_arg = 0
        for g in range(max_gap + 1):
            if best[i][g] > current_max:
                current_max = best[i][g]
                current_arg = g
            M[i][g] = current_max
            argM[i][g] = current_arg
