            simplex = [x0.copy() for _ in range(dim + 1)]
            h = 0.5
            for i in range(dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += h
