        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
            for j in range(m):  # <-- This loop is outside the k loop but inside i? No, indentation is weird.
                for k in range(p):
