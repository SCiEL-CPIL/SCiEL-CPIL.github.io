    def calculate_threshold_fast(a_trans, b_trans):
        a_mask = [0] * (1 << N)
        b_mask = [0] * (1 << N)
        for s in range(1 << N):
            ma = 0
            mb = 0
            for i in range(N):
                if s & (1 << i):
                    ma |= 1 << a_trans[i]
                    mb |= 1 << b_trans[i]
            a_mask[s] = ma
            b_mask[s] = mb
        # ... BFS ...
    