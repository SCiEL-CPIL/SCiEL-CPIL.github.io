        def _chvatal_automaton(N):
            t0 = [0] * N
            t1 = [0] * N
            for i in range(N):
                if i == 0:
                    t0[i] = 0
                    t1[i] = 1
                else:
                    t0[i] = i - 1
                    t1[i] = i
            return [t0, t1]
        