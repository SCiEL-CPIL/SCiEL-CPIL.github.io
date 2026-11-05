        def saturate(S_mask, S):
            for v in range(1, 257):
                if v not in S and not (mask[v] & S_mask):
                    S_mask |= (1 << v)
                    S.add(v)
            return S_mask, S
