<<<<<<< SEARCH
    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        S = set(S_init)
        for x in order:
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return S
=======
    def greedy_construct(S_init, order):
        """Greedy construction starting from S_init, adding elements in 'order'."""
        S = set(S_init)
        S_bm = set_to_bitmask(S)
        for x in order:
            if x in S:
                continue
            if can_add_bitmask(x, S_bm):
                S.add(x)
                S_bm[(x - 1) // 64] |= (1 << ((x - 1) % 64))
        return S
>>>>>>> REPLACE

