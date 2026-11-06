    def can_add(a, S):
        if a >= N:
            return True
        a_sq = a * a
        min_x = (a_sq + (N - a) - 1) // (N - a)
        if min_x < a:
            for x in range(min_x, a):
                if a_sq % x == 0:
                    y = a_sq // x
                    b = a + x
                    c = a + y
                    if b in S and c in S:
                        return False
        return True
