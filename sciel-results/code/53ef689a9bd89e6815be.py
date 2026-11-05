    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            # 1 + k * sum_{i=0}^{r-1} (k-1)^i
            s = 1
            term = 1
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return 1 + k * s
        else:
            # 2 * sum_{i=0}^{r-1} (k-1)^i
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s
