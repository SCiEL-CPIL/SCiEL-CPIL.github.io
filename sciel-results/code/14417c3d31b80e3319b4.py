    def can_add(S, x):
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd(a, x)
            for j in range(i + 1, len(S)):
                b = S[j]
                g_bx = gcd(b, x)
                if g_ax == g_bx:
                    if gcd(a, b) == g_ax:
                        return False
        return True
