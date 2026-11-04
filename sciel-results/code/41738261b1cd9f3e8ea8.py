    def strassen_block(ai1, ai2, aj1, aj2, bj1, bj2, bk1, bk2, ci1, ci2, ck1, ck2):
        nonlocal col
        # M1
        U[ai1*m+aj1][col] = 1
        U[ai2*m+aj2][col] = 1
        V[bj1*p+bk1][col] = 1
        V[bj2*p+bk2][col] = 1
        W[ck1*n+ci1][col] = 1
        W[ck2*n+ci2][col] = 1
        col += 1
        # M2
        U[ai2*m+aj1][col] = 1
        U[ai2*m+aj2][col] = 1
        V[bj1*p+bk1][col] = 1
        W[ck1*n+ci2][col] = 1
        col += 1
        # M3
        U[ai1*m+aj1][col] = 1
        V[bj1*p+bk2][col] = 1
        V[bj2*p+bk2][col] = -1
        W[ck2*n+ci1][col] = 1
        col += 1
        # M4
        U[ai2*m+aj2][col] = 1
        V[bj2*p+bk1][col] = 1
        V[bj1*p+bk1][col] = -1
        W[ck1*n+ci2][col] = 1
        col += 1
        # M5
        U[ai1*m+aj1][col] = 1
        U[ai1*m+aj2][col] = 1
        V[bj2*p+bk2][col] = 1
        W[ck2*n+ci1][col] = 1
        col += 1
        # M6
        U[ai2*m+aj1][col] = 1
        U[ai1*m+aj1][col] = -1
        V[bj1*p+bk1][col] = 1
        V[bj1*p+bk2][col] = 1
        W[ck2*n+ci2][col] = 1
        col += 1
        # M7
        U[ai1*m+aj2][col] = 1
        U[ai2*m+aj2][col] = -1
        V[bj2*p+bk1][col] = 1
        V[bj2*p+bk2][col] = 1
        W[ck1*n+ci2][col] = 1
        col += 1
