def verify(U,V,W):
    # round to half-integers
    Uh = [[Fraction(round(2*U[i][j]))/2 for j in range(len(U[0]))] for i in range(len(U))]
    ...
    # check all (i,j,jp,k)
    for i in range(2):
        for j in range(3):
            for jp in range(3):
                for k in range(3):
                    s = sum(Uh[i*3+j][c]*Vh[jp*3+k][c]*Wh[k*2+i][c] for c in range(r))
                    if s != (1 if j==jp else 0): return None
    return (Uh,Vh,Wh)
