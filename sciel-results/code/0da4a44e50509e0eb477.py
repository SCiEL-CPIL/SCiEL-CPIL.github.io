    def add_strassen(i0, j0, k0):
        nonlocal col
        # A indices: (i0+di, j0+dj) -> (i0+di)*m + (j0+dj)
        # B indices: (j0+dj, k0+dk) -> (j0+dj)*p + (k0+dk)
        # C indices: (k0+dk, i0+di) -> (k0+dk)*n + (i0+di)
        
        # A entries
        A00 = (i0+0)*m + (j0+0)
        A01 = (i0+0)*m + (j0+1)
        A10 = (i0+1)*m + (j0+0)
        A11 = (i0+1)*m + (j0+1)
        
        # B entries
        B00 = (j0+0)*p + (k0+0)
        B01 = (j0+0)*p + (k0+1)
        B10 = (j0+1)*p + (k0+0)
        B11 = (j0+1)*p + (k0+1)
        
        # C entries (trace convention: k*n + i)
        C00 = (k0+0)*n + (i0+0)
        C10 = (k0+0)*n + (i0+1)
        C01 = (k0+1)*n + (i0+0)
        C11 = (k0+1)*n + (i0+1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        W[C11][col] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] += -1
        W[C00][col] += -1
        W[C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += -1
        W[C11][col] += 1
        col += 1
