<<<<<<< SEARCH
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        base_k = b * 2 # global column offset for this block
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        r = base_rank
        # U: A00, A11
        U[0*m+0][r] = 1
        U[1*m+1][r] = 1
        # V: B00 (j=0, k=base_k), B11 (j=1, k=base_k+1)
        V[0*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        # W: C00 (k=base_k, i=0), C11 (k=base_k+1, i=1)
        # Note: M1 contributes to C00 and C11 with coeff 1
        W[base_k*n+0][r] = 1
        W[(base_k+1)*n+1][r] = 1
        
        # M2: (A10+A11)(B00) -> C10
        r = base_rank + 1
        # U: A10, A11
        U[1*m+0][r] = 1
        U[1*m+1][r] = 1
        # V: B00 (j=0, k=base_k)
        V[0*p+base_k][r] = 1
        # W: C10 (k=base_k, i=1)
        W[base_k*n+1][r] = 1
        
        # M3: (A00)(B01-B11) -> C01
        r = base_rank + 2
        # U: A00
        U[0*m+0][r] = 1
        # V: B01 (j=0, k=base_k+1), B11 (j=1, k=base_k+1)
        V[0*p+base_k+1][r] = 1
        V[1*p+base_k+1][r] = -1
        # W: C01 (k=base_k+1, i=0)
        W[(base_k+1)*n+0][r] = 1
        
        # M4: (A11)(B10-B00) -> C10
        r = base_rank + 3
        # U: A11
        U[1*m+1][r] = 1
        # V: B10 (j=1, k=base_k), B00 (j=0, k=base_k)
        V[1*p+base_k][r] = 1
        V[0*p+base_k][r] = -1
        # W: C10 (k=base_k, i=1)
        W[base_k*n+1][r] = 1
        
        # M5: (A00+A01)(B11) -> -C00, C01
        r = base_rank + 4
        # U: A00, A01
        U[0*m+0][r] = 1
        U[0*m+1][r] = 1
        # V: B11 (j=1, k=base_k+1)
        V[1*p+base_k+1][r] = 1
        # W: -C00 (k=base_k, i=0), C01 (k=base_k+1, i=0)
        W[base_k*n+0][r] = -1
        W[(base_k+1)*n+0][r] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        r = base_rank + 5
        # U: A10, A00
        U[1*m+0][r] = 1
        U[0*m+0][r] = -1
        # V: B00 (j=0, k=base_k), B01 (j=0, k=base_k+1)
        V[0*p+base_k][r] = 1
        V[0*p+base_k+1][r] = 1
        # W: C11 (k=base_k+1, i=1)
        W[(base_k+1)*n+1][r] = 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        r = base_rank + 6
        # U: A01, A11
        U[0*m+1][r] = 1
        U[1*m+1][r] = -1
        # V: B10 (j=1, k=base_k), B11 (j=1, k=base_k+1)
        V[1*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        # W: C00 (k=base_k, i=0)
        W[base_k*n+0][r] = 1
=======
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        base_k = b * 2 # global column offset for this block
        
        # M1: (A00+A11)(B00+B11)
        r = base_rank
        U[0*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        W[base_k*n+0][r] = 1
        W[(base_k+1)*n+1][r] = 1
        
        # M2: (A10+A11)(B00)
        r = base_rank + 1
        U[1*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+base_k][r] = 1
        W[base_k*n+0][r] = -1
        W[base_k*n+1][r] = 1
        
        # M3: (A00)(B01-B11)
        r = base_rank + 2
        U[0*m+0][r] = 1
        V[0*p+base_k+1][r] = 1
        V[1*p+base_k+1][r] = -1
        W[base_k*n+0][r] = 1
        W[base_k*n+1][r] = -1
        
        # M4: (A11)(B10-B00)
        r = base_rank + 3
        U[1*m+1][r] = 1
        V[1*p+base_k][r] = 1
        V[0*p+base_k][r] = -1
        W[base_k*n+0][r] = -1
        W[base_k*n+1][r] = 1
        
        # M5: (A00+A01)(B11)
        r = base_rank + 4
        U[0*m+0][r] = 1
        U[0*m+1][r] = 1
        V[1*p+base_k+1][r] = 1
        W[base_k*n+0][r] = 1
        W[(base_k+1)*n+0][r] = 1
        W[(base_k+1)*n+1][r] = -1
        
        # M6: (A10-A00)(B00+B01)
        r = base_rank + 5
        U[1*m+0][r] = 1
        U[0*m+0][r] = -1
        V[0*p+base_k][r] = 1
        V[0*p+base_k+1][r] = 1
        W[(base_k+1)*n+0][r] = -1
        W[base_k*n+1][r] = 1
        W[(base_k+1)*n+1][r] = 1
        
        # M7: (A01-A11)(B10+B11)
        r = base_rank + 6
        U[0*m+1][r] = 1
        U[1*m+1][r] = -1
        V[1*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        W[(base_k+1)*n+0][r] = 1
        W[(base_k+1)*n+1][r] = -1
>>>>>>> REPLACE
