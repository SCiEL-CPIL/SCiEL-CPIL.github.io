import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 4, 5
    
    def verify_numpy(U, V, W, rank):
        """Fast verification using numpy tensor contraction."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        return True
    
    def verify_exact(U, V, W, rank):
        """Exact verification with fractions to ensure correctness."""
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    # Strassen 2x2 terms (indices 0-3 map to (li, lj) pairs)
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    def build_rank36():
        """Standard rank-36: 2x(2*7 + 4) = 36"""
        rank = 36
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        for idx_u, val_u in u_t.items():
                            li, lj = idx_u // 2, idx_u % 2
                            U[li*m + (2*q + lj)][col] += val_u
                        for idx_v, val_v in v_t.items():
                            lj, lk = idx_v // 2, idx_v % 2
                            V[(2*q + lj)*p + (2*s + lk)][col] += val_v
                        for idx_w, val_w in w_t.items():
                            li, lk = idx_w // 2, idx_w % 2
                            W[(2*s + lk)*n + li][col] += val_w
                        col += 1
                else:
                    # 4 standard terms for 2x2 * 2x1
                    for idx_u in range(4):
                        li, lj = idx_u // 2, idx_u % 2
                        U[li*m + (2*q + lj)][col] = 1
                        V[(2*q + lj)*p + 4][col] = 1
                        W[4*n + li][col] = 1
                        col += 1
        assert col == rank
        return U, V, W
    
    def build_rank30():
        """
        Try rank-30 by using a combined approach:
        Split as (1+1) x 4 * 4 x (2+2+1)
        For each row i in {0,1} and each 2-col block s in {0,1}: 1x4 * 4x2 -> rank 4 (dot products)
        For each row i and last col: 1x4 * 4x1 -> rank 1
        Total: 2*(4+4+1) = 18? No, that's the number of outputs.
        
        Actually, 1x4 * 4x2: C[i,k] = sum_j A[i,j]*B[j,k] for k in block
        This is just 2 dot products of length 4, rank = 2 (one per output)
        Wait no, rank of 1x4*4x2 tensor: each output is a separate bilinear form.
        The tensor has 1*4*2 = 8 entries, but rank is at most 4 (since A has 4 entries).
        Actually rank = 2: C[i,0] = dot(a, b0), C[i,1] = dot(a, b1). These are 2 independent bilinear forms.
        So rank per (i, s-block) = 2.
        Total: 2 rows * (2 blocks of rank 2 + 1 block of rank 1) = 2*(4+1) = 10?
        
        Hmm, that doesn't seem right. Let me think again.
        For 1x4 * 4x2: the multiplication tensor T[i,j,k] = delta(i,0) for all j,k? No.
        T[i,j,k] = 1 if i=0 (only one row), so T[j,k] = 1 for all j in 0..3, k in 0..1.
        This is a 4x2 tensor of all ones. Its rank is 1! (It's outer product of [1,1,1,1] and [1,1])
        So rank(1x4*4x2) = 1.
        Similarly rank(1x4*4x1) = 1.
        Total: 2 rows * (2*1 + 1*1) = 2*3 = 6? That can't be right either.
        
        Wait, I'm confusing myself. The tensor for 1x4*4x2 multiplication:
        T[i,j,k] where i in {0}, j in {0,1,2,3}, k in {0,1}
        T[0,j,k] = 1 for all j,k.
        This is rank 1 (outer product of vector of 4 ones and vector of 2 ones).
        
        But we have 2 rows of A, so we need 2 such tensors. They share the same j,k structure but different i.
        Since i is fixed per row, the two tensors are:
        T0[j,k] = 1 (for row 0)
        T1[j,k] = 1 (for row 1)
        These are the same tensor! So we can share the same rank-1 term.
        But wait, the W factor has different i values. So:
        Term: U[j,r] * V[j,k,r] * W[k,i,r]
        For row 0: W[k,0,r], for row 1: W[k,1,r]
        If we use the same U and V, we need separate W for each i.
        So we can't fully share. Each row needs its own set of terms.
        
        Let me just try a direct construction with rank 30 using a hybrid approach.
        """
        # Alternative: use 2x4*4x5 = (2x2+2x2) * (2x2+2x2+2x1)
        # but combine the two 2x2*2x1 blocks (for s=2) across q values
        # For s=2: A_q (2x2) * B_2 (2x1) for q=0,1
        # C_0 = A_0 * B_2 (2x1 result)
        # C_1 = A_1 * B_2 (2x1 result)
        # Can we compute both with fewer than 8 terms?
        # C_0[0] = a00*b0 + a01*b1
        # C_0[1] = a10*b0 + a11*b1
        # C_1[0] = a20*b0 + a21*b1
        # C_1[1] = a30*b0 + a31*b1
        # That's 8 independent bilinear terms. Min rank = 8.
        # So no savings there.
        
        # Try different split: 2x4 = 2x(1+3), 4x5 = (1+3)x5
        # 2x1 * 1x5: rank 2 (each row dotted with the 1-col)
        # 2x3 * 3x5: use Strassen-like? 2x3*3x5... 
        # This is getting complicated. Let me just return rank 36.
        return None
    
    # Try rank-30 construction
    if time.time() < deadline - 10:
        r30 = build_rank30()
        if r30 is not None:
            U30, V30, W30 = r30
            if verify_numpy(U30, V30, W30, 30):
                if verify_exact(U30, V30, W30, 30):
                    return {"U": U30, "V": V30, "W": W30}
    
    # Rank-36 solution
    U36, V36, W36 = build_rank36()
    if verify_numpy(U36, V36, W36, 36):
        if verify_exact(U36, V36, W36, 36):
            return {"U": U36, "V": V36, "W": W36}
    
    # Schoolbook fallback
    rank_school = n * m * p
    best = {
        "U": [[0]*rank_school for _ in range(n*m)],
        "V": [[0]*rank_school for _ in range(m*p)],
        "W": [[0]*rank_school for _ in range(p*n)]
    }
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best["U"][i*m+j][c] = 1
                best["V"][j*p+k][c] = 1
                best["W"][k*n+i][c] = 1
                c += 1
    return best
