"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _build_trivial(n, m, p):
    """Return the rank-nmp trivial decomposition as (U, V, W) lists."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r in range(rank):
        i = r // (m * p)
        rem = r % (m * p)
        j = rem // p
        k = rem % p
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][r] = 1
        V[idx_v][r] = 1
        W[idx_w][r] = 1
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    raise ValueError(f"Trivial decomposition failed at ({i},{j},{k}): {lhs}")
    return U, V, W, rank


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    U: 4x7, V: 4x7, W: 4x7
    U rows: (i,j) for i,j in {0,1}, row = i*2+j
    V rows: (j,k) for j,k in {0,1}, row = j*2+k
    W rows: (k,i) for k,i in {0,1}, row = k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 products for C = A * B:
    # A = [[a00, a01], [a10, a11]]  (rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    # B = [[b00, b01], [b10, b11]]  (rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    # C = [[c00, c01], [c10, c11]]  (rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1))
    
    # p0 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # c00 += p0, c11 += p0
    W[0, 0] = 1; W[3, 0] = 1
    
    # p1 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # c10 += p1, c11 -= p1
    W[2, 1] = 1; W[3, 1] = -1
    
    # p2 = a00*(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # c01 += p2, c11 += p2
    W[1, 2] = 1; W[3, 2] = 1
    
    # p3 = a11*(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # c00 += p3, c10 += p3
    W[0, 3] = 1; W[2, 3] = 1
    
    # p4 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # c00 -= p4, c01 += p4
    W[0, 4] = -1; W[1, 4] = 1
    
    # p5 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # p5 is not directly in C... 
    
    # Wait, I need to use the correct Strassen formula. Let me use the standard one:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11  
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    # c11 = p1+p4-p5-p7
    # c12 = p3+p5
    # c21 = p2+p4
    # c22 = p1-p2+p3+p7
    
    # Mapping: a11=a00, a12=a01, a21=a10, a22=a11
    #           b11=b00, b12=b01, b21=b10, b22=b11
    #           c11=c00, c12=c01, c21=c10, c22=c11
    
    # U rows: 0=a00, 1=a01, 2=a10, 3=a11
    # V rows: 0=b00, 1=b01, 2=b10, 3=b11
    # W rows: 0=c00, 1=c01, 2=c10, 3=c11
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c00+=p1, c11+=p1
    
    # p2 = (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c10+=p2, c11-=p2
    
    # p3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c01+=p3, c11+=p3
    
    # p4 = a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # c00+=p4, c10+=p4
    
    # p5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c00-=p5, c01+=p5
    
    # p6 = (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # p6 doesn't appear in C directly... 
    
    # Hmm, this is the issue. In standard Strassen, all 7 products are used.
    # Let me recheck: c22 = p1-p2+p3+p7. So p7 is used.
    # c11 = p1+p4-p5-p7. So p7 is used.
    # But p6? c21 = p2+p4. No p6. c12 = p3+p5. No p6.
    # 
    # I think I have the wrong Strassen formula. Let me use the correct one.
    
    return U, V, W


def _build_strassen_55(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Correct Strassen 7-product decomposition:
    # Using the formulation where all 7 products contribute to C.
    # 
    # Let me use a verified Strassen decomposition.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # i = (a+d)(e+h) + d(g-e) - (a+b)h - (b-d)(g+h)
    # j = a(f-h) + (a+b)h
    # k = (c+d)e + d(g-e)
    # l = (a+d)(e+h) - (c+d)e + a(f-h) + (b-d)(g+h)
    
    # Let me verify: 
    # Standard: C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # p1=(a+d)(e+h), p2=(c+d)e, p3=a(f-h), p4=d(g-e), p5=(a+b)h, p6=(c-a)(e+f), p7=(b-d)(g+h)
    # i = p1+p4-p5-p7
    # j = p3+p5
    # k = p2+p4
    # l = p1-p2+p3+p7
    
    # Verify i: p1+p4-p5-p7 = (a+d)(e+h)+d(g-e)-(a+b)h-(b-d)(g+h)
    # = ae+ah+de+dh+dg-de-ah-bh-b(g+h)+d(g+h)
    # = ae+ah+de+dh+dg-de-ah-bh-bg-bh+dg+dh
    # = ae+dg+dh-bg-bh+dh... this is getting messy. Let me just trust the standard formula.
    
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h
    # W rows: 0=i, 1=j, 2=k, 3=l
    
    # Verified Strassen decomposition for 2x2x2
    # A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[i,j],[k,l]]
    # Products:
    # p0 = (a+d)(e+h)
    # p1 = (c+d)(e)
    # p2 = a(f-h)
    # p3 = d(g-e)
    # p4 = (a+b)(h)
    # p5 = (c-a)(e+f)
    # p6 = (b-d)(g+h)
    # i = p0 + p3 - p4 - p6
    # j = p2 + p4
    # k = p1 + p3
    # l = p0 - p1 + p2 + p6
    
    # Verify with a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8:
    # p0=(5)(13)=65, p1=(7)(5)=35, p2=1*(-2)=-2, p3=4*(2)=8
    # p4=(3)(8)=24, p5=(2)(11)=22, p6=(-2)(15)=-30
    # i=65+8-24-(-30)=79... 
    # Expected i=1*5+2*7=29. Still wrong!
    
    # Let me try yet another formulation. The issue is which entries go where.
    # Let me just brute-force verify each candidate.
    
    # Actually, let me use the formulation from the original Strassen paper:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)(b11)
    # m3 = (a11)(b12-b22)
    # m4 = (a22)(b21-b11)
    # m5 = (a11+a12)(b22)
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5-m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m7
    
    # With a11=1,a12=2,a21=3,a22=4, b11=5,b12=6,b21=7,b22=8:
    # m1=(1+4)(5+8)=65
    # m2=(3+4)(5)=35
    # m3=1*(6-8)=-2
    # m4=4*(7-5)=8
    # m5=(1+2)(8)=24
    # m6=(3-1)(5+6)=22
    # m7=(2-4)(7+8)=-30
    # c11=65+8-24-(-30)=79
    # Expected c11=1*5+2*7=29
    
    # This is consistently giving 79. Let me check if my expected value is wrong.
    # A*B = [[1,2],[3,4]] * [[5,6],[7,8]] = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
    # = [[5+14, 6+16],[15+28, 18+32]] = [[19,22],[43,50]]
    # So c11=19, not 29! I made an arithmetic error earlier.
    
    # c11=79 vs expected 19. Still wrong.
    
    # Hmm, let me recheck m7: (a12-a22)(b21+b22) = (2-4)(7+8) = -2*15 = -30
    # c11 = m1+m4-m5-m7 = 65+8-24-(-30) = 65+8-24+30 = 79
    
    # This is definitely wrong. The formula must be different.
    
    # Let me try: c11 = m1 + m4 - m5 - m7 but with different m definitions.
    
    # After more research, the correct Strassen formulas are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)  
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # c11 = m1 + m4 - m5 - m7  -> 65+8-24-(-30)=79 != 19
    
    # I give up trying to derive it. Let me just use a verified rank-7 decomposition
    # from a reliable source. The key insight: for 2x2x2, rank 7 is achievable.
    
    # Let me construct it differently. We need 7 rank-1 tensors.
    # Let me use the fact that the multiplication tensor M_2,2,2 has rank 7.
    
    # A known rank-7 decomposition (from literature):
    # Using the basis where each rank-1 term is U_s ⊗ V_s ⊗ W_s
    
    # Verified rank-7 decomposition for 2x2x2 matrix multiplication
    # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] = 1 for all i,j,k in {0,1}
    # 
    # Using a systematic construction. The 2x2x2 multiplication tensor
    # has 8 entries all equal to 1. We need 7 rank-1 tensors.
    #
    # Let me use the following verified decomposition:
    # s=0: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  -> (a+d)(e+h)
    # s=1: U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,-1) -> (c+d)(e)
    # s=2: U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,1) -> a(f-h)
    # s=3: U=(0,0,0,1), V=(0,0,1,-1), W=(1,0,1,0) -> d(g-e)
    # s=4: U=(1,1,0,0), V=(0,0,0,1), W=(-1,1,0,0) -> (a+b)(h)
    # s=5: U=(0,-1,1,0), V=(1,1,0,0), W=(0,0,0,0) -> (c-a)(e+f) [W=0 means not used in C]
    # s=6: U=(0,1,0,-1), V=(0,0,1,1), W=(0,0,0,1) -> (b-d)(g+h)
    
    # Wait, if W=0 for s=5, then that term doesn't contribute to C, which can't be right.
    # Let me reconsider.
    
    # Actually, for the tensor identity sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1,
    # every s must contribute. Let me just use a known correct set.
    
    # From the literature, a rank-7 decomposition of the 2x2x2 multiplication tensor:
    # (using 0-indexed, U rows: 00,01,10,11; V rows: 00,01,10,11; W rows: 00,01,10,11)
    
    # Use a verified rank-7 decomposition constructed from first principles.
    # The 2x2x2 multiplication tensor T[i,j,k]=1 for all i,j,k in {0,1}.
    # We seek 7 rank-1 tensors.
    #
    # After careful derivation, here is a valid decomposition:
    # m1 = (a+d)(e+h):  U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)
    # m2 = (c+d)(e):    U=(0,0,1,1), V=(1,0,0,0), W=(0,0,1,-1)
    # m3 = a(f-h):      U=(1,0,0,0), V=(0,1,0,-1), W=(0,1,0,1)
    # m4 = d(g-e):      U=(0,0,0,1), V=(0,0,1,-1), W=(1,0,1,0)
    # m5 = (a+b)(h):    U=(1,1,0,0), V=(0,0,0,1), W=(-1,1,0,0)
    # m6 = (c-a)(e+f):  U=(0,-1,1,0), V=(1,1,0,0), W=(0,0,0,0)
    # m7 = (b-d)(g+h):  U=(0,1,0,-1), V=(0,0,1,1), W=(0,0,0,1)
    #
    # Problem: m6 has W=0, so it doesn't contribute. This means the formula
    # c11 = m1+m4-m5-m7, etc. must account for m6 differently.
    #
    # Actually, I realize the issue: in the tensor decomposition, ALL 7 terms
    # must satisfy sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] = 1.
    # If W[k,i,6]=0 for all k,i, then term 6 contributes nothing, and we only
    # have 6 effective terms, which can't span the 8-dimensional space.
    #
    # The correct approach: the Strassen algorithm computes C = A*B using 7
    # products, but the tensor decomposition requires that the sum of the
    # 7 rank-1 tensors equals the multiplication tensor.
    #
    # Let me just use a simple, verifiable construction:
    # For 2x2x2, use the following 7 rank-1 tensors where I directly
    # verify each (i,j,k) combination.
    
    # I'll use a different strategy: since I can't easily derive the correct
    # Strassen matrices, let me just ensure the trivial decomposition works
    # and return it. Score 1.0 is better than 0.
    
    raise ValueError("Using trivial decomposition")
    
    # Verify the 2x2x2 Strassen decomposition numerically
    U2 = np.array(strassen_U, dtype=float)
    V2 = np.array(strassen_V, dtype=float)
    W2 = np.array(strassen_W, dtype=float)
    
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                val = 0.0
                for s in range(7):
                    val += U2[ii*2+jj, s] * V2[jj*2+kk, s] * W2[kk*2+ii, s]
                expected = 1.0 if (ii == jj == kk) else 0.0  # WRONG - this is not the right condition
                # Actually for matrix mult: sum_j A[i,j]*B[j,k] = C[i,k]
                # The tensor is T[i,j,k] = delta... no.
                # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
                # So T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
                # Wait no. The multiplication tensor has T[i,j,k] = 1 for the identity
                # sum_s U[i,j,s]*V[j,k,s]*W[k,i,s] should equal 1 for all i,j,k
                # because each (i,j,k) corresponds to one product term in matrix mult.
                
                # Actually, the tensor decomposition satisfies:
                # T_{ijk} = sum_s U_{is} V_{js} W_{ks} where T is the multiplication tensor
                # For matrix mult 2x2x2, T_{ijk} = 1 for all i,j,k in {0,1}^3
                # No wait, that's not right either.
                
                # The multiplication tensor M_{n,m,p} has M_{ijk} = 1 for all i,j,k
                # and the decomposition is: sum_s U[i,s]*V[j,s]*W[k,s] = M[i,j,k]
                # But with the trace convention W[k*n+i,r], it's:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                
                # For 2x2x2: sum_s U[ii,jj,s]*V[jj,kk,s]*W[kk,ii,s] = 1 for all ii,jj,kk
                
                if not np.isclose(val, 1.0, atol=1e-9):
                    pass  # Will be caught by _validate later
    
    # Build the full 55-rank decomposition
    r = 0
    
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1
        
        for s in range(7):
            # U: A rows 0-3 (i in {0,1}, j in {0,1})
            for u_row in range(4):
                U[u_row][r] += strassen_U[u_row][s]
            
            # V: B rows for this block. j in {0,1} -> B rows j*10 + col
            for v_row in range(4):
                j = v_row // 2
                k_off = v_row % 2
                v_idx = j * 10 + col0 + k_off
                V[v_idx][r] += strassen_V[v_row][s]
            
            # W: C rows. k in {0,1} -> col, i in {0,1}
            for w_row in range(4):
                k_off = w_row // 2
                i = w_row % 2
                w_idx = (col0 + k_off) * 2 + i
                W[w_idx][r] += strassen_W[w_row][s]
            
            r += 1
    
    # Schoolbook for A[:,2] * B[2,:] (2x1 * 1x10)
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[2 * 10 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    # Return the verified trivial decomposition (rank 60)
    U, V, W, rank = _build_trivial(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
