"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the nm x mp x pn tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Index mapping:
    # a: A[i,j] -> i*3+j
    #   a0=A[0,0], a1=A[0,1], a2=A[0,2], a3=A[1,0], a4=A[1,1], a5=A[1,2]
    # b: B[j,k] -> j*3+k
    #   b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2],
    #   b6=B[2,0], b7=B[2,1], b8=B[2,2]
    # c: C[i,k] -> k*2+i (Column-major for C)
    #   c0=C[0,0], c1=C[1,0], c2=C[0,1], c3=C[1,1], c4=C[0,2], c5=C[1,2]
    
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    # A1 = [[a0, a1], [a3, a4]]
    # B1_left = [[b0, b1], [b3, b4]]
    # Output: C[0,0]=c0, C[1,0]=c1, C[0,1]=c2, C[1,1]=c3
    
    # M1 = (a0+a4)*(b0+b4) -> C[0,0] + C[1,0] + C[0,1] + C[1,1] ... wait, Strassen outputs are sums.
    # Standard Strassen for 2x2:
    # C00 = M1 - M2 + M4 - M5 ? No.
    # Let's use the standard Strassen formulas mapped to indices.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M4 - M5 ? 
    # Actually:
    # C00 = M1 - M2 + M4 - M5 is incorrect.
    # C00 = M1 - M2 + M4 - M5 ?
    # Let's check the terms provided in the inspiration which worked.
    # Term 0: M1 = (a0+a4)*(b0+b4) -> W[0]=1, W[3]=1. This contributes to C[0,0] and C[1,0].
    # Term 1: M2 = (a3+a4)*b0 -> W[1]=1, W[3]=-1. Contributes to C[1,0] and C[1,1].
    # Term 2: M3 = a0*(b1-b4) -> W[2]=1, W[3]=1. Contributes to C[0,1] and C[1,1].
    # Term 3: M4 = a4*(b3-b0) -> W[0]=1, W[1]=1. Contributes to C[0,0] and C[1,0].
    # Term 4: M5 = (a0+a1)*b4 -> W[0]=-1, W[2]=1. Contributes to C[0,0] and C[0,1].
    # Term 5: M6 = (a3-a0)*(b0+b1) -> W[3]=1. Contributes to C[1,1].
    # Term 6: M7 = (a1-a4)*(b3+b4) -> W[0]=1. Contributes to C[0,0].
    
    # Let's verify C[0,0] (c0):
    # From T0: +1
    # From T3: +1
    # From T4: -1
    # From T6: +1
    # Sum: 1 + 1 - 1 + 1 = 2.
    # This is wrong. C[0,0] should be A[0,0]B[0,0] + A[0,1]B[1,0].
    # My Strassen mapping in the inspiration might have been for a different output layout or I copied it blindly.
    # Let's re-derive or use a known correct 7-term Strassen decomposition for 2x2.
    
    # Correct Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b00)
    # m4 = a11(b01-b10)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    #
    # c00 = m1 - m2 + m4 - m5  <-- Wait, standard is:
    # c00 = m1 - m2 + m4 - m5 ?
    # Let's check:
    # m1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
    # m2 = (a10+a11)b00 = a10b00 + a11b00
    # m4 = a11(b01-b10) = a11b01 - a11b10
    # m5 = (a00+a01)b11 = a00b11 + a01b11
    # m1 - m2 + m4 - m5 = a00b00 + a00b11 + a11b00 + a11b11 - a10b00 - a11b00 + a11b01 - a11b10 - a00b11 - a01b11
    # = a00b00 + a11b11 - a10b00 + a11b01 - a11b10 - a01b11
    # This is not c00 = a00b00 + a01b10.
    
    # The standard formulas are:
    # c00 = m1 - m2 + m4 - m5 is WRONG.
    # c00 = m1 - m2 + m4 - m5 ... no.
    # c00 = m1 - m2 + m4 - m5 ?
    # Let's look up the exact Strassen reconstruction.
    # C00 = M1 - M2 + M4 - M5 ?
    # Actually:
    # C00 = M1 - M2 + M4 - M5 is a common confusion.
    # Correct:
    # C00 = M1 - M2 + M4 - M5 ?
    # Let's just use the terms from the "current program" which were verified to work.
    # The current program uses:
    # Term 0: U[0,4]=1, V[0,4]=1, W[0,3]=1  => (A00+A11)(B00+B11) -> C00+C10
    # Term 1: U[3,4]=1, V[0]=1, W[1,3]=1,-1 => (A10+A11)B00 -> C10-C11
    # Term 2: U[0]=1, V[1,4]=1,-1, W[2,3]=1,1 => A00(B10-B00) -> C01+C11
    # Term 3: U[4]=1, V[3,0]=1,-1, W[0,1]=1,1 => A11(B01-B00) -> C00+C10
    # Term 4: U[0,1]=1, V[4]=1, W[0,2]=-1,1 => (A00+A01)B11 -> -C00+C01
    # Term 5: U[3,0]=1,-1, V[0,1]=1,1, W[3]=1 => (A10-A00)(B00+B01) -> C11
    # Term 6: U[1,4]=1,-1, V[3,4]=1,1, W[0]=1 => (A01-A11)(B10+B11) -> C00
    
    # Let's re-verify C00 (c0) with these:
    # T0: +1
    # T3: +1
    # T4: -1
    # T6: +1
    # Sum: 1+1-1+1 = 2.
    # Why did the previous program pass?
    # Maybe my manual sum is wrong or the indices are different.
    # a0=A[0,0], a4=A[1,1].
    # b0=B[0,0], b4=B[1,1].
    # T0: (a0+a4)(b0+b4) = a0b0 + a0b4 + a4b0 + a4b4.
    # T3: a4(b3-b0) = a4b3 - a4b0. (b3=B[1,0]).
    # T4: (a0+a1)b4 = a0b4 + a1b4. (a1=A[0,1]).
    # T6: (a1-a4)(b3+b4) = a1b3 + a1b4 - a4b3 - a4b4.
    # Sum for C00:
    # a0b0 (from T0)
    # a0b4 (from T0) - a0b4 (from T4? No, T4 has -1 for W[0] which is C00. So -a0b4)
    # a4b0 (from T0) - a4b0 (from T3)
    # a4b4 (from T0) - a4b4 (from T6)
    # a1b4 (from T4) + a1b4 (from T6? No, T6 W[0] is +1. So +a1b4)
    # a1b3 (from T6)
    # a4b3 (from T3) - a4b3 (from T6)
    # Result: a0b0 + a1b4 + a1b3.
    # This is NOT C00 = a0b0 + a1b3 (if B is 2x2 left part, B10=b3).
    # Wait, C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # In my mapping:
    # A[0,0]=a0, A[0,1]=a1.
    # B[0,0]=b0, B[1,0]=b3.
    # So C00 should be a0b0 + a1b3.
    # My sum was a0b0 + a1b4 + a1b3.
    # There is an extra a1b4.
    # Where did a1b4 come from?
    # T4: (a0+a1)b4. W[0]=-1. So -a0b4 - a1b4.
    # T6: (a1-a4)(b3+b4). W[0]=1. So +a1b3 + a1b4 - a4b3 - a4b4.
    # Ah, T4 W[0] is -1.
    # So -a1b4 from T4 and +a1b4 from T6 cancel out.
    # So sum is a0b0 + a1b3. Correct!
    
    # So the terms in the current program ARE correct.
    
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    # C[0,2]=c4, C[1,2]=c5
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] = a0b2 + a1b5
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] = a3b2 + a4b5
    set_term(7, {0:1}, {2:1}, {4:1})   # a0b2 -> c4
    set_term(8, {1:1}, {5:1}, {4:1})   # a1b5 -> c4
    set_term(9, {3:1}, {2:1}, {5:1})   # a3b2 -> c5
    set_term(10, {4:1}, {5:1}, {5:1})  # a4b5 -> c5
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    # A[0,2]=a2, A[1,2]=a5.
    # B[2,0]=b6, B[2,1]=b7, B[2,2]=b8.
    # C[0,0] += a2b6 -> c0
    # C[0,1] += a2b7 -> c2
    # C[0,2] += a2b8 -> c4
    # C[1,0] += a5b6 -> c1
    # C[1,1] += a5b7 -> c3
    # C[1,2] += a5b8 -> c5
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    # Verify
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
