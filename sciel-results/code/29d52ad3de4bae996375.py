import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    try:
        T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d, optimize=True)
    except Exception:
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    if not np.allclose(T, 1.0, atol=tol, rtol=0.0):
        return False
    return True

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    if not _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Attempt to construct a rank 78 decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for smaller blocks.
    
    def _add_strassen_2x2x2(U, V, W, u_base, v_base, w_base, col):
        # u_base is the global U index for local (0,0)
        # v_base is the global V index for local (0,0)
        # w_base is the global W index for local (0,0)
        # Local indices:
        # A: (0,0)->u_base+0, (0,1)->u_base+1, (1,0)->u_base+2, (1,1)->u_base+3
        # B: (0,0)->v_base+0, (0,1)->v_base+1, (1,0)->v_base+2, (1,1)->v_base+3
        # C: (0,0)->w_base+0, (0,1)->w_base+1, (1,0)->w_base+2, (1,1)->w_base+3
        
        # P1 = (A00+A11)(B00+B11)
        U[u_base+0, col] += 1
        U[u_base+3, col] += 1
        V[v_base+0, col] += 1
        V[v_base+3, col] += 1
        W[w_base+0, col] += 1
        W[w_base+3, col] += 1
        col += 1
        
        # P2 = (A10+A11)B00
        U[u_base+2, col] += 1
        U[u_base+3, col] += 1
        V[v_base+0, col] += 1
        W[w_base+2, col] += 1
        W[w_base+3, col] += 1
        col += 1
        
        # P3 = A00(B01-B11)
        U[u_base+0, col] += 1
        V[v_base+1, col] += 1
        V[v_base+3, col] += -1
        W[w_base+1, col] += 1
        W[w_base+3, col] += 1
        col += 1
        
        # P4 = A11(B10-B00)
        U[u_base+3, col] += 1
        V[v_base+2, col] += 1
        V[v_base+0, col] += -1
        W[w_base+0, col] += 1
        W[w_base+2, col] += 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[u_base+0, col] += 1
        U[u_base+1, col] += 1
        V[v_base+3, col] += 1
        W[w_base+0, col] += -1
        W[w_base+1, col] += -1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[u_base+2, col] += 1
        U[u_base+0, col] += -1
        V[v_base+0, col] += 1
        V[v_base+1, col] += 1
        W[w_base+3, col] += 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[u_base+1, col] += 1
        U[u_base+3, col] += -1
        V[v_base+2, col] += 1
        V[v_base+3, col] += 1
        W[w_base+1, col] += 1
        W[w_base+3, col] += -1
        col += 1
        
        return col

    def _add_schoolbook_2x2x1(U, V, W, u_base, v_base, w_base, col):
        # A (2x2), B (2x1), C (2x1)
        # C00 = A00*B00 + A01*B10
        # C10 = A10*B00 + A11*B10
        # U indices: 0,1,2,3
        # V indices: 0,1
        # W indices: 0,1
        
        # Term 1: A00*B00
        U[u_base+0, col] += 1
        V[v_base+0, col] += 1
        W[w_base+0, col] += 1
        col += 1
        # Term 2: A01*B10
        U[u_base+1, col] += 1
        V[v_base+1, col] += 1
        W[w_base+0, col] += 1
        col += 1
        # Term 3: A10*B00
        U[u_base+2, col] += 1
        V[v_base+0, col] += 1
        W[w_base+1, col] += 1
        col += 1
        # Term 4: A11*B10
        U[u_base+3, col] += 1
        V[v_base+1, col] += 1
        W[w_base+1, col] += 1
        col += 1
        return col

    def _add_schoolbook_1x2x1(U, V, W, u_base, v_base, w_base, col):
        # A (1x2), B (2x1), C (1x1)
        # C00 = A00*B00 + A01*B10
        # U: 0,1
        # V: 0,1
        # W: 0
        
        U[u_base+0, col] += 1
        V[v_base+0, col] += 1
        W[w_base+0, col] += 1
        col += 1
        U[u_base+1, col] += 1
        V[v_base+1, col] += 1
        W[w_base+0, col] += 1
        col += 1
        return col

    def _add_schoolbook_1x2x2(U, V, W, u_base, v_base, w_base, col):
        # A (1x2), B (2x2), C (1x2)
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # U: 0,1
        # V: 0,1,2,3
        # W: 0,1
        
        # Term 1: A00*B00
        U[u_base+0, col] += 1
        V[v_base+0, col] += 1
        W[w_base+0, col] += 1
        col += 1
        # Term 2: A01*B10
        U[u_base+1, col] += 1
        V[v_base+2, col] += 1
        W[w_base+0, col] += 1
        col += 1
        # Term 3: A00*B01
        U[u_base+0, col] += 1
        V[v_base+1, col] += 1
        W[w_base+1, col] += 1
        col += 1
        # Term 4: A01*B11
        U[u_base+1, col] += 1
        V[v_base+3, col] += 1
        W[w_base+1, col] += 1
        col += 1
        return col

    def _add_schoolbook_2x2x2(U, V, W, u_base, v_base, w_base, col):
        # A (2x2), B (2x2), C (2x2)
        # 4 terms
        # U: 0,1,2,3
        # V: 0,1,2,3
        # W: 0,1,2,3
        
        # C00
        U[u_base+0, col] += 1
        V[v_base+0, col] += 1
        W[w_base+0, col] += 1
        col += 1
        # C01
        U[u_base+0, col] += 1
        V[v_base+1, col] += 1
        W[w_base+1, col] += 1
        col += 1
        # C10
        U[u_base+2, col] += 1
        V[v_base+2, col] += 1
        W[w_base+2, col] += 1
        col += 1
        # C11
        U[u_base+3, col] += 1
        V[v_base+3, col] += 1
        W[w_base+3, col] += 1
        col += 1
        return col

    # Construct Rank 78
    U2 = np.zeros((n * m, 78))
    V2 = np.zeros((m * p, 78))
    W2 = np.zeros((p * n, 78))
    col = 0
    
    # Block 1: i in {0,1}, k in {0,1,2} -> M_{2,4,3}
    # Split j into {0,1} and {2,3}
    # Sub 1a: i in {0,1}, j in {0,1}, k in {0,1,2} -> M_{2,2,3}
    # Split k into {0} and {1,2}
    # 1a1: i in {0,1}, j in {0,1}, k in {0} -> M_{2,2,1}
    # u_base = 0*4+0 = 0
    # v_base = 0*7+0 = 0
    # w_base = 0*3+0 = 0
    col = _add_schoolbook_2x2x1(U2, V2, W2, 0, 0, 0, col)
    
    # 1a2: i in {0,1}, j in {0,1}, k in {1,2} -> M_{2,2,2}
    # u_base = 0
    # v_base = 0*7+1 = 1
    # w_base = 1*3+0 = 3
    col = _add_strassen_2x2x2(U2, V2, W2, 0, 1, 3, col)
    
    # Sub 1b: i in {0,1}, j in {2,3}, k in {0,1,2} -> M_{2,2,3}
    # 1b1: i in {0,1}, j in {2,3}, k in {0} -> M_{2,2,1}
    # u_base = 0*4+2 = 2
    # v_base = 2*7+0 = 14
    # w_base = 0*3+0 = 0
    col = _add_schoolbook_2x2x1(U2, V2, W2, 2, 14, 0, col)
    
    # 1b2: i in {0,1}, j in {2,3}, k in {1,2} -> M_{2,2,2}
    # u_base = 2
    # v_base = 2*7+1 = 15
    # w_base = 1*3+0 = 3
    col = _add_strassen_2x2x2(U2, V2, W2, 2, 15, 3, col)
    
    # Block 2: i in {0,1}, k in {3,4,5,6} -> M_{2,4,4}
    # Split j into {0,1} and {2,3}
    # Sub 2a: i in {0,1}, j in {0,1}, k in {3,4,5,6} -> M_{2,2,4}
    # Split k into {3,4} and {5,6}
    # 2a1: i in {0,1}, j in {0,1}, k in {3,4} -> M_{2,2,2}
    # u_base = 0
    # v_base = 0*7+3 = 3
    # w_base = 3*3+0 = 9
    col = _add_strassen_2x2x2(U2, V2, W2, 0, 3, 9, col)
    
    # 2a2: i in {0,1}, j in {0,1}, k in {5,6} -> M_{2,2,2}
    # u_base = 0
    # v_base = 0*7+5 = 5
    # w_base = 5*3+0 = 15
    col = _add_strassen_2x2x2(U2, V2, W2, 0, 5, 15, col)
    
    # Sub 2b: i in {0,1}, j in {2,3}, k in {3,4,5,6} -> M_{2,2,4}
    # 2b1: i in {0,1}, j in {2,3}, k in {3,4} -> M_{2,2,2}
    # u_base = 2
    # v_base = 2*7+3 = 17
    # w_base = 3*3+0 = 9
    col = _add_strassen_2x2x2(U2, V2, W2, 2, 17, 9, col)
    
    # 2b2: i in {0,1}, j in {2,3}, k in {5,6} -> M_{2,2,2}
    # u_base = 2
    # v_base = 2*7+5 = 19
    # w_base = 5*3+0 = 15
    col = _add_strassen_2x2x2(U2, V2, W2, 2, 19, 15, col)
    
    # Block 3: i in {2}, k in {0,1,2} -> M_{1,4,3}
    # Split j into {0,1} and {2,3}
    # Sub 3a: i in {2}, j in {0,1}, k in {0,1,2} -> M_{1,2,3}
    # Split k into {0} and {1,2}
    # 3a1: i in {2}, j in {0,1}, k in {0} -> M_{1,2,1}
    # u_base = 2*4+0 = 8
    # v_base = 0*7+0 = 0
    # w_base = 0*3+2 = 2
    col = _add_schoolbook_1x2x1(U2, V2, W2, 8, 0, 2, col)
    
    # 3a2: i in {2}, j in {0,1}, k in {1,2} -> M_{1,2,2}
    # u_base = 8
    # v_base = 0*7+1 = 1
    # w_base = 1*3+2 = 5
    col = _add_schoolbook_1x2x2(U2, V2, W2, 8, 1, 5, col)
    
    # Sub 3b: i in {2}, j in {2,3}, k in {0,1,2} -> M_{1,2,3}
    # 3b1: i in {2}, j in {2,3}, k in {0} -> M_{1,2,1}
    # u_base = 2*4+2 = 10
    # v_base = 2*7+0 = 14
    # w_base = 0*3+2 = 2
    col = _add_schoolbook_1x2x1(U2, V2, W2, 10, 14, 2, col)
    
    # 3b2: i in {2}, j in {2,3}, k in {1,2} -> M_{1,2,2}
    # u_base = 10
    # v_base = 2*7+1 = 15
    # w_base = 1*3+2 = 5
    col = _add_schoolbook_1x2x2(U2, V2, W2, 10, 15, 5, col)
    
    # Block 4: i in {2}, k in {3,4,5,6} -> M_{1,4,4}
    # Split j into {0,1} and {2,3}
    # Sub 4a: i in {2}, j in {0,1}, k in {3,4,5,6} -> M_{1,2,4}
    # Split k into {3,4} and {5,6}
    # 4a1: i in {2}, j in {0,1}, k in {3,4} -> M_{1,2,2}
    # u_base = 8
    # v_base = 0*7+3 = 3
    # w_base = 3*3+2 = 11
    col = _add_schoolbook_1x2x2(U2, V2, W2, 8, 3, 11, col)
    
    # 4a2: i in {2}, j in {0,1}, k in {5,6} -> M_{1,2,2}
    # u_base = 8
    # v_base = 0*7+5 = 5
    # w_base = 5*3+2 = 17
    col = _add_schoolbook_1x2x2(U2, V2, W2, 8, 5, 17, col)
    
    # Sub 4b: i in {2}, j in {2,3}, k in {3,4,5,6} -> M_{1,2,4}
    # 4b1: i in {2}, j in {2,3}, k in {3,4} -> M_{1,2,2}
    # u_base = 10
    # v_base = 2*7+3 = 17
    # w_base = 3*3+2 = 11
    col = _add_schoolbook_1x2x2(U2, V2, W2, 10, 17, 11, col)
    
    # 4b2: i in {2}, j in {2,3}, k in {5,6} -> M_{1,2,2}
    # u_base = 10
    # v_base = 2*7+5 = 19
    # w_base = 5*3+2 = 17
    col = _add_schoolbook_1x2x2(U2, V2, W2, 10, 19, 17, col)
    
    # Total columns should be 78
    if col == 78:
        if _verify_decomposition(U2, V2, W2, n, m, p):
            best_rank = 78
            best_U, best_V, best_W = U2, V2, W2
        else:
            # If verification fails, stick to schoolbook
            pass
    else:
        # If col != 78, something went wrong, stick to schoolbook
        pass

    # Check time
    if time.time() - t0 > 100:
        # Safety margin
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
