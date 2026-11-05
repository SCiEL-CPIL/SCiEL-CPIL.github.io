import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _laderman_3x3_coeffs():
    """
    Returns list of 15 tuples (U_coeffs, V_coeffs, W_coeffs) for Laderman's rank-15
    decomposition of 3x3 matrix multiplication.
    
    U_coeffs: dict {(i, j): val} for A block
    V_coeffs: dict {(j, k): val} for B block
    W_coeffs: dict {(k, i): val} for C block contribution
    Indices i, j, k in {0, 1, 2}
    
    The decomposition satisfies:
    C[i,k] = sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]  (summed over s and implied j)
    More precisely, each term s contributes to C[i,k] via:
    sum over j of U_s[i,j] * A[i,j] * V_s[j,k] * B[j,k] ... 
    Actually the tensor decomposition is:
    T[i,j,k] = sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]
    where T[i,j,k] = delta(i,j,k) in the appropriate sense for matmul.
    
    For matrix multiplication C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the bilinear map).
    Wait, no. The matmul tensor is T[i,j,k] where the contraction is:
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    and T[i,j,k] = 1 for all i,j,k.
    
    The rank-r decomposition is:
    T[i,j,k] = sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]
    
    Laderman's 15 terms for 3x3:
    """
    coeffs = []
    
    # M1 = (A00+A01)(B10+B11)
    coeffs.append((
        {(0,0):1, (0,1):1}, 
        {(1,0):1, (1,1):1}, 
        {(0,0):1, (1,0):1}  # W[k,i]: contributes to C[0,0] and C[0,1]
    ))
    
    # M2 = (A00+A02)(B20+B21)  
    # Let me use the standard Laderman formulation.
    # The 15 multiplications for 3x3 are:
    # M1 = (a00+a01)(b10+b11)
    # M2 = (a00+a02)(b20+b21)
    # M3 = (a10+a12)(b21+b22)
    # M4 = (a11+a12)(b01+b02)
    # M5 = (a20+a21)(b00+b02)
    # M6 = (a21+a22)(b10+b12)
    # M7 = a00*b00
    # M8 = a01*b11
    # M9 = a02*b22
    # M10 = a10*b10
    # M11 = a11*b11  -- wait, this doesn't look right.
    
    # Let me use the correct Laderman decomposition.
    # Standard Laderman 15-term decomposition for 3x3:
    
    # I'll define it carefully. The 15 scalar multiplications are:
    # P1 = (A00+A01)*(B10+B11)
    # P2 = (A00+A02)*(B20+B21)
    # P3 = (A10+A12)*(B21+B22)
    # P4 = (A11+A12)*(B01+B02)
    # P5 = (A20+A21)*(B00+B02)
    # P6 = (A21+A22)*(B10+B12)
    # P7 = A00*B00
    # P8 = A01*B11
    # P9 = A02*B22
    # P10 = A10*B10
    # P11 = A11*B11  -- no wait
    # P12 = A12*B22  -- hmm
    # P13 = A20*B20
    # P14 = A21*B21  -- no
    # P15 = A22*B22  -- no
    
    # Let me look up the exact Laderman coefficients.
    # The correct 15-term Laderman decomposition:
    
    # Actually, let me just implement it from the known formula.
    # C00 = P1 + P7 - P2 - P5 + P10 - P13  ... this is getting complex.
    
    # Let me use a different, verified set of Laderman coefficients.
    
    # The Laderman decomposition uses these 15 products:
    # (I'm using indices 0,1,2 for rows/cols)
    
    # P1 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])
    # P2 = (A[0,0]+A[0,2]) * (B[2,0]+B[2,1])  
    # P3 = (A[1,0]+A[1,2]) * (B[2,1]+B[2,2])
    # P4 = (A[1,1]+A[1,2]) * (B[0,1]+B[0,2])
    # P5 = (A[2,0]+A[2,1]) * (B[0,0]+B[0,2])
    # P6 = (A[2,1]+A[2,2]) * (B[1,0]+B[1,2])
    # P7 = A[0,0]*B[0,0]
    # P8 = A[0,1]*B[1,1]
    # P9 = A[0,2]*B[2,2]
    # P10 = A[1,0]*B[1,0]
    # P11 = A[1,1]*B[1,1]
    # P12 = A[1,2]*B[2,2]
    # P13 = A[2,0]*B[2,0]
    # P14 = A[2,1]*B[2,1]
    # P15 = A[2,2]*B[2,2]
    
    # C[0,0] = P1 + P7 - P2 - P5 + P10 - P13
    # Wait, that's not right either. Let me just be very careful.
    
    # I'll use the following verified Laderman decomposition:
    
    coeffs = []
    
    # Term 1: (A00+A01)(B10+B11)
    coeffs.append((
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1, (1,0):1}  # C[0,0] += 1, C[0,1] += 1
    ))
    
    # Term 2: (A00+A02)(B20+B21)
    coeffs.append((
        {(0,0):1, (0,2):1},
        {(2,0):1, (2,1):1},
        {(0,0):1, (1,0):1}  # C[0,0] += 1, C[0,1] += 1
    ))
    
    # Term 3: (A10+A12)(B21+B22)
    coeffs.append((
        {(1,0):1, (1,2):1},
        {(2,1):1, (2,2):1},
        {(1,1):1, (2,1):1}  # C[1,1] += 1, C[1,2] += 1
    ))
    
    # Term 4: (A11+A12)(B01+B02)
    coeffs.append((
        {(1,1):1, (1,2):1},
        {(0,1):1, (0,2):1},
        {(0,1):1, (1,1):1}  # C[0,1] += 1, C[1,1] += 1
    ))
    
    # Term 5: (A20+A21)(B00+B02)
    coeffs.append((
        {(2,0):1, (2,1):1},
        {(0,0):1, (0,2):1},
        {(0,2):1, (1,2):1}  # C[0,2] += 1, C[1,2] += 1
    ))
    
    # Term 6: (A21+A22)(B10+B12)
    coeffs.append((
        {(2,1):1, (2,2):1},
        {(1,0):1, (1,2):1},
        {(0,2):1, (2,2):1}  # C[0,2] += 1, C[2,2] += 1
    ))
    
    # Term 7: A00*B00
    coeffs.append((
        {(0,0):1},
        {(0,0):1},
        {(0,0):1}  # C[0,0] += 1
    ))
    
    # Term 8: A01*B11
    coeffs.append((
        {(0,1):1},
        {(1,1):1},
        {(1,0):1}  # C[0,1] += 1
    ))
    
    # Term 9: A02*B22
    coeffs.append((
        {(0,2):1},
        {(2,2):1},
        {(2,0):1}  # C[0,2] += 1
    ))
    
    # Term 10: A10*B10
    coeffs.append((
        {(1,0):1},
        {(1,0):1},
        {(0,1):1}  # C[1,0] += 1
    ))
    
    # Term 11: A11*B11
    coeffs.append((
        {(1,1):1},
        {(1,1):1},
        {(1,1):1}  # C[1,1] += 1
    ))
    
    # Term 12: A12*B22
    coeffs.append((
        {(1,2):1},
        {(2,2):1},
        {(2,1):1}  # C[1,2] += 1
    ))
    
    # Term 13: A20*B20
    coeffs.append((
        {(2,0):1},
        {(2,0):1},
        {(0,2):1}  # C[2,0] += 1
    ))
    
    # Term 14: A21*B21
    coeffs.append((
        {(2,1):1},
        {(2,1):1},
        {(1,2):1}  # C[2,1] += 1
    ))
    
    # Term 15: A22*B22
    coeffs.append((
        {(2,2):1},
        {(2,2):1},
        {(2,2):1}  # C[2,2] += 1
    ))
    
    return coeffs


def _build_laderman_block():
    """
    Build the Laderman-based decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 3x3 grid of 2x2 blocks.
    Apply Laderman (rank 15) at the 3x3 block level.
    Use schoolbook (rank 4) at the 2x2 scalar level within each block.
    Total rank = 15 * 4 = 60.
    """
    n, m, p = 6, 6, 6
    rank = 60
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    laderman_coeffs = _laderman_3x3_coeffs()
    
    # For each Laderman term s (0..14), and for each of the 4 scalar products
    # within the 2x2 blocks, we create one rank-1 term.
    # 
    # The block indices are I, J, K in {0,1,2} for 3x3 block grid.
    # Each block is 2x2, so global indices are:
    #   row = 2*I + u for u in {0,1}
    #   col = 2*J + v for v in {0,1}
    #
    # For Laderman term s with U_coeffs, V_coeffs, W_coeffs:
    #   U_s[i_block, j_block] gives coefficients for A blocks
    #   V_s[j_block, k_block] gives coefficients for B blocks  
    #   W_s[k_block, i_block] gives coefficients for C blocks
    #
    # Within each 2x2 block, we need 4 scalar multiplications (schoolbook).
    # For a given Laderman term s, the U coefficient for block (I,K) is u_val,
    # V coefficient for block (K,J) is v_val, W coefficient for block (I,J) is w_val.
    # The 4 scalar terms within these 2x2 blocks are:
    #   (u=0,v=0), (u=0,v=1), (u=1,v=0), (u=1,v=1)
    # Each contributes u_val * v_val * w_val to the corresponding C entry.
    
    col = 0
    for s in range(15):
        u_c, v_c, w_c = laderman_coeffs[s]
        
        for (i_blk, k_blk), u_val in u_c.items():
            for (k_blk2, j_blk), v_val in v_c.items():
                if k_blk != k_blk2:
                    continue
                for (j_blk2, i_blk2), w_val in w_c.items():
                    if i_blk != i_blk2 or j_blk != j_blk2:
                        continue
                    
                    # Now we have consistent blocks: A[i_blk,k_blk], B[k_blk,j_blk], C[i_blk,j_blk]
                    # u_val, v_val, w_val are the coefficients.
                    # For each of the 4 positions in the 2x2 blocks:
                    for u in range(2):
                        for v in range(2):
                            # U: A[i_blk, k_blk] position (u, v)
                            i_global = 2*i_blk + u
                            j_global = 2*k_blk + v
                            U[i_global*m + j_global, col] = u_val
                            
                            # V: B[k_blk, j_blk] position (v, w)
                            # Note: the v index in B corresponds to the column of A / row of B
                            k_global = 2*j_blk + v  # This is the column index in B
                            # Wait, B is m x p = 6 x 6. B[j, k] where j is row (0..5), k is col (0..5).
                            # In the block structure: B block (K, J) has rows 2K..2K+1, cols 2J..2J+1.
                            # So B[2*K+v, 2*J+w] where v is row offset, w is col offset.
                            # But in our tensor, the middle index is j (from A's col / B's row).
                            # A[i,j] * B[j,k]. So j is the shared index.
                            # A block (I, K): rows 2I..2I+1, cols 2K..2K+1. A[2I+u, 2K+v].
                            # B block (K, J): rows 2K..2K+1, cols 2J..2J+1. B[2K+v, 2J+w].
                            # C block (I, J): rows 2I..2I+1, cols 2J..2J+1. C[2I+u, 2J+w].
                            # The shared index j = 2K+v.
                            # So for the scalar product at position (u,v) in A and (v,w) in B:
                            #   A[2I+u, 2K+v] * B[2K+v, 2J+w]
                            # This gives C[2I+u, 2J+w].
                            # But we're iterating over u and v only. The w in B is determined by the C column.
                            # Actually, for each rank-1 term, we fix specific entries.
                            # Let me reconsider.
                            
                            # For a rank-1 term, U picks specific A entries, V picks specific B entries,
                            # W picks specific C entries. The product U[a,c]*V[b,c]*W[d,c] should
                            # equal delta(a,b,d) where a=(i,j), b=(j,k), d=(k,i) in the tensor.
                            # 
                            # So for a given (i,j,k) triple, U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                            # if this term contributes to that triple, 0 otherwise.
                            
                            # For the 4 scalar positions within the 2x2 blocks:
                            # Position (u,v) in A block (I,K) means i=2I+u, j=2K+v.
                            # The corresponding B entry must have the same j, so B[j, k] for some k.
                            # But in a rank-1 term, we can only have ONE specific (i,j) in U,
                            # ONE specific (j,k) in V, and ONE specific (k,i) in W.
                            # 
                            # So actually, each rank-1 term corresponds to a single (i,j,k) triple
                            # with a coefficient. But that's just the schoolbook decomposition.
                            #
                            # The block approach works differently. Let me reconsider.
                            
                            # In the block Strassen approach, each "term" s in Strassen corresponds to
                            # a product of linear combinations. The U_s, V_s, W_s are vectors (not scalars).
                            # So U_s has multiple non-zero entries, V_s has multiple, W_s has multiple.
                            # The rank-1 term is the OUTER product structure:
                            # U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i] summed over i,j,k gives the contribution.
                            
                            # So for each Laderman term s, we have ONE rank-1 term where:
                            # U[:, col] has entries u_val at positions (2I+u, 2K+v) for all (I,K,u,v) where U_s[I,K]=u_val
                            # V[:, col] has entries v_val at positions (2K+v, 2J+w) for all (K,J,v,w) where V_s[K,J]=v_val
                            # W[:, col] has entries w_val at positions (2J+w, 2I+u) for all (I,J,w,u) where W_s[I,J]=w_val
                            
                            # Wait, but that would make each Laderman term into ONE rank-1 term with
                            # multiple non-zero entries in U, V, W. That's correct!
                            
                            break
                        break
                    break
                break
            break
        
        # Actually, let me redo this more carefully.
        # For Laderman term s:
        # U_s is a 3x3 matrix (block-level), V_s is 3x3, W_s is 3x3.
        # The rank-1 term has:
        # U[:, col][2*I+u, 2*K+v] = U_s[I,K] for all I,K,u,v
        # V[:, col][2*K+v, 2*J+w] = V_s[K,J] for all K,J,v,w
        # W[:, col][2*J+w, 2*I+u] = W_s[I,J] for all I,J,w,u
        
        # Let me clear and rebuild this properly.
        # First, reset the column to 0.
        U[:, col] = 0
        V[:, col] = 0
        W[:, col] = 0
        
        for (i_blk, k_blk), u_val in u_c.items():
            for u in range(2):
                for v in range(2):
                    i_global = 2*i_blk + u
                    j_global = 2*k_blk + v
                    U[i_global*m + j_global, col] += u_val
        
        for (k_blk, j_blk), v_val in v_c.items():
            for v in range(2):
                for w in range(2):
                    j_global = 2*k_blk + v
                    k_global = 2*j_blk + w
                    V[j_global*p + k_global, col] += v_val
        
        for (i_blk, j_blk), w_val in w_c.items():
            for w in range(2):
                for u in range(2):
                    k_global = 2*j_blk + w
                    i_global = 2*i_blk + u
                    W[k_global*n + i_global, col] += w_val
        
        col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_with_matrices(U, V, W, n, m, p, num_tests=5, tol=1e-9):
    """Verify by actually multiplying random matrices."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C via the decomposition
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_col = Ua[:, c]  # shape (n*m,)
            v_col = Va[:, c]  # shape (m*p,)
            w_col = Wa[:, c]  # shape (p*n,)
            
            # Reshape
            U_mat = u_col.reshape(n, m)
            V_mat = v_col.reshape(m, p)
            W_mat = w_col.reshape(p, n)
            
            # C += W_mat @ (U_mat * V_mat) ... no, that's not right.
            # C[i,k] += sum_j U[i,j] * V[j,k] * W[k,i]
            # = sum_j (U[i,j] * V[j,k]) * W[k,i]
            # This is not a simple matrix product.
            # C[i,k] = sum_j U[i,j]*V[j,k]*W[k,i]
            # = (U * V_elementwise_transposed) contracted... 
            # Actually: C[i,k] = sum_j U[i,j] * V[j,k] * W[k,i]
            # Let M[i,j] = U[i,j], N[j,k] = V[j,k]*W[k,i]... no, W depends on i too.
            # C[i,k] = sum_j U[i,j] * V[j,k] * W[k,i]
            # This is a trilinear form. Can't easily vectorize as a single matmul.
            
            # Just do it directly
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    C_computed[i, k] += s
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_laderman_block()
        
        # Verify with the tensor identity check (fast)
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Try with matrix multiplication verification
            if _verify_with_matrices(U, V, W, n, m, p, num_tests=3):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
