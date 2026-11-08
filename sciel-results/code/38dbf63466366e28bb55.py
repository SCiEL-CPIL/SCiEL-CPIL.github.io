"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Strategy: Try FFT-based structured decomposition first, then fall back to Strassen
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Try FFT-based polynomial multiplication approach
    if time.time() < deadline:
        try:
            U_fft, V_fft, W_fft = _build_fft_solution(n, m, p)
            if _verify_solution(U_fft, V_fft, W_fft, n, m, p):
                r_fft = len(U_fft[0])
                if r_fft < best_rank:
                    best_U, best_V, best_W = U_fft, V_fft, W_fft
                    best_rank = r_fft
        except Exception:
            pass
    
    # Try block-Strassen solution
    if time.time() < deadline:
        try:
            U_str, V_str, W_str = _build_strassen_solution(n, m, p)
            if _verify_solution(U_str, V_str, W_str, n, m, p):
                r_str = len(U_str[0])
                if r_str < best_rank:
                    best_U, best_V, best_W = U_str, V_str, W_str
                    best_rank = r_str
        except Exception:
            pass
    
    # Try to reduce rank by checking for redundant columns in best solution
    if best_U is not None and time.time() < deadline:
        U_red, V_red, W_red = _reduce_rank(best_U, best_V, best_W, n, m, p)
        if len(U_red[0]) < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = len(U_red[0])
    
    if best_U is None:
        # Fallback: return a simple (though high-rank) solution
        U_simple, V_simple, W_simple = _build_simple_solution(n, m, p)
        return {"U": U_simple, "V": V_simple, "W": W_simple}
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_fft_solution(n, m, p):
    """
    Build an FFT-based structured decomposition.
    
    Key insight: Matrix multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    can be viewed as polynomial multiplication. We embed the matrices into
    polynomial structures where multiplication corresponds to convolution.
    
    For n=2, m=4, p=6:
    - Represent A as 2 polynomials of degree 3 (one per row)
    - Represent B as 4 polynomials of degree 5 (one per column of B, transposed view)
    - Use FFT to compute convolutions with fewer multiplications
    
    The rank equals the number of distinct frequency components needed.
    """
    # We'll use a structured approach based on polynomial multiplication
    # For multiplying polynomials of degree d1 and d2, we need d1+d2+1 multiplications
    # but can use FFT to reduce this to O(log(d1+d2)) with precomputed transforms
    
    # For our case:
    # A is 2x4, so each row is a polynomial of degree 3
    # B is 4x6, so we can view columns as polynomials of degree 5
    
    # Strategy: Use a 2D FFT-like structure
    # The number of multiplications needed is related to the number of non-zero
    # terms in the frequency domain representation
    
    # For now, implement a structured decomposition based on block polynomial multiplication
    # Split the middle dimension into blocks and use polynomial multiplication within blocks
    
    # Block sizes: split m=4 into 2 blocks of size 2
    block_size = 2
    num_blocks = m // block_size  # 2 blocks
    
    # For each pair of blocks (one from A, one from B), we do polynomial multiplication
    # A block is 2x2, B block is 2x6
    # This is equivalent to multiplying a degree-1 polynomial (from A block) by a degree-5 polynomial (from B block)
    # which requires 6 multiplications (or fewer with FFT)
    
    # Total multiplications: num_blocks * 6 = 2 * 6 = 12? No, that's not right.
    
    # Let me think differently. Use the fact that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can group the sum by j mod 2:
    # = sum_{j even} A[i,j] * B[j,k] + sum_{j odd} A[i,j] * B[j,k]
    # = sum_{l=0}^{1} A[i,2l] * B[2l,k] + sum_{l=0}^{1} A[i,2l+1] * B[2l+1,k]
    
    # Now, for each parity class, we have 2 terms. We can use Strassen-like tricks
    # to compute these with fewer multiplications.
    
    # Actually, let's use a different approach: Kronecker product structure
    # A (2x4) = A1 (2x2) ⊗ I_2 + A2 (2x2) ⊗ J_2 where J_2 = [[0,1],[1,0]]
    # B (4x6) = B1 (2x6) ⊗ I_2 + B2 (2x6) ⊗ J_2
    # Then A*B = (A1*B1) ⊗ I_2 + (A1*B2) ⊗ J_2 + (A2*B1) ⊗ J_2 + (A2*B2) ⊗ K_2
    # where K_2 = J_2*J_2 = I_2
    # = (A1*B1 + A2*B2) ⊗ I_2 + (A1*B2 + A2*B1) ⊗ J_2
    
    # So we need to compute 4 products of 2x2 and 2x6 matrices
    # Each 2x2 * 2x6 can be done with 2*2*6 = 24 scalar multiplications (naive)
    # Or with Strassen: 7 multiplications for 2x2*2x2, but we have 2x2*2x6
    
    # For 2x2 * 2x6:
    # Let A = [[a,b],[c,d]], B = [[e,f,g,h,i,j],[k,l,m,n,o,p]]
    # Result = [[ae+ck, af+cl, ag+cm, ah+cn, ai+co, aj+cp],
    #           [ce+dk, cf+dl, cg+dm, ch+dn, ci+do, cj+dp]]
    # Naive: 24 multiplications
    # With Strassen-like: 7 multiplications for the 2x2 part, but B has 6 columns
    # We can group columns: B = [B_left (2x3), B_right (2x3)]
    # Then A*B_left and A*B_right each take 7 multiplications (Strassen for 2x2*2x2 applied to 2x2*2x3? No)
    
    # Actually, for 2x2 * 2x6, we can use:
    # M1 = (a+b)*(e+k)
    # M2 = (a+b)*(f+l)
    # M3 = (a+b)*(g+m)
    # M4 = (a+b)*(h+n)
    # M5 = (a+b)*(i+o)
    # M6 = (a+b)*(j+p)
    # M7 = (c-d)*(e+k)
    # M8 = (c-d)*(f+l)
    # M9 = (c-d)*(g+m)
    # M10 = (c-d)*(h+n)
    # M11 = (c-d)*(i+o)
    # M12 = (c-d)*(j+p)
    # Then:
    # [0,0] = M1 - M7
    # [0,1] = M2 - M8
    # [0,2] = M3 - M9
    # [0,3] = M4 - M10
    # [0,4] = M5 - M11
    # [0,5] = M6 - M12
    # [1,0] = M7 + (c-d)*(e+k) ... this doesn't work cleanly
    
    # Let me use a simpler approach: just use the naive multiplication but with a clever
    # factorization that reduces the rank.
    
    # For now, return a placeholder that will be replaced with a proper implementation
    # in the next iteration. For this version, we'll use a rank-24 solution based on
    # the Kronecker structure above.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll construct this based on the Kronecker product decomposition
    # A = A1 ⊗ I + A2 ⊗ J, B = B1 ⊗ I + B2 ⊗ J
    # A*B = (A1*B1 + A2*B2) ⊗ I + (A1*B2 + A2*B1) ⊗ J
    
    # For each of the 4 products (A1*B1, A1*B2, A2*B1, A2*B2), we use naive multiplication
    # Each 2x2 * 2x6 uses 24 scalar multiplications, but we can share some structure
    
    # Actually, let's just return the Strassen solution for now and improve in next iteration
    return _build_strassen_solution(n, m, p)

def _build_strassen_solution(n, m, p):
    """Build the standard block-Strassen solution with rank 42."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return U, V, W

def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by finding linearly dependent columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check for columns that are linear combinations of others
    # This is complex for trilinear forms, so we do a simple check:
    # If two columns have the same pattern up to scaling, we might be able to merge
    # For now, just return the original if no obvious reduction
    
    # Try to find if any column is zero
    for c in range(rank):
        if np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0):
            # Remove this column
            U_new = np.delete(U_arr, c, axis=1)
            V_new = np.delete(V_arr, c, axis=1)
            W_new = np.delete(W_arr, c, axis=1)
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return U, V, W

def _build_simple_solution(n, m, p):
    """Build a simple (high-rank) solution as fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W

def _build_custom_block_solution(n, m, p, m_split):
    """Build a solution with custom block splitting of the middle dimension."""
    # This is a placeholder - implement specific block structures
    raise NotImplementedError("Custom block solution not yet implemented")

def _solve_for_rank(n, m, p, r):
    """Try to solve for a rank-r decomposition. Returns None if not found."""
    # This would require solving a large nonlinear system
    # For now, return None
    return None, None, None
# EVOLVE-BLOCK-END
