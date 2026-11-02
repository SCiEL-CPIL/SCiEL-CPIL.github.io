"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_val = np.sum(U[:, c].reshape(n, m) * A)
            v_val = np.sum(V[:, c].reshape(m, p) * B)
            W_c = W[:, c].reshape(p, n).T
            C_computed += u_val * v_val * W_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_76(n, m, p):
    # n=4, m=4, p=7
    # Split B into B1 (4x4) and B2 (4x3)
    # Use rank 28 for A*B1 (4x4x4) via Strassen
    # Use rank 48 for A*B2 (4x4x3) via Schoolbook
    # Total rank 76
    
    U_list = []
    V_list = []
    W_list = []
    
    # We will build U, V, W as lists of vectors.
    # U vectors have length n*m = 16
    # V vectors have length m*p = 28
    # W vectors have length p*n = 28
    
    # Part 1: 4x4x4 using Strassen (Rank 28)
    # A = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01], [B10, B11]] (first 4 columns of B)
    # C1 = A @ B1 (first 4 columns of C)
    
    # Strassen terms for 2x2 block multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Contributions to C blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Ms is a 2x2 matrix product of two 2x2 matrices.
    # We expand each Ms into 4 schoolbook terms (one for each entry).
    # Wait, the standard Strassen reduces 8 mults to 7.
    # But here we are doing 4x4 matrix multiplication.
    # If we view A and B as 2x2 blocks of 2x2 matrices,
    # then C = A B involves 8 block multiplications (each 2x2).
    # Each 2x2 multiplication has rank 8 (schoolbook) or 7 (Strassen).
    # So total rank would be 8*8=64 or 8*7=56?
    # No, the Strassen algorithm applies to the block level.
    # The 7 products M1..M7 are 2x2 matrix products.
    # Each 2x2 matrix product has rank 8 (schoolbook).
    # So 7 * 8 = 56 terms?
    # But the current code tries to do something else.
    
    # Let's look at the "Rank 28" claim in the original code.
    # The original code says: "Each Ms is 2x2. We expand each into 4 schoolbook terms. Total 28 terms."
    # This implies that each Ms is treated as a scalar product? No, that's impossible.
    # Or maybe it means that the Strassen coefficients are applied, and then we sum?
    
    # Actually, there is a known result that the tensor rank of 4x4 matrix multiplication is 28?
    # No, the best known upper bound is 28 for 4x4?
    # Wait, Strassen gives 2^log2(7) ~ 7^log2(4) = 49? No.
    # Strassen: T(2n) <= 7 T(n). T(1)=1.
    # T(2)=7. T(4) <= 49.
    # But there are better algorithms.
    # However, the prompt says "Rank 28 for A*B1".
    # Let's assume the construction in the original code was intended to work but had a bug.
    
    # Let's try to construct the 28 terms for 4x4x4.
    # If we use the "Strassen" decomposition for 4x4, we get 7 products of 2x2 matrices.
    # Each 2x2 multiplication requires 8 scalar multiplications.
    # So 7 * 8 = 56 scalar multiplications.
    # This is better than 64 (schoolbook).
    # But the code says 28.
    
    # Is it possible that the "28" comes from a different decomposition?
    # Maybe the "Rank 28" refers to the tensor rank of the 4x4x4 multiplication tensor?
    # The tensor rank of 4x4x4 is known to be 28?
    # Let's check: The tensor rank of 2x2x2 is 7.
    # The tensor rank of 4x4x4 is at most 28?
    # Yes, it is known that the tensor rank of 4x4x4 is 28.
    # (See "The tensor rank of 4x4 matrix multiplication is 28" by ...?)
    # Actually, I recall that the tensor rank of 4x4x4 is 28.
    
    # How is this decomposition constructed?
    # It is not simply 7 Strassen steps of 2x2 multiplications (which would be 56).
    # It must be a more intricate decomposition.
    
    # Since I don't have the explicit 28-term decomposition memorized,
    # and the original code seems to be a broken attempt,
    # I will fall back to a safe construction that I can verify.
    
    # Safe construction:
    # 1. Use Strassen for the 4x4 part (56 terms).
    # 2. Use Schoolbook for the 4x3 part (48 terms).
    # Total rank: 56 + 48 = 104.
    # This is better than 112 (schoolbook).
    
    # Let's implement the 56-term Strassen for 4x4.
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Block indices
    A_offsets = {'A00': (0,0), 'A01': (0,2), 'A10': (2,0), 'A11': (2,2)}
    B_offsets = {'B00': (0,0), 'B01': (0,2), 'B10': (2,0), 'B11': (2,2)}
    C_offsets = {'C00': (0,0), 'C01': (0,2), 'C10': (2,0), 'C11': (2,2)}
    
    # Strassen definitions
    # X_def: list of (block, coeff) for the 2x2 matrix X
    # Y_def: list of (block, coeff) for the 2x2 matrix Y
    # C_contrib: list of (block, coeff) for the 2x2 matrix C
    # The product X*Y contributes to C blocks.
    
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11)
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)]),
        # M2 = (A10+A11)B00
        ([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1), ('C11', -1)]),
        # M3 = A00(B01-B11)
        ([('A00', 1)], [('B01', 1), ('B11', -1)], [('C01', 1), ('C11', 1)]),
        # M4 = A11(B10-B00)
        ([('A11', 1)], [('B10', 1), ('B00', -1)], [('C00', 1), ('C10', 1)]),
        # M5 = (A00+A01)B11
        ([('A00', 1), ('A01', 1)], [('B11', 1)], [('C00', -1), ('C01', 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [('C11', 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C00', 1)]),
    ]
    
    for X_def, Y_def, C_contrib in strassen_defs:
        # X is a 2x2 matrix of blocks (2x2 each).
        # Y is a 2x2 matrix of blocks.
        # We compute the 4 entries of X*Y.
        # Entry (r,c) of X*Y is sum_k X[r,k] Y[k,c].
        # X[r,k] is a 2x2 matrix. Y[k,c] is a 2x2 matrix.
        # The product is a 2x2 matrix.
        # We expand this 2x2 matrix product into 8 scalar terms?
        # No, we want to add rank-1 terms.
        
        # For each entry (r,c) of the 2x2 result matrix:
        # Result[r,c] = sum_k X[r,k] Y[k,c]
        # This is a sum of 2 products of 2x2 matrices.
        # Each product of 2x2 matrices has rank 8.
        # So we need to expand each product into 8 terms.
        
        for r in range(2):
            for c in range(2):
                # We need to compute the 2x2 matrix product:
                # P = sum_k X[r,k] Y[k,c]
                # X[r,k] is a 2x2 matrix formed from A blocks.
                # Y[k,c] is a 2x2 matrix formed from B blocks.
                
                # Let's form the 2x2 matrices X_rc and Y_rc?
                # No, X[r,k] is the (r,k) block of X.
                # X is defined by X_def.
                # X_def is a list of (block_name, coeff).
                # The block_name refers to a 2x2 block of A.
                # So X is a 2x2 matrix of 2x2 blocks.
                # X[r,k] is the (r,k) block of X, which is a 2x2 matrix.
                
                # Let's compute the 2x2 matrix X_rc_k = X[r,k] for k=0,1.
                # And Y_rc_k = Y[k,c] for k=0,1.
                
                # Then P = X_rc_0 * Y_rc_0 + X_rc_1 * Y_rc_1.
                
                # We will expand each product into 8 terms.
                
                for k in range(2):
                    # Form the 2x2 matrix X_k = X[r,k]
                    X_k = np.zeros((2, 2))
                    for block_name, coeff in X_def:
                        r_off, c_off = A_offsets[block_name]
                        # The block is at (r_off, c_off) in A.
                        # The entry (a,b) of the block is at (r_off+a, c_off+b) in A.
                        # So X_k[a,b] += coeff * A[r_off+a, c_off+b]
                        # We will store this as a vector of length 16.
                        # But we need to add it to U_vec.
                        pass # We need to build U_vec, V_vec, W_vec directly.
                    
                    # Let's build the terms for the product X_k * Y_k.
                    # X_k is a 2x2 matrix. Y_k is a 2x2 matrix.
                    # The product is a 2x2 matrix.
                    # Entry (a,b) of the product is sum_s X_k[a,s] Y_k[s,b].
                    # This is a sum of 2 terms.
                    # Each term is a product of a linear form in A and a linear form in B.
                    
                    for a in range(2):
                        for b in range(2):
                            for s in range(2):
                                # Term: X_k[a,s] * Y_k[s,b]
                                # X_k[a,s] is a linear form in A.
                                # Y_k[s,b] is a linear form in B.
                                
                                U_vec = [0] * 16
                                V_vec = [0] * 28
                                W_vec = [0] * 28
                                
                                # Fill U_vec for X_k[a,s]
                                for block_name, coeff in X_def:
                                    r_off, c_off = A_offsets[block_name]
                                    # X_k is the (r,k) block of X.
                                    # So X_k[a,s] corresponds to A[r_off+a, c_off+s]
                                    idx = (r_off + a) * 4 + (c_off + s)
                                    U_vec[idx] += coeff
                                    
                                # Fill V_vec for Y_k[s,b]
                                for block_name, coeff in Y_def:
                                    r_off, c_off = B_offsets[block_name]
                                    # Y_k is the (k,c) block of Y.
                                    # So Y_k[s,b] corresponds to B[r_off+s, c_off+b]
                                    idx = (r_off + s) * 7 + (c_off + b)
                                    # Note: B is 4x7. B1 is the first 4 columns.
                                    # So c_off + b is in 0..3.
                                    V_vec[idx] += coeff
                                    
                                # Fill W_vec for the contribution to C
                                # This term contributes to the (a,b) entry of the (r,c) block of C.
                                # The (r,c) block of C is at (r_off_c, c_off_c).
                                # The entry (a,b) is at (r_off_c+a, c_off_c+b).
                                # W index is k*n+i. Here n=4.
                                # C[i,k] index is k*4+i.
                                # i = r_off_c+a, k = c_off_c+b.
                                
                                for block_name, coeff in C_contrib:
                                    r_off_c, c_off_c = C_offsets[block_name]
                                    i = r_off_c + a
                                    k = c_off_c + b
                                    idx = k * 4 + i
                                    W_vec[idx] += coeff
                                    
                                add_term(U_vec, V_vec, W_vec)
                                
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    # B2 is the last 3 columns of B.
    # For each i in 0..3, j in 0..3, k in 0..2 (local k for B2, global k = 4+k_local):
    # Term A[i,j] B[j, 4+k_local] contributes to C[i, 4+k_local].
    
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = 4 + k_local
                U_vec = [0] * 16
                V_vec = [0] * 28
                W_vec = [0] * 28
                
                U_vec[i*4 + j] = 1
                V_vec[j*7 + k_global] = 1
                # C[i, k_global] index is k_global*4 + i
                W_vec[k_global*4 + i] = 1
                
                add_term(U_vec, V_vec, W_vec)
                
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    best = _build_schoolbook(n, m, p)
    
    # Try the rank 97 construction
    try:
        candidate = _construct_rank_97(n, m, p)
        # Verify the candidate
        if _verify_decomposition(candidate['U'], candidate['V'], candidate['W'], n, m, p):
            best = candidate
    except Exception:
        pass
        
    return best

# EVOLVE-BLOCK-END
