import time
import numpy as np
import sympy as sp
from sympy import symbols, solve, Matrix, Rational, Integer
import warnings

# Suppress sympy warnings for speed
warnings.filterwarnings("ignore")

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, T, tol=1e-9):
    """Verify tensor decomposition."""
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _construct_tensor(n, m, p):
    """Construct the matrix multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i"""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _try_rank_decomposition(n, m, p, target_rank, timeout=100):
    """
    Attempt to find a rank-r decomposition using Gröbner basis elimination.
    This is computationally expensive, so we use it cautiously.
    """
    start_time = time.time()
    if time.time() - start_time > timeout:
        return None

    # We formulate the problem as finding vectors u_r, v_r, w_r such that:
    # T[a,b,c] = sum_r u_r[a] * v_r[b] * w_r[c]
    # Since T is 0/1, we can pick a subset of non-zero entries to constrain the system.
    # However, a full Gröbner basis on 128+ variables is likely too slow.
    # Instead, we will use a heuristic: 
    # 1. Start with a known valid decomposition (e.g., schoolbook or Strassen if available).
    # 2. Use numerical linear algebra to attempt rank reduction.
    # 
    # Given the constraints (150s total), symbolic Gröbner for 4x4x8 is likely infeasible 
    # in Python within the time limit. 
    # 
    # ALTERNATIVE STRATEGY:
    # The prompt suggests a "rank-minimal" search. 
    # Known bounds:
    # 4x4x4 rank is 49 (BCL). 
    # 4x4x8 is effectively two 4x4x4 multiplications sharing A.
    # If we treat B as two 4x4 blocks B1, B2, and C as C1, C2.
    # C1 = A*B1, C2 = A*B2.
    # We can use the BCL algorithm for 4x4x4 twice. Rank = 49 + 49 = 98.
    # Can we do better? 
    # If we use Strassen (rank 7) recursively:
    # 4x4x4 via Strassen: 7 * 7 = 49? No, Strassen is for 2x2.
    # 4x4x4 using Strassen blocks:
    # Split 4x4 into 2x2 blocks. 
    # The 4x4x4 tensor can be decomposed using the 7-term Strassen algorithm 
    # applied to the 2x2 blocks, but the "scalar" multiplications are 2x2x2 multiplications.
    # If we use schoolbook for the inner 2x2x2, rank = 7 * 4 = 28? No, that's not right.
    # 
    # Let's look at the "Strassen recursion" for 4x4x4.
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # M1 = (A00+A11)(B00+B11)
    # ...
    # Each M term is a product of 2x2 matrices.
    # If we compute each 2x2 product using schoolbook (4 mults), total rank = 7 * 4 = 28.
    # Wait, is rank(4,4,4) <= 28? No, lower bound is 27, known is 49.
    # Why is the recursive Strassen rank 49 and not 28?
    # Because the "additions" in Strassen are linear combinations of the *entries* of the blocks.
    # When we expand M1 = (A00+A11)(B00+B11), we are effectively doing:
    # A00B00 + A00B11 + A11B00 + A11B11.
    # If we use schoolbook for the inner products, we are not sharing the intermediate sums.
    # 
    # Actually, the standard "Strassen for 4x4" using Strassen for 2x2 inside gives rank 7*7=49.
    # Let's verify this construction.
    # We can build a 4x4x4 decomposition with rank 49 by applying the 7-term Strassen 
    # to the outer 2x2 structure, where each "multiplication" of 2x2 matrices is 
    # implemented by the 7-term Strassen algorithm for 2x2x2.
    
    # Let's try to construct this rank-49 4x4x4 block.
    
    def build_2x2x2_strassen():
        """Returns U, V, W for 2x2x2 Strassen (rank 7)"""
        U = np.zeros((4, 7), dtype=np.float64)
        V = np.zeros((4, 7), dtype=np.float64)
        W = np.zeros((4, 7), dtype=np.float64)
        
        # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
        
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # M3 = A00(B10-B11)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[0,2]=-1; W[1,2]=1
        
        # M4 = A11(B01-B00)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[1,3]=-1; W[3,3]=1
        
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=1; W[1,4]=-1; W[3,4]=1
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[2,5]=-1; W[3,5]=1
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[2,6]=-1
        
        return U, V, W

    def build_4x4x4_strassen_recursion():
        """
        Construct 4x4x4 using Strassen recursion.
        Outer level: 7 terms (Strassen for 2x2 blocks).
        Inner level: Each term is a 2x2x2 multiplication, implemented by Strassen (rank 7).
        Total rank: 7 * 7 = 49.
        """
        U_outer, V_outer, W_outer = build_2x2x2_strassen()
        rank_outer = 7
        
        # We need to map the "scalar" indices of the outer 2x2x2 to the "block" indices of the inner 2x2x2.
        # The outer tensor T4[a,b,c] is a 4x4x4 tensor.
        # a = i*2 + i', b = j*2 + j', c = k*2 + k' where i,j,k in {0,1} are block indices, i',j',k' in {0,1} are local indices.
        # T4[a,b,c] = T2[ (i,j), (j,k), (k,i) ] * T2_inner[ (i',j'), (j',k'), (k',i') ]
        # Wait, this factorization is only true if the tensor is a Kronecker product.
        # The matrix multiplication tensor M_{n} is NOT a Kronecker product of M_{n/2}.
        # M_4 != M_2 \otimes M_2.
        # However, the *algorithm* can be recursive.
        # Strassen's algorithm computes C = AB by 7 multiplications of size n/2.
        # If we use Strassen for the n/2 multiplications, the total number of scalar multiplications is 7 * (rank of M_{n/2}).
        # So, rank(M_4) <= 7 * rank(M_2) = 7 * 7 = 49.
        
        # How to construct the matrices?
        # The outer Strassen defines linear combinations of the blocks A_00, A_01, etc.
        # Let's denote the outer indices as I (block row), J (block col), K (block row of B / col of C).
        # The outer algorithm computes:
        # M_IJK = (Sum_a U_Ia A_a) * (Sum_b V_Jb B_b)
        # And C_c = Sum_k W_Kc M_K...
        
        # Actually, it's easier to think of it as:
        # We have 7 "virtual" multiplications. Each virtual multiplication is between two 2x2 matrices.
        # We implement each virtual multiplication using the inner Strassen (7 terms).
        # So we have 7 * 7 = 49 scalar terms.
        
        U4 = np.zeros((16, 49), dtype=np.float64)
        V4 = np.zeros((16, 49), dtype=np.float64)
        W4 = np.zeros((16, 49), dtype=np.float64)
        
        # Inner Strassen matrices
        U2, V2, W2 = build_2x2x2_strassen()
        
        # Outer Strassen matrices (for the 2x2 block structure)
        # U_outer[I, t] tells us how block A_I contributes to term t.
        # V_outer[J, t] tells us how block B_J contributes to term t.
        # W_outer[K, t] tells us how term t contributes to block C_K.
        
        # Note: In the 2x2x2 Strassen, indices are:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # For 4x4, global indices:
        # A: a = i*4 + j. Block (i, j). Local index inside block is (i', j').
        # Wait, standard row-major: A[i, j] is index i*4+j.
        # Block (I, J) contains A[I*2+i', J*2+j'] for i',j' in 0,1.
        # Global index = (I*2+i')*4 + (J*2+j') = I*8 + i'*4 + J*2 + j'.
        
        # Let's iterate over the 7 outer terms t.
        for t in range(7):
            # Identify which blocks of A are involved in outer term t
            # U_outer[a_block, t] is non-zero for a_block in {0,1,2,3}
            # a_block corresponds to (I, J) in 2x2 block grid? No.
            # In 2x2x2, A is 2x2. Index 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
            # So a_block = I*2 + J.
            
            # Identify which blocks of B are involved
            # b_block = J*2 + K.
            
            # Identify which blocks of C are involved
            # c_block = K*2 + I.
            
            # For each inner term s (0..6) of the 2x2x2 Strassen:
            for s in range(7):
                term_idx = t * 7 + s
                
                # The inner Strassen computes M_inner = A_inner * B_inner.
                # A_inner is a 2x2 matrix formed by linear combos of the *elements* of the A_blocks?
                # No. The recursive Strassen works on the *blocks* as if they were scalars.
                # So, for outer term t, we have "virtual" matrices A_t and B_t of size 2x2.
                # A_t is a 2x2 matrix where entry (i', j') is the linear combination of A-blocks defined by U_outer?
                # No, that's not how it works.
                
                # Let's re-read Strassen recursion.
                # C = AB.
                # A = [[A00, A01], [A10, A11]].
                # M1 = (A00+A11)(B00+B11).
                # Here A00+A11 is a 2x2 matrix sum.
                # So we are multiplying two 2x2 matrices: X = A00+A11 and Y = B00+B11.
                # To compute X*Y, we use the inner Strassen (or schoolbook).
                # If we use inner Strassen, we need to decompose X and Y into their 2x2x2 tensor terms.
                
                # So, for outer term t, we have a linear combination of A-blocks to form X_t.
                # And a linear combination of B-blocks to form Y_t.
                # Then we compute Z_t = X_t * Y_t using inner Strassen.
                # Finally, C-blocks are linear combinations of Z_t.
                
                # How to map this to the global U, V, W?
                # The global term is:
                # u_global[a, term_idx] * v_global[b, term_idx] * w_global[c, term_idx]
                
                # Let's decompose X_t.
                # X_t = Sum_{I,J} (U_outer_coeff[I,J,t]) * A_{I,J}
                # Wait, U_outer in my 2x2x2 block function is defined for the 2x2 tensor.
                # U_outer[a_idx, t] where a_idx is 0..3 for the 2x2 A.
                # a_idx = I*2 + J.
                # So X_t[I', J'] = Sum_{I,J} U_outer[I*2+J, t] * A_{I,J}[I', J']
                # Note: A_{I,J} is the block at row I, col J. Its elements are A[I*2+i', J*2+j'].
                
                # So, the element X_t[i', j'] is:
                # X_t[i', j'] = Sum_{I,J} U_outer[I*2+J, t] * A[I*2+i', J*2+j']
                
                # Similarly for Y_t[j', k']:
                # Y_t[j', k'] = Sum_{J,K} V_outer[J*2+K, t] * B[J*2+j', K*2+k']
                
                # The inner Strassen computes Z_t[i', k'] = Sum_{j''} X_t[i', j''] * Y_t[j'', k']
                # And the inner Strassen decomposes this product into 7 terms:
                # Z_t[i', k'] = Sum_{s=0}^6 u_inner[i', s] * v_inner[j'', s] * w_inner[k', s] ... wait.
                # The inner Strassen computes the product of two 2x2 matrices X and Y.
                # The tensor for 2x2x2 multiplication is T2[x_idx, y_idx, z_idx].
                # T2[x, y, z] = 1 if x=i*2+j, y=j*2+k, z=k*2+i.
                # The decomposition is: T2 = Sum_s u2[:,s] v2[:,s] w2[:,s]^T.
                # So, (X*Y)[z_idx] = Sum_s u2[z_idx, s] * (Sum_x u2[x_idx, s] X[x_idx]) * (Sum_y v2[y_idx, s] Y[y_idx])
                # Wait, the standard form is T[a,b,c] = Sum_r u_r[a] v_r[b] w_r[c].
                # So (X*Y)[c] = Sum_r w_r[c] * (Sum_a u_r[a] X[a]) * (Sum_b v_r[b] Y[b]).
                
                # So, for a fixed outer term t and inner term s:
                # The contribution to C is:
                # w_outer[K, t] * w_inner[k', s] * (Sum_a u_inner[a_idx, s] X_t[a_idx]) * (Sum_b v_inner[b_idx, s] Y_t[b_idx])
                
                # Let's expand the sums:
                # Term = w_outer[K, t] * w_inner[k', s] * 
                #        (Sum_{a_idx} u_inner[a_idx, s] * Sum_{I,J} U_outer[I*2+J, t] * A[I*2+i_a, J*2+j_a]) *
                #        (Sum_{b_idx} v_inner[b_idx, s] * Sum_{J,K} V_outer[J*2+K, t] * B[J*2+j_b, K*2+k_b])
                
                # Where a_idx = i_a*2 + j_a, b_idx = j_b*2 + k_b.
                # And a_idx, b_idx, c_idx (k') are local indices in 0..3.
                
                # We can collect the coefficients for each global A element A[I*2+i_a, J*2+j_a]:
                # Coeff_U[I*2+i_a, J*2+j_a, term_idx] = Sum_{a_idx} u_inner[a_idx, s] * U_outer[I*2+J, t] * (delta condition?)
                # Wait, a_idx determines i_a, j_a.
                # a_idx = i_a*2 + j_a.
                # So for a given a_idx, i_a = a_idx // 2, j_a = a_idx % 2.
                # The sum over a_idx in the inner term s is:
                # Sum_{a_idx} u_inner[a_idx, s] * (Sum_{I,J} U_outer[I*2+J, t] * A[I*2 + (a_idx//2), J*2 + (a_idx%2)])
                
                # This means the coefficient for A[I*2+i, J*2+j] in this term is:
                # u_inner[i*2+j, s] * U_outer[I*2+J, t]
                
                # Similarly for B:
                # Coeff_V[J*2+j, K*2+k, term_idx] = v_inner[j*2+k, s] * V_outer[J*2+K, t]
                
                # And for C:
                # The output is C[K*2+k', I*2+i']? No.
                # C is 4x4. Block (K, I) contains C[K*2+k', I*2+i'].
                # The inner product Z_t has indices (i', k').
                # Z_t[i', k'] corresponds to C_block[K, I] at local position (i', k')?
                # Let's check the indices.
                # X_t is 2x2, Y_t is 2x2. Z_t = X_t Y_t is 2x2.
                # Z_t[i', k'] contributes to C_block[K, I][i', k'].
                # So global C index is c = K*4 + I*2 + k'? No.
                # C row index = K*2 + k'. C col index = I*2 + i'.
                # Global c = (K*2+k')*4 + (I*2+i') = K*8 + k'*4 + I*2 + i'.
                
                # The inner Strassen term s contributes w_inner[c_inner, s] to Z_t[c_inner].
                # c_inner = k'*2 + i'.
                # So the coefficient for C[K*2+k', I*2+i'] is:
                # w_outer[K, t] * w_inner[k'*2+i', s]
                
                # Let's implement this.
                
                # Precompute inner coefficients for this s
                # u_inner[a_idx, s] for a_idx in 0..3
                # v_inner[b_idx, s] for b_idx in 0..3
                # w_inner[c_idx, s] for c_idx in 0..3
                
                for I in range(2):
                    for J in range(2):
                        # A block (I, J)
                        for i_local in range(2):
                            for j_local in range(2):
                                a_global = (I*2 + i_local)*4 + (J*2 + j_local)
                                a_idx = i_local*2 + j_local
                                coeff_u = U2[a_idx, s] * U_outer[I*2+J, t]
                                if coeff_u != 0:
                                    U4[a_global, term_idx] += coeff_u
                
                for J in range(2):
                    for K in range(2):
                        # B block (J, K)
                        for j_local in range(2):
                            for k_local in range(2):
                                b_global = (J*2 + j_local)*4 + (K*2 + k_local)
                                b_idx = j_local*2 + k_local
                                coeff_v = V2[b_idx, s] * V_outer[J*2+K, t]
                                if coeff_v != 0:
                                    V4[b_global, term_idx] += coeff_v
                
                for K in range(2):
                    for I in range(2):
                        # C block (K, I)
                        for k_local in range(2):
                            for i_local in range(2):
                                c_global = (K*2 + k_local)*4 + (I*2 + i_local)
                                c_idx = k_local*2 + i_local
                                coeff_w = W2[c_idx, s] * W_outer[K, t]
                                if coeff_w != 0:
                                    W4[c_global, term_idx] += coeff_w

        # Verify 4x4x4
        T4 = _construct_tensor(4, 4, 4)
        if _verify(U4, V4, W4, T4):
            return U4, V4, W4
        else:
            return None, None, None

    U4, V4, W4 = build_4x4x4_strassen_recursion()
    
    if U4 is not None:
        # Extend to 4x4x8
        # B is 4x8. Split into B1 (4x4) and B2 (4x4).
        # C is 4x8. Split into C1 (4x4) and C2 (4x4).
        # C1 = A*B1, C2 = A*B2.
        # We can use the same U4 for both.
        # V1 = V4, V2 = V4 (but indices shifted for B2).
        # W1 = W4, W2 = W4 (but indices shifted for C2).
        
        rank_block = 49
        U_full = np.zeros((16, 2*rank_block), dtype=np.float64)
        V_full = np.zeros((32, 2*rank_block), dtype=np.float64)
        W_full = np.zeros((32, 2*rank_block), dtype=np.float64)
        
        # Block 1: B1, C1
        U_full[:, 0:rank_block] = U4
        V_full[0:16, 0:rank_block] = V4
        W_full[0:16, 0:rank_block] = W4
        
        # Block 2: B2, C2
        U_full[:, rank_block:2*rank_block] = U4
        V_full[16:32, rank_block:2*rank_block] = V4
        W_full[16:32, rank_block:2*rank_block] = W4
        
        # Verify full
        T_full = _construct_tensor(4, 4, 8)
        if _verify(U_full, V_full, W_full, T_full):
            return U_full, V_full, W_full
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS

    # Build the full 4x4x8 multiplication tensor
    T = _construct_tensor(n, m, p)

    # Schoolbook decomposition (rank 128)
    rank_schoolbook = n * m * p
    U = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook

    # Try to build a verified 4x4x4 block and extend to 4x4x8
    if time.time() < deadline - 10:
        U_block, V_block, W_block = _try_rank_decomposition(n, m, p, 98, timeout=120)
        if U_block is not None:
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = U_block.shape[1]

    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}
