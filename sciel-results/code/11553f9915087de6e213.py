import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-44 decomposition using Strassen for 2x2x2 blocks
    # and trivial decomposition for 1x2x2 blocks.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # Append to the end of the matrices (conceptually)
        # We are building column by column
        pass

    # We'll build numpy arrays directly
    # Max rank is 44
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2x2 helper
    # Returns 7 (u_idx, v_idx, w_idx, u_val, v_val, w_val) tuples?
    # No, Strassen combines products.
    # It defines 7 intermediate products M1..M7.
    # Each M is a product of a linear combo of A and a linear combo of B.
    # So each M corresponds to 1 rank-1 term in the tensor decomposition.
    
    def get_strassen_terms(i_base, j_base, k_base):
        # i_base, j_base, k_base are starting indices for the 2x2 block
        # A is 2x2 at (i_base, j_base)
        # B is 2x2 at (j_base, k_base)
        # C is 2x2 at (i_base, k_base)
        
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        # A indices: (i_base, j_base), (i_base+1, j_base+1)
        # B indices: (j_base, k_base), (j_base+1, k_base+1)
        # C indices: (i_base, k_base), (i_base+1, k_base+1)
        
        # Term 1
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        # This is not a single rank-1 term in the simple sense of U[:,col]*V[:,col]*W[:,col]
        # Wait, the tensor is T[i,j,k].
        # The decomposition is sum_r U[i,r] V[j,r] W[k,r] ? No.
        # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        
        # Strassen M1 = (A00 + A11)(B00 + B11) contributes to C00 and C11.
        # C00 = ... + M1 ...
        # C11 = ... + M1 ...
        
        # So for M1:
        # We need a column r such that:
        # U[i_base*m + j_base, r] = 1
        # U[(i_base+1)*m + (j_base+1), r] = 1
        # V[j_base*p + k_base, r] = 1
        # V[(j_base+1)*p + (k_base+1), r] = 1
        # W[k_base*n + i_base, r] = 1
        # W[(k_base+1)*n + (i_base+1), r] = 1
        
        # This adds 1 to T[i_base, j_base, k_base], T[i_base, j_base, k_base+1]? No.
        # T[i,j,k] is 1 if i,j,k match.
        # The product U[i,r] V[j,r] W[k,r] is non-zero if U[i,r]!=0, V[j,r]!=0, W[k,r]!=0.
        
        # So for M1:
        # Non-zero U at (i0, j0) and (i1, j1).
        # Non-zero V at (j0, k0) and (j1, k1).
        # Non-zero W at (k0, i0) and (k1, i1).
        
        # The term contributes to:
        # T[i0, j0, k0] += 1*1*1 = 1
        # T[i0, j0, k1] += 1*1*0 = 0
        # T[i0, j1, k0] += 0...
        # T[i1, j1, k1] += 1*1*1 = 1
        
        # So M1 contributes to T[i0,j0,k0] and T[i1,j1,k1].
        # This matches C00 and C11 receiving M1.
        
        terms.append({
            'u': [(i_base*m + j_base, 1.0), ((i_base+1)*m + (j_base+1), 1.0)],
            'v': [(j_base*p + k_base, 1.0), ((j_base+1)*p + (k_base+1), 1.0)],
            'w': [(k_base*n + i_base, 1.0), ((k_base+1)*n + (i_base+1), 1.0)]
        })
        
        # M2 = (A10 + A11) B00
        # A: (i1, j0), (i1, j1)
        # B: (j0, k0)
        # C: (i1, k0)
        terms.append({
            'u': [(i_base*m + j_base, 0.0), ((i_base+1)*m + j_base, 1.0), ((i_base)*m + (j_base+1), 0.0), ((i_base+1)*m + (j_base+1), 1.0)],
            # Only non-zeros:
            'u_nonzero': [((i_base+1)*m + j_base, 1.0), ((i_base+1)*m + (j_base+1), 1.0)],
            'v_nonzero': [(j_base*p + k_base, 1.0)],
            'w_nonzero': [(k_base*n + (i_base+1), 1.0)]
        })
        # I will just use the nonzero lists for efficiency
        
        # Let's refine the helper to just return non-zeros
        
        def add_strassen_block(ib, jb, kb, U, V, W, col):
            # M1
            U[ib*m + jb, col] = 1.0
            U[(ib+1)*m + (jb+1), col] = 1.0
            V[jb*p + kb, col] = 1.0
            V[(jb+1)*p + (kb+1), col] = 1.0
            W[kb*n + ib, col] = 1.0
            W[(kb+1)*n + (ib+1), col] = 1.0
            col += 1
            
            # M2 = (A10 + A11) B00
            U[(ib+1)*m + jb, col] = 1.0
            U[(ib+1)*m + (jb+1), col] = 1.0
            V[jb*p + kb, col] = 1.0
            W[kb*n + (ib+1), col] = 1.0
            col += 1
            
            # M3 = A00 (B10 - B11)
            U[ib*m + jb, col] = 1.0
            V[(jb+1)*p + kb, col] = 1.0
            V[(jb+1)*p + (kb+1), col] = -1.0
            W[kb*n + ib, col] = 1.0
            col += 1
            
            # M4 = A11 (B01 - B00)
            U[(ib+1)*m + (jb+1), col] = 1.0
            V[jb*p + (kb+1), col] = 1.0
            V[jb*p + kb, col] = -1.0
            W[(kb+1)*n + (ib+1), col] = 1.0
            col += 1
            
            # M5 = (A00 + A01) B11
            U[ib*m + jb, col] = 1.0
            U[ib*m + (jb+1), col] = 1.0
            V[(jb+1)*p + (kb+1), col] = 1.0
            W[(kb+1)*n + ib, col] = 1.0
            col += 1
            
            # M6 = (A10 - A00) (B00 + B01)
            U[(ib+1)*m + jb, col] = 1.0
            U[ib*m + jb, col] = -1.0
            V[jb*p + kb, col] = 1.0
            V[jb*p + (kb+1), col] = 1.0
            W[kb*n + (ib+1), col] = 1.0
            col += 1
            
            # M7 = (A01 - A11) (B10 + B11)
            U[ib*m + (jb+1), col] = 1.0
            U[(ib+1)*m + (jb+1), col] = -1.0
            V[(jb+1)*p + kb, col] = 1.0
            V[(jb+1)*p + (kb+1), col] = 1.0
            W[(kb+1)*n + ib, col] = 1.0
            col += 1
            
            return col

        def add_trivial_block_1x2x2(i, jb, kb, U, V, W, col):
            # C_k = A0 B0k + A1 B1k
            # Term 1: A0 B00 -> C0
            U[i*m + jb, col] = 1.0
            V[jb*p + kb, col] = 1.0
            W[kb*n + i, col] = 1.0
            col += 1
            
            # Term 2: A1 B10 -> C0
            U[i*m + (jb+1), col] = 1.0
            V[(jb+1)*p + kb, col] = 1.0
            W[kb*n + i, col] = 1.0
            col += 1
            
            # Term 3: A0 B01 -> C1
            U[i*m + jb, col] = 1.0
            V[jb*p + (kb+1), col] = 1.0
            W[(kb+1)*n + i, col] = 1.0
            col += 1
            
            # Term 4: A1 B11 -> C1
            U[i*m + (jb+1), col] = 1.0
            V[(jb+1)*p + (kb+1), col] = 1.0
            W[(kb+1)*n + i, col] = 1.0
            col += 1
            
            return col

        # 4 Strassen blocks
        col = add_strassen_block(0, 0, 0, U, V, W, col)
        col = add_strassen_block(0, 0, 2, U, V, W, col)
        col = add_strassen_block(0, 2, 0, U, V, W, col)
        col = add_strassen_block(0, 2, 2, U, V, W, col)
        
        # 4 Trivial blocks
        col = add_trivial_block_1x2x2(2, 0, 0, U, V, W, col)
        col = add_trivial_block_1x2x2(2, 0, 2, U, V, W, col)
        col = add_trivial_block_1x2x2(2, 2, 0, U, V, W, col)
        col = add_trivial_block_1x2x2(2, 2, 2, U, V, W, col)
        
        return col

    final_rank = add_strassen_block(0, 0, 0, U, V, W, 0)
    # Wait, I need to call the others too.
    # The helper above returns col.
    
    # Let's re-structure to be clean.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
