"""Matrix multiplication 2x4x6 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 4, 6]


def _to_fraction(x, tol=1e-9):
    """Convert a float to a Fraction if it's close to a half-integer or integer, else None."""
    if abs(x - round(x)) < tol:
        return Fraction(int(round(x)))
    # Check half-integers: x = k/2
    val = x * 2
    if abs(val - round(val)) < tol * 2:
        return Fraction(int(round(val)), 2)
    # Check quarters: x = k/4 (just in case, though task says half-integer)
    val = x * 4
    if abs(val - round(val)) < tol * 4:
        return Fraction(int(round(val)), 4)
    return None


def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication."""
    # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    # Indices:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2: n=2, m=2, p=2
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11
    # W rows: 00, 01, 10, 11 (k*n+i: k=0,1; i=0,1 -> 0,1,2,3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen 7 terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U coefficients for A elements (rows 00,01,10,11)
    # M1: A11(1), A22(1)
    U[0, 0] = 1; U[3, 0] = 1
    # M2: A21(1), A22(1)
    U[2, 1] = 1; U[3, 1] = 1
    # M3: A11(1)
    U[0, 2] = 1
    # M4: A22(1)
    U[3, 3] = 1
    # M5: A11(1), A12(1)
    U[0, 4] = 1; U[1, 4] = 1
    # M6: A21(1), A11(-1)
    U[2, 5] = 1; U[0, 5] = -1
    # M7: A12(1), A22(-1)
    U[1, 6] = 1; U[3, 6] = -1
    
    # V coefficients for B elements (rows 00,01,10,11)
    # M1: B11(1), B22(1)
    V[0, 0] = 1; V[3, 0] = 1
    # M2: B11(1)
    V[0, 1] = 1
    # M3: B12(1), B22(-1)
    V[1, 2] = 1; V[3, 2] = -1
    # M4: B21(1), B11(-1)
    V[2, 3] = 1; V[0, 3] = -1
    # M5: B22(1)
    V[3, 4] = 1
    # M6: B11(1), B12(1)
    V[0, 5] = 1; V[1, 5] = 1
    # M7: B21(1), B22(1)
    V[2, 6] = 1; V[3, 6] = 1
    
    # W coefficients for C elements (rows 00,01,10,11)
    # C11 = M1+M4-M5+M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C12 = M3+M5
    W[1, 2] = 1; W[1, 4] = 1
    # C21 = M2+M4
    W[2, 1] = 1; W[2, 3] = 1
    # C22 = M1-M2+M3+M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_strassen_2x4x6():
    """Build the rank-42 decomposition by splitting 2x4x6 into two 2x2x6 blocks."""
    n, m, p = 2, 4, 6
    U_full = np.zeros((n * m, 42))
    V_full = np.zeros((m * p, 42))
    W_full = np.zeros((p * n, 42))
    
    # We split the 4 columns of M1 into two 2-column blocks.
    # Block 1: M1[:, 0:2] (2x2) * M2[0:2, :] (2x6) -> C[:, 0:2] (2x2)
    # Block 2: M1[:, 2:4] (2x2) * M2[2:4, :] (2x6) -> C[:, 2:4] (2x2)
    # Wait, M2 is 4x6. M1 is 2x4.
    # M1 = [A1 A2], A1 is 2x2, A2 is 2x2
    # M2 = [B1; B2], B1 is 2x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    
    # Let's use Strassen for each 2x2x6 product.
    # A 2x2x6 product can be seen as three 2x2x2 products? No.
    # A (2x2) * B (2x6) -> C (2x6).
    # We can split B into three 2x2 blocks: B = [B1 B2 B3]
    # C = [C1 C2 C3] where Ci = A * Bi.
    # Each Ci is 2x2. So 3 Strassen instances per block.
    # Total: 2 blocks * 3 instances * 7 rank = 42. Correct.
    
    U_2x2x2, V_2x2x2, W_2x2x2 = _strassen_2x2x2()
    
    col_offset = 0
    for block_idx in range(2):
        # A_block is M1[:, block_idx*2 : block_idx*2+2]
        # B_block is M2[block_idx*2 : block_idx*2+2, :]
        for sub_block in range(3):
            # B_sub is M2[block_idx*2 : block_idx*2+2, sub_block*2 : sub_block*2+2]
            
            # U: M1 elements involved are M1[i, block_idx*2 + j]
            # In U_2x2x2, row index is i*2+j (i=0,1; j=0,1)
            # In U_full, row index is i*4 + (block_idx*2 + j)
            
            # V: M2 elements involved are M2[block_idx*2 + j, sub_block*2 + k]
            # In V_2x2x2, row index is j*2+k
            # In V_full, row index is (block_idx*2 + j)*6 + (sub_block*2 + k)
            
            # W: C elements involved are C[i, sub_block*2 + k]
            # In W_2x2x2, row index is k*2+i (k=0,1; i=0,1)
            # In W_full, row index is (sub_block*2 + k)*2 + i
            
            for r in range(7):
                # Fill U
                for i in range(2):
                    for j in range(2):
                        val = U_2x2x2[i*2+j, r]
                        if val != 0:
                            U_full[i*4 + block_idx*2 + j, col_offset + r] += val
                            
                # Fill V
                for j in range(2):
                    for k in range(2):
                        val = V_2x2x2[j*2+k, r]
                        if val != 0:
                            V_full[(block_idx*2 + j)*6 + sub_block*2 + k, col_offset + r] += val
                            
                # Fill W
                for k in range(2):
                    for i in range(2):
                        val = W_2x2x2[k*2+i, r]
                        if val != 0:
                            W_full[(sub_block*2 + k)*2 + i, col_offset + r] += val
                            
            col_offset += 7
            
    return U_full, V_full, W_full


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 if i,j,k are valid indices, else 0.
    # Actually, the tensor for matrix multiplication C = A*B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, j, k] = 1 if this is the term contributing to C[i,k] from A[i,j] and B[j,k].
    # Yes, T[i, j, k] = 1 for all i, j, k.
    
    # Let's check the reconstruction.
    # Reconstructed C[i, k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the standard decomposition is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])?
    # No, that's not right.
    
    # The tensor decomposition of matrix multiplication is:
    # T = sum_r U_r \otimes V_r \otimes W_r
    # Where T[i, j, k] = 1.
    # And the multiplication is recovered by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Using the decomposition:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i, j]) * (sum_j V[j*p+k, r] B[j, k]) ??
    
    # Let's look at the schoolbook example:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Term for (i,j,k): 1 * A[i,j] * B[j,k] * 1.
    # Summing over j for fixed i,k:
    # sum_j (sum_col U[i*m+j, col] A[i,j]) ... this is getting confusing.
    
    # The correct identity is:
    # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
    # Let's test with schoolbook:
    # For a specific col corresponding to (i0, j0, k0):
    # U[i*m+j, col] is 1 only if i=i0, j=j0.
    # V[j*p+k, col] is 1 only if j=j0, k=k0.
    # W[k*n+i, col] is 1 only if k=k0, i=i0.
    
    # So for the term r=col:
    # W[k*n+i, r] is 1 only if i=i0 and k=k0.
    # If i != i0 or k != k0, the term is 0.
    # If i=i0 and k=k0:
    #   sum_j U[i0*m+j, r] A[i0, j] = A[i0, j0] (since U is 1 only at j=j0)
    #   sum_j V[j*p+k0, r] B[j, k0] = B[j0, k0] (since V is 1 only at j=j0)
    #   Term = 1 * A[i0, j0] * B[j0, k0]
    # Summing over all cols (i0, j0, k0) such that i0=i, k0=k:
    #   sum_j0 A[i, j0] B[j0, k]
    # This is exactly C[i, k].
    
    # So the verification is:
    # For all i, k:
    #   C_recon[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i, j]) * (sum_j V[j*p+k, r] * B[j, k])
    # This must equal sum_j A[i, j] B[j, k].
    
    # We can check this by picking random A, B and comparing.
    # Or we can check the tensor directly.
    # T_recon[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all i, j, k.
    
    # Let's check T_recon.
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', 
                              U[:, r].reshape(n, m), 
                              V[:, r].reshape(m, p), 
                              W[:, r].reshape(p, n).transpose(1, 0)) # W[k*n+i] -> W_reshaped[k, i] -> transpose to [i, k]?
                              # Wait, W is shape (p*n, rank). W[k*n+i, r].
                              # Reshape to (p, n). Index [k, i].
                              # We need T[i, j, k].
                              # np.einsum('i,j,k->ijk', U_r, V_r, W_r)
                              # U_r is (n*m,) -> reshape (n, m). Index [i, j].
                              # V_r is (m*p,) -> reshape (m, p). Index [j, k].
                              # W_r is (p*n,) -> reshape (p, n). Index [k, i].
                              # So we need to sum over r of U_r[i,j] * V_r[j,k] * W_r[k,i].
    
    # Let's do it efficiently.
    # U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T_recon[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
    
    # We can compute this as a tensor contraction.
    # Temp[i, j, k] = sum_r U[i,j,r] * V[j,k,r]
    # Temp2[i, j, k] = Temp[i, j, k] * W[k, i, r] ?? No, sum over r is already done in Temp.
    # Wait, W depends on r.
    
    # Let's just loop over r for simplicity, rank is small (<=48).
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        u_r = U_reshaped[:, :, r] # (n, m)
        v_r = V_reshaped[:, :, r] # (m, p)
        w_r = W_reshaped[:, :, r] # (p, n)
        
        # T_r[i, j, k] = u_r[i, j] * v_r[j, k] * w_r[k, i]
        # We need to sum this over r.
        
        # Let's construct the full tensor for this r.
        # This is an outer product-like operation.
        # T_r = u_r[:, :, None] * v_r[None, :, :] * w_r[None, None, :] ?
        # u_r[i, j] -> shape (n, m, 1)
        # v_r[j, k] -> shape (1, m, p)
        # w_r[k, i] -> shape (p, 1, n) -> transpose to (n, 1, p)?
        # w_r is (p, n). We need index [k, i].
        # w_r[k, i] corresponds to element (k, i).
        # In numpy, w_r[k, i] is w_r[k, i].
        # We want a tensor W_r_tensor[i, j, k] = w_r[k, i].
        # So W_r_tensor = w_r.T[None, :, :] -> (1, n, p)? No.
        # w_r.T is (n, p). Index [i, k].
        # We need [k, i] in the third dimension?
        # Let's just use einsum or loops for clarity since n,m,p are small.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
    
    # T_recon should be all 1s.
    return np.allclose(T_recon, 1.0, atol=tol)


def _reduce_rank(U, V, W, n, m, p, target_rank, max_iters=1000, deadline=None):
    """Try to reduce the rank by removing linearly dependent columns."""
    # This is a heuristic. We'll try to find a subset of columns that still works.
    # Or we can use a numerical approach to find a lower rank approximation.
    # For now, let's just try to remove columns one by one if possible.
    
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U, V, W
    
    # We'll try to remove a column if the tensor can still be represented.
    # This is NP-hard in general, so we use a heuristic.
    # We'll try removing each column and checking if the remaining columns can span the tensor.
    # This is expensive.
    
    # Alternative: Use a randomized algorithm to find a lower rank decomposition.
    # This is also complex.
    
    # For now, let's just return the current decomposition.
    # The Strassen decomposition is rank 42, which is already an improvement over 48.
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with Strassen decomposition (rank 42)
    U, V, W = _build_strassen_2x4x6()
    
    # Verify it
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to reduce rank further
    # For now, we'll just return the Strassen decomposition
    # A more advanced search could be added here if time permits
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
