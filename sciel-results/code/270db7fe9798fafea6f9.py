"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Schoolbook rank: n*m*p = 32
    # Try to find a lower rank decomposition using known techniques.
    # For 2x4x4 matrix multiplication, the schoolbook rank is 32.
    # Let's check if there's a known lower rank.
    
    # The matrix multiplication tensor T[i,j,k] = delta_{i,i'} * delta_{k,k'} 
    # where (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{k,k'}
    
    # For the standard basis: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: T[i,j,k] for input (A[i,j], B[j,k]) -> output (C[i,k])
    # T is nonzero when... actually the multiplication tensor maps (A,B) to C
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # So the tensor entries are: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # As a 3-index tensor: T[i,j,k] = 1 if we consider it as a trilinear form.
    # Actually, the multiplication tensor T has T[i,j,k] = delta_{i,i'} delta_{k,k'} ... 
    # Let me think more carefully.
    
    # The multiplication tensor is a 3-tensor T in U* x V* x W* where:
    # U has basis e_{i,j} for A[i,j], V has basis f_{j,k} for B[j,k], W has basis g_{i,k} for C[i,k]
    # T(e_{i,j}, f_{j',k}, g_{i',k'}) = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Wait, that's not right either. Let's use the standard definition.
    # The matrix multiplication tensor m_{n,m,p} is in R^{n*m} x R^{m*p} x R^{p*n}
    # m_{n,m,p}(a, b, c) = sum_{i,j,k} a_{i,j} b_{j,k} c_{k,i} (tr(ABC))
    # Or equivalently, the structure constants: m_{n,m,p}[i*m+j, j*p+k, k*n+i] = 1
    
    # So the tensor has 1s at positions where the indices are consistent.
    # Specifically, T[a_idx, b_idx, c_idx] = 1 iff a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    # for some valid i,j,k.
    
    # We want to decompose T = sum_r U[:,r] * V[:,r] * W[:,r] (outer products)
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook: for each (i,j,k), one term with U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # This gives rank 32.
    
    # Can we do better? For 2x2x2, Strassen gives rank 7 instead of 8.
    # For 2x4x4, let's try to use Strassen-like ideas.
    
    # One approach: block the 2x4 matrix into two 2x2 blocks, and 4x4 into four 2x2 blocks.
    # A = [A11 A12; A21 A22] where each is 2x2
    # B = [B11 B12; B21 B22] where each is 2x2
    # C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22; A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    # Each 2x2 product can be done with rank 7 (Strassen), so we'd need 4*7 = 28?
    # But the blocks share factors, so it might be less.
    
    # Actually, let me just try to construct a decomposition and verify.
    # Let's start with schoolbook rank 32 and see if we can reduce.
    
    # For now, return schoolbook rank 32.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to list of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
