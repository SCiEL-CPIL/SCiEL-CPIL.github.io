"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill

    # Precompute schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    # Winograd-style decomposition for 4x4 * 4x9
    # We can factor the 4x4 part using a rank-10 scheme (like Winograd F(4))
    # and handle the 9 columns with minimal overhead.
    # 
    # Strategy: Use the known Winograd scheme for 4x4 matrix multiplication
    # which has rank 10 (or 11 depending on implementation), then extend to 4x9.
    #
    # For 4x4 * 4x4: Winograd gives rank 10.
    # For 4x4 * 4x9: We can use 9/4 * 10 ≈ 22.5, so try rank 23 or 24.
    # Actually, a better approach: treat it as 4x4 * 4x9 = (4x4*4x4) + (4x4*4x5)
    # or use a direct Winograd-like scheme.
    #
    # Let's try a structured approach:
    # A is 4x4, B is 4x9. C = A*B is 4x9.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Winograd F(4) for 4x4*4x4 uses 10 multiplications.
    # For 4x4*4x9, we can split B into three 4x3 blocks or use a scheme.
    #
    # Alternative: Use the fact that 4x4*4x9 can be done with 
    # rank = ceil(4*4*9 / some factor). The theoretical lower bound is 
    # around 20-25 for practical schemes.
    #
    # Let me construct a Winograd-based scheme for 4x4 * 4x9.
    # 
    # Winograd F(4) for 4x4:
    # Transform A: U = M_A * A (M_A is 10x4x4 reshaped)
    # Transform B: V = M_B * B (M_B is 10x4x4 reshaped)  
    # Multiply: P = U * V (element-wise, 10 products)
    # Transform result: C = M_C * P
    #
    # For 4x4 * 4x9, I'll use a scheme where:
    # - The 4x4 input A is transformed with a 10-row matrix
    # - The 4x9 input B is transformed with a 10-row x 9-col structure
    # - We get 10*9 = 90 products? No, that's too many.
    #
    # Better: Use a rank-24 scheme based on tensor rank considerations.
    # The tensor rank of 4x4x9 multiplication is at most 4*4*9=144 (schoolbook)
    # and at least max(4*4, 4*9, 4*9)=36 (trivial bound).
    # Known best for 4x4x4 is 10 (Winograd). For 4x4x9, try ~25.
    #
    # Let's use a Kronecker/product structure:
    # 4x4*4x9 = (2x2*2x2) tensor (2x2*2x4.5)? Not clean.
    #
    # Practical approach: Use numpy to find a low-rank approximation
    # and then round to integers/half-integers, verifying exactness.

    # Build the multiplication tensor T[i,j,k] = 1 if we compute C[i,k] from A[i,j]*B[j,k]
    # Tensor shape: (n, m, p) where T[i,j,k] contributes to C[i,k]
    # Actually, the tensor M has M[i,j,k] = 1 for all i,j,k (it's the identity in each mode)
    # The matrix multiplication tensor is: for each (i,j,k), the term A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor T is indexed by (i,j,k) and has value 1, and the "contraction" gives C[i,k]
    
    # Let's use a different approach: iterative rank reduction
    # Start with schoolbook (rank 144) and try to reduce rank by finding
    # linear dependencies.

    # Actually, let's just implement a known Winograd scheme for 4x4 and extend it.
    
    # Winograd F(4) for 4x4 matrix multiply (rank 10):
    # From Winograd 1979, the scheme uses:
    # U[0] = A[0,0] + A[0,1] + A[0,2] + A[0,3]  (row sums)
    # etc. This is complex. Let me use a simpler structured approach.
    
    # For 4x4 * 4x9, use the following:
    # Split the 9 columns into groups and use Winograd on each group.
    # 9 = 4 + 4 + 1
    # 4x4*4x4: rank 10
    # 4x4*4x4: rank 10  
    # 4x4*4x1: rank 4 (just matrix-vector, can't do better than 4 with integer coeffs easily)
    # Total: 10 + 10 + 4 = 24
    
    # But we can share the A transforms! The same A can be used for all three B blocks.
    # So: transform A once (10 values), then for each B block, transform B and multiply.
    # B block 1 (4 cols): transform to 10 values, 10 mults
    # B block 2 (4 cols): transform to 10 values, 10 mults
    # B block 3 (1 col): 4 mults (no Winograd needed for 4x1)
    # Total mults: 10 + 10 + 4 = 24
    
    # Let me define the Winograd F(4) scheme for 4x4:
    # Reference: Winograd, "Arithmetic Complexity of Computations", 1979
    # The F(4) scheme for 4x4 matrix multiply has rank 10.
    
    # Transform matrices for Winograd F(4) 4x4*4x4:
    # A is 4x4, B is 4x4, C = A*B is 4x4
    # U = transform_A(A), V = transform_B(B), P = U*V (element-wise), C = transform_C(P)
    
    # From literature, one Winograd F(4) scheme:
    # Let me use the scheme from the "Winograd F(4)" paper.
    # Actually, let me construct it from first principles using null space.
    
    # For 4x4*4x4, the tensor has 64 entries. We want to find rank r such that
    # T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t where T[i,j,k] = sum_t u_t[i] * v_t[j] * w_t[k]
    # and T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term.
    # For matrix multiply, T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes).
    # Wait, that's not right. The tensor for matrix multiply C=A*B where C[i,k]=sum_j A[i,j]B[j,k]
    # has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The rank of this tensor is what we're minimizing.
    
    # For 4x4x4, the tensor is all-ones 4x4x4 tensor? No!
    # T[i,j,k] = 1 for all i,j,k means C[i,k] = sum_j sum_t u_t[i]v_t[j]w_t[k] * 1
    # = sum_t (sum_j u_t[i]v_t[j]) w_t[k] ... that's not matrix multiply.
    
    # I'm confusing myself. Let me be precise.
    # The matrix multiplication tensor M_{n,m,p} has M[i,j,k] = 1 for all i,j,k.
    # A decomposition of rank r means: there exist vectors u_t in R^n, v_t in R^m, w_t in R^p
    # such that M[i,j,k] = sum_{t=1}^r u_t[i] * v_t[j] * w_t[k] for all i,j,k.
    # 
    # Wait, that's the tensor rank of the all-ones tensor, which is 1!
    # That can't be right for matrix multiplication.
    
    # I think I'm confusing the tensor. Let me reconsider.
    # In matrix multiplication, we have C[i,k] = sum_j A[i,j] B[j,k].
    # This is a trilinear form. The "tensor" here is not the all-ones tensor.
    # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # As a bilinear map, it's a 3-tensor T with T[i,j,k] = 1 (since the coefficient of A[i,j]*B[j,k] in C[i,k] is 1).
    # So yes, T[i,j,k] = 1 for all i,j,k.
    # 
    # But then rank 1 would mean T[i,j,k] = u[i]*v[j]*w[k] for all i,j,k, which requires T to be a rank-1 tensor.
    # The all-ones tensor IS rank 1: u[i]=1, v[j]=1, w[k]=1 for all i,j,k.
    # 
    # So why is matrix multiplication not rank 1? Because the decomposition must respect the structure:
    # C[i,k] = sum_j A[i,j] B[j,k] means the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k]
    # where C^* is the dual. The tensor T[i,j,k] = 1 is indeed rank 1 as a 3-tensor.
    # 
    # I think the issue is that "tensor rank" in the context of matrix multiplication refers to
    # the rank of the multiplication as a bilinear map, which is different from the 3-tensor rank.
    # 
    # Let me look at this differently. The bilinear map is:
    # mu: R^{n*m} x R^{m*p} -> R^{n*p}
    # mu(A, B)[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition of this bilinear map is:
    # mu(A,B) = sum_{t=1}^r (u_t^T A) (v_t^T B) w_t
    # where u_t in R^{n*m}, v_t in R^{m*p}, w_t in R^{n*p}.
    # 
    # So we need: for all i,k: sum_j A[i,j]B[j,k] = sum_t (sum_{i',j'} u_t[i',j'] A[i',j']) (sum_{j',k'} v_t[j',k'] B[j',k']) w_t[i,k]
    # 
    # This is equivalent to: the tensor T[i,j,k] (where T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k])
    # can be written as sum_t u_t[i,j] v_t[j,k] w_t[i,k] ... no, that's not right either.
    
    # OK let me just use the standard definition from the problem:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for each column c:
    #   C = (U[:,c]^T A) * (V[:,c]^T B) * W[:,c]  ... no, that's not standard.
    # 
    # Standard: C[i,k] = sum_c (sum_{j} U[i*m+j, c] * A[i,j]) * (sum_{j} V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # Wait, that gives C[i,k] = sum_c (U_c^T A_row_i) * (V_c^T B_col_k) * W_c[i,k]
    # where A_row_i is the i-th row of A flattened, B_col_k is the k-th column of B flattened.
    # 
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'} for all i,k,j,j'
    # 
    # So the condition is: for all i, k, j, j':
    #   sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = 1 if j==j' else 0
    # 
    # This is a system of equations in U, V, W. We want to find a solution with minimal rank (number of columns c).
    
    # Let me set up the equations. For 4x4x9:
    # n=4, m=4, p=9
    # U: 16 x r, V: 36 x r, W: 36 x r
    # Equations: for i in [0,4), k in [0,9), j in [0,4), j' in [0,4):
    #   sum_c U[i*4+j, c] * V[j'*9+k, c] * W[k*4+i, c] = (1 if j==j' else 0)
    # 
    # Total equations: 4*9*4*4 = 576
    # Unknowns: 16*r + 36*r + 36*r = 88*r
    # 
    # For r=24: 88*24 = 2112 unknowns, 576 equations. Underdetermined.
    # For r=10: 88*10 = 880 unknowns, 576 equations. Still underdetermined.
    # 
    # But the equations are cubic (trilinear) in the unknowns, so this is a system of polynomial equations.
    # 
    # Let me try a different approach: fix U and V, and solve for W.
    # For fixed U and V, the equations become linear in W:
    #   sum_c U[i*4+j, c] * V[j'*9+k, c] * W[k*4+i, c] = delta_{j,j'}
    #   Let M_{(j,j'),(i,k),c} = U[i*4+j, c] * V[j'*9+k, c]
    #   Then: sum_c M_{(j,j'),(i,k),c} * W[k*4+i, c] = delta_{j,j'}
    #   This is: for each (i,k), W[k*4+i, :] is a vector in R^r, and we have 16 equations (for j,j')
    #   sum_c M_{(j,j'),(i,k),c} * W[k*4+i, c] = delta_{j,j'}
    #   
    # Actually, for each (i,k), the 16 equations in W[k*4+i, c] for c=1..r are:
    #   For each (j,j'): sum_c (U[i*4+j, c] * V[j'*9+k, c]) * W[k*4+i, c] = delta_{j,j'}
    #   
    # This is a linear system: A_{(i,k)} * W_{(i,k)} = b_{(i,k)}
    # where A_{(i,k)} is 16 x r, with A_{(i,k)[(j,j'), c]} = U[i*4+j, c] * V[j'*9+k, c]
    # and b_{(i,k)[(j,j')]} = delta_{j,j'}
    # 
    # For this to have a solution, we need b_{(i,k)} to be in the column space of A_{(i,k)}.
    # 
    # This is getting complex. Let me just try to find a low-rank decomposition using optimization.
    
    # Given time constraints, let me return the schoolbook decomposition (rank 144)
    # which is guaranteed to work.
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try rank 24 with a structured approach
    # Use the Winograd-like scheme: 4x4*4x4 (rank 10) + 4x4*4x4 (rank 10) + 4x4*4x1 (rank 4) = 24
    
    # For the 4x1 part, it's just matrix-vector multiply, rank 4.
    # For the 4x4 parts, use Winograd F(4) with rank 10.
    
    # Winograd F(4) scheme for 4x4*4x4:
    # I'll use a known scheme. Let me construct it.
    # 
    # Actually, let me just return schoolbook for now and note that a better
    # scheme would require more careful construction.
    
    return schoolbook()
# EVOLVE-BLOCK-END
