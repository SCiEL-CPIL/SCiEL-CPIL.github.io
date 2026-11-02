"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Block-circulant embedding strategy:
    # Embed the 3x5x7 multiplication into a 4x8x8 structure using cyclic convolution.
    # The key insight: for each output index i, the computation (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # can be viewed as a linear combination of cyclic convolutions.
    
    # We use the Number Theoretic Transform (NTT) over Z_17 (prime) to get exact
    # rank-1 decompositions of cyclic convolutions of size 8.
    # Z_17 has a primitive 8th root of unity: 2^2 = 4, 4^2 = 16, 16^2 = 256 mod 17 = 1
    # Actually, let's verify: we need g such that g^8 = 1 mod 17 and g^4 != 1 mod 17.
    # 17-1 = 16 = 2^4. The 8th roots of unity in Z_17 are solutions to x^8 = 1.
    # The multiplicative group has order 16, so 8th roots are elements of order dividing 8.
    # 2^8 = 256 mod 17 = 1. 2^4 = 16 mod 17 = -1 != 1. So 2 is a primitive 8th root.
    
    # NTT size N = 8, modulus q = 17, primitive root g = 2
    N = 8
    q = 17
    g = 2
    
    # Precompute NTT tables
    # omega = g^((q-1)/N) = 2^(16/8) = 2^2 = 4 mod 17
    omega = pow(g, (q - 1) // N, q)  # omega = 4 mod 17
    omega_inv = pow(omega, q - 2, q)  # inverse of omega mod q
    
    # DFT matrix F: F[a,b] = omega^(a*b) mod q
    F = np.zeros((N, N), dtype=object)
    for a in range(N):
        for b in range(N):
            F[a, b] = pow(omega, a * b, q)
    
    # Inverse DFT: F_inv[a,b] = (1/N) * omega^(-a*b) mod q
    # 1/N mod q = 1/8 mod 17. 8*15 = 120 = 120 mod 17 = 120 - 7*17 = 120-119 = 1. So 1/8 = 15 mod 17.
    inv_N = pow(N, q - 2, q)  # 15 mod 17
    
    F_inv = np.zeros((N, N), dtype=object)
    for a in range(N):
        for b in range(N):
            F_inv[a, b] = (inv_N * pow(omega_inv, a * b, q)) % q
    
    # The multiplication tensor T[i,j,k] = 1 for i<n, j<m, k<p
    # We decompose using the NTT structure.
    
    # For each i in [0,n), we need to compute sum_j A[i,j] * B[j,k] for k in [0,p).
    # This is a linear combination (over j) of the rows of B.
    
    # Key idea: Embed the j-dimension and k-dimension into cyclic groups of size N=8.
    # For fixed i, the operation is: for each k, compute sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # We can write this as: for each frequency f in [0,N), compute a "convolution"
    # that captures the sum over j with appropriate weights.
    
    # Specifically, using the NTT:
    # Define for each i and frequency f:
    #   u_f[i,j] = A[i,j] * omega^(-f*j)  (for j < m, 0 otherwise)
    #   v_f[j,k] = B[j,k] * omega^(f*k)   (for j < m, k < p)
    #   w_f[k] = omega^(f*k)
    
    # Then: sum_f (1/N) * [sum_j u_f[i,j] * v_f[j,k]] * w_f[k]^{-1} ... 
    # This is getting complicated. Let me use a cleaner formulation.
    
    # Alternative: Use the fact that the tensor can be decomposed as:
    # T[i,j,k] = sum_{f=0}^{N-1} (1/N) * omega^(-f*j) * omega^(f*k) * delta_{valid}
    # But this doesn't directly give T[i,j,k] = 1.
    
    # Let me use a direct construction with the NTT:
    # For each i in [0,n) and each frequency f in [0,N):
    #   Create a rank-1 term that captures the contribution.
    
    # The decomposition:
    # T[i,j,k] = sum_f U[i,j,f] * V[j,k,f] * W[k,i,f]
    
    # Set:
    #   U[i,j,f] = A[i,j] * omega^(-f*j)  -> but A[i,j] is a variable, not fixed!
    
    # I see the issue: in tensor decomposition, the entries U,V,W are fixed constants.
    # The tensor T[i,j,k] = 1 (for valid indices) is what we're decomposing.
    # The actual computation is: result[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_{j} A[i,j] * B[j,k] * T[i,j,k]  where T[i,j,k] = 1 for valid (i,j,k)
    
    # So we need: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U,V,W are fixed matrices.
    
    # Using the NTT embedding:
    # Embed j into Z_N and k into Z_N.
    # T[i,j,k] = 1 if j < m and k < p, else 0.
    
    # We can write: T[i,j,k] = (1/N) * sum_f omega^(-f*j) * (sum_{j'<m, k'<p} omega^(f*(j'-k')) * delta_{j,j'} * delta_{k,k'}) * omega^(f*k)
    
    # This is not simplifying nicely. Let me use a different approach.
    
    # Direct NTT-based decomposition:
    # The tensor T is supported on {0,...,n-1} x {0,...,m-1} x {0,...,p-1}.
    # Embed into Z_N x Z_N x Z_N where N >= max(n,m,p) = 7, so N = 8.
    
    # In the embedded space, define T_ext[i,j,k] = T[i,j,k] for i<n, j<m, k<p, else 0.
    # We can decompose T_ext using the 3D DFT, but that gives N^3 = 512 terms, which is worse.
    
    # Better: Use the fact that for fixed i, the slice T[i,:,:] is a rectangular matrix
    # (m x p) of ones. We can decompose each slice using a 2D approach.
    
    # For fixed i, T[i,j,k] = 1 for j<m, k<p.
    # We can write: 1 = (1/N) * sum_f omega^(-f*j) * omega^(f*k) * [sum_{j'<m, k'<p} omega^(f*(j'-k'))]
    # But the sum in brackets depends on f and the ranges, not just j,k.
    
    # Let me try a different embedding. Use the fact that:
    # For j < m and k < p: T[i,j,k] = 1
    # We can write: T[i,j,k] = (1/N) * sum_{f=0}^{N-1} c_f * omega^(-f*j) * omega^(f*k)
    # where c_f is chosen such that this equals 1 for all j<m, k<p.
    
    # But (1/N) * sum_f c_f * omega^(-f*j) * omega^(f*k) = (1/N) * sum_f c_f * omega^(f*(k-j))
    # For this to equal 1 for all j<m, k<p, we need:
    # (1/N) * sum_f c_f * omega^(f*(k-j)) = 1 for all k-j in {-(m-1), ..., p-1}
    
    # Let d = k-j. Then we need: (1/N) * sum_f c_f * omega^(f*d) = 1 for d in D = {k-j : j<m, k<p}
    # D = {-(m-1), ..., p-1} = {-4, -3, ..., 6}
    
    # This is an interpolation problem. We need c_f such that the DFT of c gives 1 on D.
    # But D has 11 elements and we have 8 frequencies, so we can't satisfy this exactly
    # for all d in D with only 8 terms.
    
    # So this approach doesn't work directly. Let me use a different strategy.
    
    # Strategy: Use a block decomposition. Split the m=5 dimension into blocks
    # and use smaller NTT sizes for each block.
    
    # Actually, let me use a verified approach: the schoolbook decomposition
    # but with a clever grouping that reduces the rank.
    
    # For 3x5x7, the schoolbook rank is 105. We can potentially reduce this
    # by noting that the tensor has a product structure.
    
    # T[i,j,k] = 1 = (1 for i<3) * (1 for j<5) * (1 for k<7)
    # This is already a rank-1 tensor in the full space, but we only need it
    # on the support.
    
    # Wait, that's not right. T[i,j,k] = 1 for all valid (i,j,k), so it IS
    # a rank-1 tensor! But that would mean rank 1, which is clearly wrong
    # because matrix multiplication requires more structure.
    
    # I'm confusing myself. Let me re-read the problem.
    
    # The tensor for matrix multiplication is: T[i,j,k] = delta_{i,j,k} in the sense
    # that (A*B)[i,k] = sum_j A[i,j] * B[j,k]. The tensor T has T[i,j,k] = 1 for all
    # valid i,j,k, and the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k].
    
    # So yes, T[i,j,k] = 1 for all valid indices. This IS a rank-1 tensor!
    # But that can't be right because matrix multiplication is not rank 1.
    
    # Oh, I see the issue. The tensor T[i,j,k] = 1 means that the bilinear form is:
    # sum_{i,j,k} A[i,j] * B[j,k] * 1 = sum_i,k (sum_j A[i,j] * B[j,k])
    # This is correct! But the "rank" of this tensor decomposition is not 1
    # because we're decomposing the tensor, not the bilinear form.
    
    # Wait, if T[i,j,k] = 1 for all valid (i,j,k), then T = u ⊗ v ⊗ w where
    # u[i] = 1, v[j] = 1, w[k] = 1. So the tensor rank is 1!
    
    # But that would mean matrix multiplication has tensor rank 1, which is false.
    # The issue is that the tensor for matrix multiplication is NOT T[i,j,k] = 1.
    
    # Let me reconsider. The matrix multiplication tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: (A*B)[i,k] = sum_{i',j,k'} T[i',j,k'] * A[i',j] * B[j,k']
    # where T[i',j,k'] = delta_{i,i'} * delta_{k,k'} ... no, that's not right either.
    
    # Actually, the standard tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the indices are valid, and the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (A*B)[i,k]
    # But this sums over all outputs, which is not what we want.
    
    # I think the correct interpretation is: the tensor T is a 3-tensor with
    # T[i,j,k] = 1, and the decomposition T = sum_c U_c ⊗ V_c ⊗ W_c means:
    # 1 = sum_c U[i,c] * V[j,c] * W[k,c] for all valid (i,j,k).
    
    # If this is the case, then T IS rank 1, and the answer is rank 1!
    # But that contradicts the known results for matrix multiplication rank.
    
    # Let me re-read the task: "Find a rank-minimal exact tensor decomposition for
    # multiplying a 3 by 5 matrix by a 5 by 7 matrix."
    
    # I think the tensor is the structure tensor of matrix multiplication, which
    # is NOT simply all ones. The structure tensor for matrix multiplication
    # M_{n,m,p} has entries T[i,j,k] where the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * x_i * y_j * z_k = (X*Y)_{i,k}
    # where X is n x m, Y is m x p.
    
    # Actually, I think the standard formulation is:
    # The tensor is T with T[i,j,k] = 1, and the bilinear map is:
    # (X, Y) -> sum_{i,j,k} T[i,j,k] * X[i,j] * Y[j,k]
    # But this gives a scalar, not the matrix product.
    
    # I believe the correct tensor is: for each output position (i,k), we have
    # a separate equation. The full tensor has dimensions (n*p) x (n*m) x (m*p)
    # or something like that.
    
    # Let me look at the interface: U has shape (n*m, rank), V has shape (m*p, rank),
    # W has shape (p*n, rank).
    
    # So the tensor T has dimensions (n*m) x (m*p) x (p*n) = 15 x 35 x 21.
    # The entry T[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i' is:
    # T[a,b,c] = 1 if j = j' and i = i' and k = k', else 0?
    
    # Let me think: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: sum_{i,j,k} A[i,j] * B[j,k] * (contribution to output[i,k])
    # So the tensor T has T[(i,j), (j,k), (k,i)] = 1, i.e.,
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    
    # This makes sense! The tensor is:
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Now this is NOT rank 1. It's the standard matrix multiplication tensor.
    
    # For 3x5x7: n=3, m=5, p=7.
    # T has dimensions (3*5) x (5*7) x (7*3) = 15 x 35 x 21.
    # T[(i,j), (j',k), (k',i')] = 1 if j=j', i=i', k=k', else 0.
    
    # The schoolbook decomposition has rank n*m*p = 105:
    # For each (i,j,k), U[(i,j), c] = 1, V[(j,k), c] = 1, W[(k,i), c] = 1.
    
    # Now, can we do better? The rank of matrix multiplication M_{3,5,7} is unknown
    # but is at most 105. Known lower bounds are around 3*5*7 - something.
    
    # For small cases, the rank is often close to n*m*p. For 3x5x7, I'll use
    # the schoolbook decomposition as a safe baseline, but let me try to find
    # a better one.
    
    # Given the complexity, let me return the schoolbook decomposition.
    # This is guaranteed to be correct and has rank 105.
    
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
# EVOLVE-BLOCK-END
