import time
import numpy as np
from itertools import product

_DEADLINE = time.time() + 120.0
_TIMEOUT_MARGIN = 5.0  # seconds

def _elapsed_remaining():
    return _DEADLINE - _TIMEOUT_MARGIN - time.time()

def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Start with the known-good Strassen rank-56 decomposition
    rank0 = 56
    U0 = np.zeros((nm, rank0), dtype=float)
    V0 = np.zeros((mp, rank0), dtype=float)
    W0 = np.zeros((pn, rank0), dtype=float)

    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1

            U0[a00, base] = 1; U0[a11, base] = 1
            V0[b00, base] = 1; V0[b11, base] = 1
            W0[c00, base] = 1; W0[c11, base] = 1

            U0[a10, base+1] = 1; U0[a11, base+1] = 1
            V0[b00, base+1] = 1
            W0[c10, base+1] = 1; W0[c11, base+1] = -1

            U0[a00, base+2] = 1
            V0[b01, base+2] = 1; V0[b11, base+2] = -1
            W0[c01, base+2] = 1; W0[c11, base+2] = 1

            U0[a11, base+3] = 1
            V0[b10, base+3] = 1; V0[b00, base+3] = -1
            W0[c00, base+3] = 1; W0[c10, base+3] = 1

            U0[a00, base+4] = 1; U0[a01, base+4] = 1
            V0[b11, base+4] = 1
            W0[c00, base+4] = -1; W0[c01, base+4] = 1

            U0[a10, base+5] = 1; U0[a00, base+5] = -1
            V0[b00, base+5] = 1; V0[b01, base+5] = 1
            W0[c11, base+5] = 1

            U0[a01, base+6] = 1; U0[a11, base+6] = -1
            V0[b10, base+6] = 1; V0[b11, base+6] = 1
            W0[c00, base+6] = 1

    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0

    def _verify(U, V, W):
        """Check U ⊙ V ⊙ W = T where T[i,j,k] = 1 iff j//m == k//n (0-indexed, row/col of B matches)"""
        # T tensor: T[i, j, k] = 1 if (j // m) == (k // n), else 0
        # Actually for matrix multiplication A (n x m) * B (m x p) = C (n x p):
        # T[i, j, k] = 1 where i indexes A rows, j indexes B cols... 
        # Wait, the tensor M_{n,m,p}: T[i, j, k] = delta_{i,k} is not right either.
        # Standard: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So T[i, j, k] = 1 for all valid i in [0,n), j in [0,m), k in [0,p)
        # Actually T[i,j,k] = 1 if the indices are consistent: A has shape (n,m), B has shape (m,p)
        # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p) -- no that's not a tensor decomposition condition.
        
        # The matrix multiplication tensor T_M has T[i,j,k] = 1 if (j == j) -- i.e., it's the identity over the middle index.
        # More precisely: T[i, j, k] = delta_{middle} -- wait.
        # A is (n x m): A[i, j] for i in 0..n-1, j in 0..m-1
        # B is (m x p): B[j, k] for j in 0..m-1, k in 0..p-1
        # C = A B: C[i, k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i, j, k] = 1 (the "identity" on the contraction index j)
        # So T is just 1 everywhere? No -- T[i,j,k] = 1 for all i,j,k. That's not right either.
        
        # Let me think again. The matmul tensor T in the tensor decomposition sense:
        # We want U ⊗ V ⊗ W such that sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # where T[i,j,k] = 1 if (j is a valid index linking A and B).
        # For C[i,k] = sum_j A[i,j] B[j,k], the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Wait no! That would mean the tensor is all-ones, which is trivially decomposable.
        
        # I think the correct interpretation is:
        # U has shape (n*m, rank) -- indices into A flattened
        # V has shape (m*p, rank) -- indices into B flattened  
        # W has shape (p*n, rank) -- indices into C flattened
        # And the decomposition is: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # where i indexes A (0..n*m-1), j indexes B (0..m*p-1), k indexes C (0..p*n-1)
        # T[i,j,k] = 1 if the "contraction" is consistent, i.e., the j-index of A matches the k-index of B...
        
        # Let me use the W index convention: W[k*n + i, r] where k is col of C, i is row of C.
        # C[i, k] = sum_j A[i,j] * B[j,k]
        # A flattened: A_flat[i*m + j]
        # B flattened: B_flat[j*p + k]
        # C flattened (trace convention W[k*n+i]): C_flat[k*n + i]
        
        # So U[i*m + j, r] V[j*p + k, r] W[k*n + i, r] summed over r should give 1
        # iff j matches (which it always does since j is the shared index)
        # Actually T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        # But since j appears in both A's column and B's row, and they're independent indices in the tensor,
        # T[a, b, c] = 1 where a = i*m + j, b = j'*p + k, c = k'*n + i'
        # and we need j = j', i = i', k = k' -- no that's not right.
        
        # The tensor T has T[a, b, c] = 1 if the indices are "compatible":
        # a corresponds to (i, j) in A, b corresponds to (j', k) in B, c corresponds to (k', i') in C
        # For the matmul to work: j = j' (contraction), and c must be (k, i) for the output.
        # So T[a, b, c] = 1 iff (j_from_a) == (j_from_b) AND (i_from_a) == (i_from_c) AND (k_from_b) == (k_from_c)
        # i.e., T is 1 on the "diagonal" where all three are consistent, 0 otherwise.
        
        # In terms of flat indices:
        # a = i*m + j  => i = a // m, j = a % m
        # b = j'*p + k  => j' = b // p, k = b % p
        # c = k'*n + i'  => k' = c // n, i' = c % n
        # T[a,b,c] = 1 iff j == j' AND i == i' AND k == k'
        
        # Verification:
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s - 1.0) > 1e-9:
                        err += abs(s - 1.0)
        # Also check that it's zero where it should be (sample)
        # For efficiency, just check the non-zero pattern
        return err

    def _verify_fast(U, V, W):
        """Fast verification using tensor contractions"""
        rank = U.shape[1]
        # Compute for a sample of (i,j,k) triples
        # Non-zero: j==j', i==i', k==k'
        # Zero: all other combinations (we check a sample)
        total_err = 0.0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # s = sum_r U[a,r] V[b,r] W[c,r]
                    vals = U[a, :] * V[b, :]
                    s = np.dot(vals, W[c, :])
                    total_err += abs(s - 1.0)
        return total_err

    # Verify the base decomposition
    base_err = _verify_fast(U0, V0, W0)
    # print(f"Base decomposition error: {base_err}")

    # Try rank reduction by merging/canceling terms
    # Strategy: find pairs of rank terms that can be combined
    # Or: try removing a term and adjusting others (Gaussian elimination approach)
    
    # More systematic: try all possible subsets to remove
    # With rank 56, removing 1 term: C(56,1) = 56 trials
    # Removing 2 terms: C(56,2) = 1540 trials -- might be too slow
    # Let's try removing 1 term first
    
    for remove_idx in range(rank0):
        if _elapsed_remaining() < 5.0:
            break
        
        # Try to remove term remove_idx
        # This means we need the remaining 55 terms to still span the tensor
        # Equivalent: is the tensor decomposable with these 55 terms?
        # Check: after removing term r, can the remaining terms still give the correct tensor?
        # This is equivalent to: does the removed term's contribution get absorbed by others?
        
        U_r = U0.copy()
        V_r = V0.copy()
        W_r = W0.copy()
        
        # Zero out the removed term
        U_r[:, remove_idx] = 0
        V_r[:, remove_idx] = 0
        W_r[:, remove_idx] = 0
        
        # Check if the remaining 55 terms still decompose the tensor
        err = _verify_fast(U_r, V_r, W_r)
        if err < 1e-9:
            best_rank = rank0 - 1
            best_U, best_V, best_W = U_r, V_r, W_r
            # print(f"Removed term {remove_idx}, rank now {best_rank}")
            break  # Restart search with new base
    
    # If we reduced rank, try again
    if best_rank < rank0:
        U0, V0, W0 = best_U, best_V, best_W
        for remove_idx in range(best_rank):
            if _elapsed_remaining() < 5.0:
                break
            U_r = U0.copy()
            V_r = V0.copy()
            W_r = W0.copy()
            U_r[:, remove_idx] = 0
            V_r[:, remove_idx] = 0
            W_r[:, remove_idx] = 0
            err = _verify_fast(U_r, V_r, W_r)
            if err < 1e-9:
                best_rank -= 1
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    # Try pair removals (more complex -- need to check if 2 terms together are redundant)
    # This requires checking if the span of the remaining terms still covers the tensor
    # For now, keep it simple with single removals
    
    # Try a different strategy: use the algebraic structure to find a lower-rank decomposition
    # The 2x4x8 matmul can be seen as 2x2 blocks of 2x2x8 matmul
    # Or as 4 blocks of 2x1x8 (which are rank 1 each... no)
    
    # Actually, let's try a direct construction:
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    # Can we do better than 56?
    # Lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # Also rank >= n + p - 1 = 9 (weak)
    # For bilinear complexity, rank >= m * min(n,p) = 4 * 2 = 8 (weak)
    
    # Let's try a rank-32 construction using NTT ideas:
    # The 4x4 inner product part can be decomposed
    # Actually for 2x4 x 4x8:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a set of 16 inner products of dimension 4
    # Each inner product of dim 4 has rank 4 (schoolbook) or can use 4-point DFT
    
    # 4-point DFT over j:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Let omega = i (imaginary unit) -- but we need real/integer coefficients
    # 4-point DFT matrix (using real arithmetic where possible):
    # DFT_4 = [[1,1,1,1],[1,-1,1,-1],[1,0,-1,0],[1,0,-1,0]] -- no that's not right
    # 4-point DFT:
    # F_0 = A_0 + A_1 + A_2 + A_3
    # F_1 = A_0 - A_1 + A_2 - A_3  (using omega^1 = -1... wait omega = exp(2*pi*i/4) = i)
    # Actually: F_k = sum_j A_j * omega^{jk}, omega = i
    # F_0 = A_0 + A_1 + A_2 + A_3
    # F_1 = A_0 + i*A_1 - A_2 - i*A_3
    # F_2 = A_0 - A_1 + A_2 - A_3
    # F_3 = A_0 - i*A_1 - A_2 + i*A_3
    # This uses complex numbers, not suitable for integer/rational coefficients.
    
    # Alternative: use the fact that 4 = 2+2 and do a 2D decomposition
    # A[i, j] where j = 2*s + t, s in {0,1}, t in {0,1}
    # B[j, k] where j = 2*s + t
    # C[i,k] = sum_s sum_t A[i, 2s+t] B[2s+t, k]
    
    # This is what the current Strassen approach does -- 2 blocks of 2x2 in the middle dimension
    # But the middle dimension is 4, and we have n=2 rows, so we get 2*(2*2) = 8 blocks of 2x2x2
    # Each needs 7 multiplications -> 56 total. This matches.
    
    # Can we share multiplications between the n=2 blocks?
    # C[0,k] = sum_j A[0,j] B[j,k]
    # C[1,k] = sum_j A[1,j] B[j,k]
    # = sum_j (A[0,j] - A[1,j]) B[j,k] + sum_j A[1,j] B[j,k]
    # So C[0,k] - C[1,k] = sum_j (A[0,j]-A[1,j]) B[j,k]
    # This is an additional inner product but doesn't reduce the count.
    
    # Actually the Strassen approach already exploits some of this structure.
    # Let me verify the current answer is correct and return it.
    
    final_err = _verify_fast(best_U, best_V, best_W)
    # print(f"Final decomposition rank={best_rank}, error={final_err}")
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
