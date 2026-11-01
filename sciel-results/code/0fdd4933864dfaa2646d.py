"""Program for matrix multiplication 2x2 by 2x13 with rank-minimal decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    # Trivial decomposition as fallback: rank = n*m*p = 52
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1

    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid exact decomposition of matmul tensor.
        
        T[i,j,j',k,k',i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        should equal delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        """
        T = np.einsum('ar,br,cr->abc', U, V, W)
        T4 = T.reshape(n, m, m, p, p, n)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1 if (j == jp and i == ip and k == kp) else 0
                                if abs(T4[i, j, jp, k, kp, ip] - expected) > 1e-6:
                                    return False
        return True

    def build_structured_decomposition(rank, seed):
        """Build a structured rank-rank decomposition for 2x2 * 2x13.
        
        Key insight from inspiration: for 2x2 * 2x1 (i.e., multiplying a 2x2 
        by a 2-vector), rank 3 suffices using the identity:
          (a00+a01)(b0+b1) - a00*b0 - a01*b1
        gives us the cross terms. We can apply similar tricks per column.
        
        For each column k of B, we need C[:,k] = A @ B[:,k].
        The standard 2x2 * 2x1 requires rank 3 (Strassen-like for small case).
        So for 13 columns, a naive per-column approach gives rank 13*3 = 39.
        
        Let's try to exploit structure to do better, but first establish rank 39.
        """
        rng = np.random.RandomState(seed)
        
        # Per-column rank-3 decomposition
        # For column k: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        #                 C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
        # 
        # Rank-3 trick: 
        # Let x = A[0,0] + A[0,1], y = B[0,k] + B[1,k]
        # Then A[0,0]*B[0,k] = x*y - A[0,1]*B[0,k] - A[0,0]*B[1,k] + A[0,1]*B[1,k]
        # This doesn't directly help for the full matrix product.
        #
        # Actually for 2x2 * 2x1, we need:
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # Rank 3: use (a00+a01)*(b0+b1), a00*b0, a11*b1
        # c0 = (a00+a01)(b0+b1) - a00*b0 - a01*b1 ... no that's not right either
        #
        # Standard rank-3 for 2x2*2x1:
        # c0 = (a00+a01)*(b0+b1) - a00*b0 - a01*b1 ... still 4 terms
        #
        # Actually the rank of the 2x2*2x1 tensor is 3:
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # We can write:
        # c0 = (a00+a01)*b0 + a01*(b1-b0) ... no
        #
        # Let me think differently. The tensor T_{ijk} for 2x2*2x1:
        # T_{i,j,k} where i in {0,1} (row of A), j in {0,1} (col of A / row of B), k in {0,1} (col of B / row of C)
        # T_{i,j,k} = delta_{j,k} (it's just identity contraction)
        # Wait, for matrix product (AB)_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor is T_{i,j,k} = delta_{j,k}... no.
        #
        # For multiplying A (2x2) by B (2x1):
        # C_i = sum_j A_{ij} B_j
        # Tensor: T_{i,j,k} = delta_{j,k} * 1... no, that's not a tensor decomposition.
        #
        # The multiplication tensor for m x n * n x p has entries T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # For 2x2 * 2x1: n=2, m=2, p=1
        # T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # with i,i' in {0,1}, j,j' in {0,1}, k,k' in {0}
        # So T_{(i,j),(j',0),(0,i')} = delta_{j,j'} * delta_{i,i'}
        #
        # This is just the identity tensor on the j dimension, and identity on i.
        # Rank of this: it's essentially two independent 2x2 identity-like tensors.
        # Actually it factors: T = sum_j delta_{j,j'} * (identity on i)
        # The rank should be 2 * 1 = 2? No...
        #
        # Let me just use the trivial per-column approach: for each of the 13 columns,
        # we need to compute 2 outputs from 2 inputs, which is a 2x2*2x1 problem.
        # The rank of the 2x2*2x1 multiplication tensor is actually 3.
        #
        # For 2x2 * 2x1: 
        # We need to compute:
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        #
        # A rank-3 decomposition:
        # Let u1 = [1,1,0,0] (a00, a01, b0, b1) -> wait, let me use the standard approach.
        #
        # Actually, I recall that the rank of the 2x2*2x1 tensor is 3.
        # One way: 
        # c0 = (a00+a01)*b0 + a01*(b1-b0) ... this uses 2 products
        # c1 = (a10+a11)*b0 + a11*(b1-b0) ... this uses 2 products
        # Total: 4 products, not better than trivial.
        #
        # Another approach:
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        # = [a00, a01; a10, a11] @ [b0; b1]
        #
        # Rank 3: 
        # c0 = (a00+a01)*(b0+b1) - a00*b0 - a01*b1 ... 3 terms but one is negative
        # Hmm, that's still 3 products: (a00+a01)*(b0+b1), a00*b0, a01*b1
        # c0 = (a00+a01)*(b0+b1) - a00*b0 - a01*b1
        # But we need c1 too: c1 = a10*b0 + a11*b1
        # 
        # Let me use a different decomposition. For the 2x2*2x1 case, 
        # the tensor rank is indeed 3 (it's a well-known result).
        #
        # One rank-3 decomposition:
        # r1: A_part = [1,0; 0,0], B_part = [1;0], C_part = [1;0]  -> a00*b0 contributes to c0
        # r2: A_part = [0,0; 0,1], B_part = [0;1], C_part = [0;1]  -> a11*b1 contributes to c1
        # r3: A_part = [0,1; 1,0], B_part = [1;1], C_part = [1;1]  -> (a01*b1 + a10*b0) contributes to both
        # Check: c0 = a00*b0 + (a01*b1 + a10*b0)*1 ... that gives a00*b0 + a01*b1 + a10*b0, wrong.
        #
        # Let me look this up more carefully. The 2x2*2x1 tensor:
        # T_{i,j,k} where i=0,1 (output), j=0,1 (A col / B row), k=0 (B col / C row... wait)
        # 
        # For (A @ B) where A is 2x2 and B is 2x1:
        # C_0 = A_{0,0}*B_0 + A_{0,1}*B_1
        # C_1 = A_{1,0}*B_0 + A_{1,1}*B_1
        #
        # Tensor T_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # Here i,i' in {0,1}, j,j' in {0,1}, k,k' in {0}
        # T_{(i,j),(j',0),(0,i')} = delta_{j,j'} * delta_{i,i'}
        #
        # Nonzero entries: T_{(0,0),(0,0),(0,0)}=1, T_{(0,1),(1,0),(0,0)}=1, T_{(1,0),(0,0),(0,1)}=1, T_{(1,1),(1,0),(0,1)}=1
        #
        # So we have 4 nonzero entries. Can we cover them with 3 rank-1 tensors?
        # T1 = e_{(0,0)} ⊗ e_{(0,0)} ⊗ e_{(0,0)}  -> covers (0,0),(0,0),(0,0)
        # T2 = e_{(0,1)} ⊗ e_{(1,0)} ⊗ e_{(0,0)}  -> covers (0,1),(1,0),(0,0)
        # T3 = e_{(1,0)} ⊗ e_{(0,0)} ⊗ e_{(0,1)}  -> covers (1,0),(0,0),(0,1)
        # T4 = e_{(1,1)} ⊗ e_{(1,0)} ⊗ e_{(0,1)}  -> covers (1,1),(1,0),(0,1)
        #
        # That's 4. Can we do 3?
        # Let's try:
        # U1 = [1,0,0,0]^T (for i*m+j: i=0,j=0 -> index 0; i=0,j=1 -> index 1; i=1,j=0 -> index 2; i=1,j=1 -> index 3)
        # V1 = [1,0]^T (for j'*p+k: j'=0,k=0 -> index 0; j'=1,k=0 -> index 1)
        # W1 = [1,0]^T (for k'*n+i': k'=0,i'=0 -> index 0; k'=0,i'=1 -> index 1)
        #
        # U2 = [0,1,0,0]^T, V2 = [0,1]^T, W2 = [1,0]^T
        # U3 = [0,0,1,0]^T, V3 = [1,0]^T, W3 = [0,1]^T
        # U4 = [0,0,0,1]^T, V4 = [0,1]^T, W4 = [0,1]^T
        #
        # Sum: T1 gives (0,0,0,0),(0,0,0,0),(0,0,0,0) = 1 at position (i=0,j=0),(j'=0,k=0),(k'=0,i'=0)
        # T2 gives (0,1),(0,0,0,1) -> wait, W2=[1,0] means k'=0,i'=0. So (i=0,j=1),(j'=1,k=0),(k'=0,i'=0)
        # T3 gives (i=1,j=0),(j'=0,k=0),(k'=0,i'=1)
        # T4 gives (i=1,j=1),(j'=1,k=0),(k'=0,i'=1)
        #
        # Total: 4 rank-1 tensors. The rank is 4 for 2x2*2x1? But I thought it was 3...
        #
        # Actually, for general m x n * n x p, the rank is at most m*n*p and at least max(m*n, n*p, p*n).
        # For 2x2*2x1: max(4, 2, 2) = 4. So rank >= 4. And we showed rank = 4.
        #
        # So for 2x2*2x13: rank >= max(4, 26, 26) = 26. Trivial rank = 52.
        #
        # Let me try to find a rank-26 or lower decomposition.
        #
        # The lower bound is max(n*m, m*p, p*n) = max(4, 26, 26) = 26.
        #
        # Let's try rank 26, 27, ..., 30 with random search.
        
        # For now, return None to indicate we need to search
        return None

    best_rank = rank_trivial
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}

    # Try to find lower rank decompositions via random search
    # Lower bound is 26, so try from 30 down to 26
    for target_rank in range(30, 25, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank * 7)
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            if check_decomposition(U, V, W):
                found = True
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
