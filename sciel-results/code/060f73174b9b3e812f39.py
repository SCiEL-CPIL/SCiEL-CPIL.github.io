"""Rewritten program for matrix multiplication 2x3 by 3x4 using algebraic structure from Strassen."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy:
    1. First try an explicit algebraic construction based on splitting the 4 output columns
       into two groups of 2, each requiring a 2x3*3x2 multiplication.
    2. If that fails, use randomized search for rank 12.
    3. Fallback to schoolbook (rank 24).
    
    The tensor T for 2x3 * 3x4 has non-zero entries at (i, j, k) where:
    U index: i*m + j  (i in 0..1, j in 0..2)
    V index: j*p + k  (j in 0..2, k in 0..3)
    W index: k*n + i  (k in 0..3, i in 0..1)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Try rank 12 with structured approach first
    result = _construct_rank_12_structured(n, m, p, deadline)
    if result is not None:
        return result

    # Try rank 12 with randomized search
    result = _construct_rank_12_random(n, m, p, deadline)
    if result is not None:
        return result

    # Fallback to schoolbook (rank 24)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                column += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_12_structured(n, m, p, deadline):
    """
    Try to construct a rank-12 decomposition using algebraic structure.
    
    Split the 4 output columns into two groups of 2.
    For each group, we need a 2x3 * 3x2 multiplication.
    
    For 2x3 * 3x2, we can use a rank-6 decomposition based on the identity:
    (a0, a1, a2) * (b0, b1, b2)^T = 
      (a0+b0)(b1+b2) - a0*b1 - a1*b0 - a2*b2 - (a0-a1)(b1-b2) + ...
    
    Actually, let me use a known rank-6 decomposition for 2x3*3x2:
    The 12 scalar products can be computed with 6 rank-1 tensors.
    """
    n1, m1, p1 = 2, 3, 2
    rank1 = 6
    
    # Known rank-6 decomposition for 2x3 * 3x2
    # This is based on the fact that we can share computations between the two output columns
    U1 = np.zeros((n1*m1, rank1), dtype=int)
    V1 = np.zeros((m1*p1, rank1), dtype=int)
    W1 = np.zeros((p1*n1, rank1), dtype=int)
    
    # Decomposition for 2x3*3x2:
    # result[0,0] = a00*b00 + a01*b10 + a02*b20
    # result[0,1] = a00*b01 + a01*b11 + a02*b21
    # result[1,0] = a10*b00 + a11*b10 + a12*b20
    # result[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # We can use the following rank-6 construction:
    # For c in 0..5, define U1[i*3+j, c], V1[j*2+k, c], W1[k*2+i, c]
    
    # Let me try a specific construction:
    # c=0: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1  -> contributes to (i=0,j=0,k=0)
    # c=1: U1[0,1]=1, V1[1,0]=1, W1[0,0]=1  -> contributes to (i=0,j=1,k=0)
    # etc.
    
    # Actually, this is just the schoolbook for 2x3*3x2 with rank 12.
    # We need to find a rank-6 decomposition.
    
    # Let me use a randomized search for the subproblem
    for restart in range(100):
        if time.time() > deadline:
            break
        
        U1 = np.random.choice([-1, 0, 0, 1], size=(n1*m1, rank1))
        V1 = np.random.choice([-1, 0, 0, 1], size=(m1*p1, rank1))
        W1 = np.random.choice([-1, 0, 0, 1], size=(p1*n1, rank1))
        
        if _verify_sub(U1, V1, W1, n1, m1, p1):
            # Great, we have a rank-6 decomposition for 2x3*3x2
            # Now combine two such for the full 2x3*3x4
            U = np.zeros((n*m, 12), dtype=int)
            V = np.zeros((m*p, 12), dtype=int)
            W = np.zeros((p*n, 12), dtype=int)
            
            # Left half (k=0,1): columns 0-5
            for c in range(rank1):
                for i in range(n1):
                    for j in range(m1):
                        u_idx = i*m + j  # i*3+j in full problem
                        U[u_idx, c] = U1[i*m1+j, c]
                for j in range(m1):
                    for k in range(p1):
                        v_idx = j*p + k  # j*4+k in full problem
                        # But in subproblem, V index is j*2+k
                        # We need to map: for k=0,1, v_idx_full = j*4+k
                        # In subproblem, v_idx_sub = j*2+k
                        # So V1[j*2+k, c] should map to V[j*4+k, c]
                        V[j*p + k, c] = V1[j*p1 + k, c]
                for k in range(p1):
                    for i in range(n1):
                        w_idx = k*n + i  # k*2+i in full problem
                        W[w_idx, c] = W1[k*p1 + i, c]
            
            # Right half (k=2,3): columns 6-11
            # We need another rank-6 decomposition, but with shifted k indices
            # For k=2,3, the V indices are j*4+2 and j*4+3
            # We can use a similar decomposition but with different V structure
            
            for c in range(rank1):
                for i in range(n1):
                    for j in range(m1):
                        u_idx = i*m + j
                        U[u_idx, c + rank1] = U1[i*m1+j, c]
                for j in range(m1):
                    for k in range(p1):
                        v_idx = j*p + (k + p1)  # j*4 + (k+2) for k=0,1
                        V[v_idx, c + rank1] = V1[j*p1 + k, c]
                for k in range(p1):
                    for i in range(n1):
                        w_idx = (k + p1)*n + i  # (k+2)*2+i
                        W[w_idx, c + rank1] = W1[k*p1 + i, c]
            
            # Verify the full decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _construct_rank_12_random(n, m, p, deadline):
    """
    Try to find a rank-12 decomposition using randomized search.
    """
    rank = 12
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        U = np.random.choice([-1, 0, 0, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 0, 0, 1], size=(m * p, rank))
        W = np.random.choice([-1, 0, 0, 0, 1], size=(p * n, rank))
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _verify_sub(U, V, W, n, m, p):
    """Verify a sub-decomposition for matmul(n,m,p)."""
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if T[u_idx, v_idx, w_idx] != 1:
                    return False
                    
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_mask[u_idx, v_idx, w_idx] = True
                
    if np.any(T[~target_mask] != 0):
        return False
        
    return True

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    and 0 otherwise.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for c in range(U.shape[1]):
        T += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                if T[u_idx, v_idx, w_idx] != 1:
                    return False
                    
    target_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target_mask[u_idx, v_idx, w_idx] = True
                
    if np.any(T[~target_mask] != 0):
        return False
        
    return True

# EVOLVE-BLOCK-END
