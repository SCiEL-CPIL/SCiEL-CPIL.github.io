"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The tensor T is defined by:
    For A (n x m) * B (m x p) = C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The decomposition uses U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    where T[i,j,k] = 1 if A[i,j]*B[j,k] contributes to C[i,k], which is always true.
    
    Wait - this interpretation gives an all-ones tensor which has rank 1. That can't be right.
    
    Let me reconsider. The correct tensor for matrix multiplication is:
    T[i, j, k] where the indices correspond to (row of A, col of A / row of B, col of B)
    and the "output" index is determined by the constraint that col of A = row of B.
    
    Actually, I think the standard formulation is:
    The multiplication tensor is a 3-tensor with entries indexed by (i, j, k) where
    i ranges over rows of A (0..n-1), j over columns of A / rows of B (0..m-1), 
    k over columns of B (0..p-1).
    
    The tensor value T[i,j,k] = 1 for all valid (i,j,k) because every A[i,j]*B[j,k] 
    contributes to C[i,k].
    
    But this gives rank 1! So I must be wrong.
    
    Let me look at this from the Strassen perspective. For 2x2*2x2:
    Schoolbook uses 8 multiplications. The tensor has 8 "terms" A[i,j]*B[j,k] for 
    i,j,k in {0,1}. The rank of this tensor is 7 (Strassen).
    
    So the tensor is NOT all-ones in the sense I was thinking. Let me reconsider.
    
    Actually, I think the issue is that the tensor is defined over the full index space
    (n*m) x (m*p) x (p*n), and T[a,b,c] = 1 if and only if the indices are "compatible" 
    in a specific way.
    
    For the schoolbook decomposition:
    For each (i,j,k) with i in [0,n), j in [0,m), k in [0,p):
      U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    This means the tensor entry T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k).
    
    So the tensor is NOT all ones. It's a specific pattern where T[a,b,c] = 1 if and only
    if there exist i,j,k such that a = i*m+j, b = j*p+k, c = k*n+i.
    
    In other words, T is a 0-1 tensor where T[a,b,c] = 1 iff the "middle indices" match:
    if a = i*m+j and b = j'*p+k, then we need j = j', and c = k*n+i.
    
    So T[a,b,c] = 1 iff:
      Let i = a // m, j = a % m
      Let j' = b // p, k = b % p  
      Let k' = c // n, i' = c % n
      Then T[a,b,c] = 1 iff j == j' AND i == i'
    
    This is the correct tensor! It's not all ones.
    
    Now I can search for low-rank decompositions of this tensor.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Build the target tensor as a sparse representation
    # T[a,b,c] = 1 iff (a%m == b//p) and (a//m == c%n)
    # where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        # Check all entries of the tensor
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                if j != j2:
                    # T[a,b,c] should be 0 for all c
                    for c in range(p * n):
                        s = sum(U[a][c] * V[b][c] * W[c] for c in range(rank))
                        if abs(s) > 1e-9:
                            return False
                else:
                    # T[a,b,c] = 1 iff c%n == i
                    for c in range(p * n):
                        i2 = c % n
                        target = 1.0 if i2 == i else 0.0
                        s = sum(U[a][c] * V[b][c] * W[c] for c in range(rank))
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using matrix operations."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # For each c, compute outer product contribution
        # This is expensive, so use a different approach
        # Check a random subset of entries
        np.random.seed(12345)
        n_samples = 2000
        for _ in range(n_samples):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            i2 = c % n
            
            if j != j2:
                target = 0.0
            else:
                target = 1.0 if i2 == i else 0.0
            
            s = np.sum(U[a] * V[b] * W[c])
            if abs(s - target) > 1e-8:
                return False
        return True
    
    def verify_full(U, V, W):
        """Full verification - vectorized."""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Pre-compute target tensor T as a 3D array (n*m, m*p, p*n)
        # This is large: 16 * 36 * 36 = 20736 floats. Fine.
        T = np.zeros((n * m, m * p, p * n))
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                if j == j2:
                    for c in range(p * n):
                        i2 = c % n
                        if i2 == i:
                            T[a, b, c] = 1.0
        
        # Compute sum_c U[a,c] * V[b,c] * W[c]
        # This is a tensor contraction.
        # Reshape U, V, W to broadcast?
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We want S[a, b, c] = sum_k U[a, k] * V[b, k] * W[c, k]
        
        # Efficient way:
        # S = np.einsum('ak,bk,ck->abc', U, V, W)
        # This might be memory intensive or slow if not optimized, but 16*36*36 is small.
        try:
            S = np.einsum('ak,bk,ck->abc', U, V, W)
            if np.max(np.abs(S - T)) > 1e-8:
                return False
        except:
            # Fallback to loop if einsum fails
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        s = np.dot(U[a] * V[b], W[c])
                        if abs(s - T[a, b, c]) > 1e-8:
                            return False
        return True
    
    # Pre-compute target tensor T and t_c vectors
    T_flat = np.zeros((n * m, m * p, p * n))
    t_vectors = []
    for c in range(p * n):
        t_c = np.zeros(n * m * m * p)
        idx2 = 0
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                if j == j2:
                    i2 = c % n
                    if i2 == i:
                        T_flat[a, b, c] = 1.0
                        t_c[idx2] = 1.0
                idx2 += 1
        t_vectors.append(t_c)
    t_vectors = np.array(t_vectors) # (p*n, n*m*m*p)

    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a low-rank decomposition
    # We will try to find a real-valued solution and then check if it can be rounded.
    # Since exact integer solutions are hard to find with random search, we rely on the structure.
    # If we fail to find rank < 144, we return 144.
    
    # To improve search, we can try to initialize with schoolbook and reduce rank by zeroing out columns
    # and re-optimizing.
    
    # First, let's see if we can find a solution with rank < 144 using the previous method but faster.
    # We will only try a few ranks close to 144 to see if we can shave off a bit.
    
    # Actually, let's just try to verify if the schoolbook is the best we can do quickly.
    # The random search is unlikely to find a better one.
    # However, we can try a "greedy" reduction on the schoolbook solution.
    
    # Let's try to reduce the rank of the schoolbook solution.
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to reduce rank by merging columns
    # This is a complex optimization.
    
    # Given the time, I will just return the schoolbook solution if the random search fails.
    # But I will make the random search faster by using the pre-computed t_vectors.
    
    for target_rank in [132, 120, 108]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(2):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + attempt * 7)
            U = np.random.uniform(-0.5, 0.5, (n * m, target_rank))
            V = np.random.uniform(-0.5, 0.5, (m * p, target_rank))
            W = np.random.uniform(-0.5, 0.5, (p * n, target_rank))
            
            for it in range(20):
                if time.time() > deadline - 8:
                    break
                
                # Build B
                B = np.zeros((n * m * m * p, target_rank))
                idx = 0
                for a in range(n * m):
                    for b in range(m * p):
                        B[idx] = U[a] * V[b]
                        idx += 1
                
                # Solve for W
                # W[c] = lstsq(B, t_vectors[c])
                try:
                    # Solve all c at once if possible?
                    # W = (B^T B)^-1 B^T t_vectors^T ?
                    # B is (576, R). t_vectors is (36, 576).
                    # We want W (36, R) such that B W^T = t_vectors^T
                    # W^T = B^+ t_vectors^T
                    # W = (B^+ t_vectors^T)^T
                    # Using lstsq for each c is safer.
                    for c in range(p * n):
                        W[c] = np.linalg.lstsq(B, t_vectors[c], rcond=None)[0]
                except:
                    pass
                
                # Check error
                # S = np.einsum('ak,bk,ck->abc', U, V, W)
                # err = np.linalg.norm(S - T_flat)
                try:
                    S = np.einsum('ak,bk,ck->abc', U, V, W)
                    err = np.linalg.norm(S - T_flat)
                except:
                    err = 1.0
                
                if err < 1e-6:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify_full(U_list, V_list, W_list):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
            if found:
                break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
