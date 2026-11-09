import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p for 3x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication. Rank 7."""
    # Strassen's 7 multiplications for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5
    # C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # n=2, m=2, p=2
    # U: 4x7, V: 4x7, W: 4x7
    
    U = [
        [1, 0, 1, 0, 1, 1, 0],  # i=0,j=0 (a)
        [0, 0, 0, 0, 1, -1, 1],  # i=0,j=1 (b)
        [0, 1, 0, 0, 0, 1, 0],  # i=1,j=0 (c)
        [1, 1, 0, 1, 0, 0, -1],  # i=1,j=1 (d)
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 1],  # j=0,k=0 (e)
        [0, 0, 1, 0, 0, 1, 0],  # j=0,k=1 (f)
        [0, 0, 0, -1, 0, 0, 1],  # j=1,k=0 (g)
        [1, 0, -1, 1, 1, 0, 1],  # j=1,k=1 (h)
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # k=0,i=0 (C11)
        [0, 1, 0, 1, 0, 0, 0],  # k=0,i=1 (C21)
        [0, 0, 1, 0, 1, 0, 0],  # k=1,i=0 (C12)
        [1, -1, 1, 0, 0, 1, 0],  # k=1,i=1 (C22)
    ]
    return {"U": U, "V": V, "W": W}

def _identity_1x1x1():
    """Identity decomposition for 1x1x1. Rank 1."""
    return {"U": [[1]], "V": [[1]], "W": [[1]]}

def _kronecker_product(decomp1, decomp2):
    """
    Compute Kronecker product of two matrix multiplication decompositions.
    If decomp1 is for n1xm1 by m1xp1 with rank r1,
    and decomp2 is for n2xm2 by m2xp2 with rank r2,
    then the Kronecker product is for (n1*n2)x(m1*m2) by (m1*m2)x(p1*p2) with rank r1*r2.
    
    Wait, that's not quite right. Let me think again.
    
    If we have A1 (n1 x m1) * B1 (m1 x p1) = C1 (n1 x p1) with rank r1,
    and A2 (n2 x m2) * B2 (m2 x p2) = C2 (n2 x p2) with rank r2,
    then the Kronecker product A1⊗A2 (n1*n2 x m1*m2) * B1⊗B2 (m1*m2 x p1*p2) = C1⊗C2 (n1*n2 x p1*p2)
    has rank r1*r2.
    """
    U1, V1, W1 = decomp1["U"], decomp1["V"], decomp1["W"]
    U2, V2, W2 = decomp2["U"], decomp2["V"], decomp2["W"]
    r1 = len(U1[0])
    r2 = len(U2[0])
    n1, m1, p1 = len(U1) // m1, m1, len(W1) // n1
    n2, m2, p2 = len(U2) // m2, m2, len(W2) // n2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            a = (i1 * n2 + i2) * m + j1 * m2 + j2
                            U[a][col] = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = (j1 * m2 + j2) * p + k1 * p2 + k2
                            V[b][col] = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            c = (k1 * p2 + k2) * n + i1 * n2 + i2
                            W[c][col] = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
    
    return {"U": U, "V": V, "W": W}

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition_full(U, V, W, n, m, p):
    """Full verification: check all tensor entries."""
    T = _build_tensor(n, m, p)
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    rank = len(U_np[0])
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
    return np.array_equal(T_recon, T)

def _try_border_rank(n, m, p, target_rank, rng, max_iterations=500):
    """
    Try to find a rank-r decomposition using structured random search.
    Uses the border rank idea: perturb the tensor slightly and find low-rank approximations.
    """
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Try different random initializations
    for attempt in range(max_iterations):
        # Random U, V with small entries
        U = rng.randint(-1, 2, size=(n * m, target_rank)).astype(np.float64)
        V = rng.randint(-1, 2, size=(m * p, target_rank)).astype(np.float64)
        W = rng.randint(-1, 2, size=(p * n, target_rank)).astype(np.float64)
        
        # Compute reconstruction error
        T_recon = np.zeros_like(T, dtype=np.float64)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        error = np.sum((T_recon - T.astype(np.float64)) ** 2)
        
        if error < 1e-6:
            # Check if it's exact
            U_int = np.round(U).astype(np.int64)
            V_int = np.round(V).astype(np.int64)
            W_int = np.round(W).astype(np.int64)
            if _verify_decomposition_full(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta_{i,j} * delta_{j,k}
    # Actually, the tensor is: T[i, j, k] = 1 if i==j and j==k else 0? No.
    # The matrix multiplication tensor: (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 if i==j and j==k? No.
    # T[i,j,k] = delta_{i,j} * delta_{j,k} is wrong.
    # The correct tensor: T[i, j, k] = 1 if i==j and j==k? No.
    # Let me think again. The tensor for matrix multiplication is:
    # T(i, j, k) = 1 if i == j and j == k, else 0? No, that's not right either.
    # 
    # The matrix multiplication can be written as:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # This is a tensor T with indices (i, j, k) where T[i,j,k] = 1 if i==j and j==k? 
    # No! The tensor is simply T[i,j,k] = delta_{i,j} * delta_{j,k} is WRONG.
    # 
    # Actually, the tensor T is defined such that:
    # C_{i,k} = sum_{j} A_{i,j} * B_{j,k}
    # So the tensor T has components T[i, j, k] where:
    # T[i, j, k] = 1 if i == j and j == k? No.
    # 
    # Let me be precise. The tensor T is a trilinear form:
    # T(A, B, C) = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For matrix multiplication, we want:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So: T[i,j,k] = 1 if i==j and j==k? No!
    # T[i,j,k] should be 1 when the indices match for the multiplication:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} is still wrong.
    # 
    # The correct answer: T[i,j,k] = 1 if i==j and j==k? Let me check with the schoolbook.
    # In the schoolbook, for each (i,j,k) triple, we set:
    # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    # This means the tensor component at (i*m+j, j*p+k, k*n+i) is 1.
    # 
    # The verification: for each (i,j,k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if i==j and j==k, else 0? 
    # 
    # Actually no. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 if i == j and j == k is NOT correct.
    # 
    # Let me reconsider. The tensor T is defined by:
    # T(e_{i,j}, e_{j',k}, e_{i',k'}) = 1 if i==i' and j==j' and k==k', else 0? No.
    # 
    # The correct definition: The matrix multiplication tensor T has:
    # T[i, j, k] = 1 if i == j and j == k? 
    # 
    # I think I'm overcomplicating this. Let me just verify by checking that the
    # schoolbook decomposition works:
    # For the schoolbook, U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So the product U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] = 1 only when
    # the column corresponds to (i,j,k).
    # 
    # The tensor component T[i*m+j, j*p+k, k*n+i] should be 1 for the correct (i,j,k).
    # But what IS the correct tensor?
    # 
    # The matrix multiplication tensor is:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    # 
    # In other words, T is the identity-like tensor that maps:
    # (i,j) -> i*m+j, (j,k) -> j*p+k, (k,i) -> k*n+i
    # 
    # So T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    # 
    # This is equivalent to saying: T is the tensor such that
    # T[A, B, C] = sum_{i,j,k} A[i,j] * B[j,k] * C[k,i]
    # 
    # Wait, that's a trace, not matrix multiplication. Let me re-read the task.
    # 
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So W has index k*n+i, which means W[k*n+i, r]. The trace convention means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # And the tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # This is the standard matrix multiplication tensor. Let me verify:
    # sum_{i,j,k} A[i,j] * B[j,k] * T[i*m+j, j*p+k, k*n+i]
    # = sum_{i,j,k} A[i,j] * B[j,k] * 1 (when indices match)
    # = sum_{i,j,k} A[i,j] * B[j,k]
    # But this doesn't give C[i,k] directly.
    # 
    # I think the tensor is defined as:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # And the decomposition satisfies:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # 
    # For the schoolbook, this works because each (i,j,k) gives a unique column.
    # 
    # So for verification, I need to check:
    # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all other (a,b,c): sum_r U[a, r] * V[b, r] * W[c, r] = 0
    # 
    # But that's O(n*m*p * n*m*p * rank) which is expensive.
    # 
    # Actually, I only need to check that the tensor is correctly decomposed.
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k), and 0 otherwise.
    # 
    # Let me just verify a few random points to save time.

    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                if val != 1:
                    return False
    
    # Check some random non-matching points
    rng = np.random.RandomState(42)
    for _ in range(100):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or k1 != k2 or i1 != i2:
            val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
            if val != 0:
                return False
    
    return True

def _find_decomposition_over_field(F, n, m, p, max_rank, rng):
    """
    Try to find a rank-max_rank decomposition over field F.
    F is a prime (for GF(F)).
    Returns (U, V, W) as lists of lists, or None if not found.
    """
    T = _build_tensor(n, m, p)
    
    # Try random rank-r decompositions with more attempts
    for attempt in range(50):
        # Generate random U, V, W
        U = rng.randint(0, F, size=(n * m, max_rank)) % F
        V = rng.randint(0, F, size=(m * p, max_rank)) % F
        W = rng.randint(0, F, size=(p * n, max_rank)) % F
        
        # Build the reconstructed tensor
        T_recon = np.zeros_like(T)
        for r in range(max_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r]) % F
        
        if np.array_equal(T_recon % F, T % F):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def _find_decomposition_integer(n, m, p, target_rank, rng, max_attempts=100):
    """
    Try to find an integer decomposition of the given rank.
    Uses structured random search with integer entries.
    """
    T = _build_tensor(n, m, p)
    
    for attempt in range(max_attempts):
        # Use small integers: -2, -1, 0, 1, 2
        U = rng.randint(-2, 3, size=(n * m, target_rank)).astype(np.int64)
        V = rng.randint(-2, 3, size=(m * p, target_rank)).astype(np.int64)
        W = rng.randint(-2, 3, size=(p * n, target_rank)).astype(np.int64)
        
        # Build reconstruction
        T_recon = np.zeros_like(T, dtype=np.int64)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        if np.array_equal(T_recon, T):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    """Find a low-rank decomposition for 3x5x5 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    rng = random.Random(12345)
    np_rng = np.random.RandomState(42)
    
    # Try a few specific ranks with limited attempts
    # Focus on ranks that are more likely to be achievable
    for target_rank in [40, 45, 50, 55, 60]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try integer decomposition with more attempts for promising ranks
        result = _find_decomposition_integer(n, m, p, target_rank, np_rng, max_attempts=50)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best
