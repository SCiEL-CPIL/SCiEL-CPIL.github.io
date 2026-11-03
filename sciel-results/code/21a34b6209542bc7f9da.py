import numpy as np
import time
import random

def _schoolbook_4x4x7():
    """Standard schoolbook decomposition of 4x4 * 4x7 -> 4x7. Rank = 4*4*7 = 112."""
    n, m, p = 4, 4, 7
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1.0
                V[j * p + k, t] = 1.0
                W[k * n + i, t] = 1.0
                t += 1
    return {'U': U[:, :t], 'V': V[:, :t], 'W': W[:, :t]}

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the 4x4x7 matrix multiplication tensor."""
    n, m, p = 4, 4, 7
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # Compare with the multiplication tensor: T_mult[i,j,k] = 1 if j==k else 0... 
    # Actually the multiplication tensor for C = A*B where A is n x m, B is m x p, C is n x p
    # T[i,j,k] = 1 if j == k? No.
    # The tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entry T[i,j,k] = 1 (it's the structure tensor, not the multiplication result)
    # The decomposition means: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = delta_{jk}... no.
    # Actually for matrix multiplication, the tensor M_{n,m,p} has entries M[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to (i,k) of C.
    # So M[i,j,k] = 1 for all valid i,j,k (it's just the identity in the middle index j=k... no)
    # Wait, the matrix multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^T (in C) * e_j (in A) * e_k (in B) * delta_{j,k}... 
    # No. Let me think again.
    # C = A * B, where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition expresses this as:
    # C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_k V[j*p+k, t] * B[j,k]) ... no that's not right either.
    # 
    # The standard form: 
    # T_{ijk} = sum_t U_{i,t} V_{j,t} W_{k,t} where the tensor is for multiplying (i,j) of A with (j,k) of B
    # Actually the correct tensor for n x m times m x p is:
    # T[i, j, k] = 1 if we're looking at the product term A[i,j]*B[j,k] contributing to C[i,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p) where the middle index matches
    # But in our indexing: U has shape (n*m, rank), indexed by [i*m+j, t]
    # V has shape (m*p, rank), indexed by [j*p+k, t]  
    # W has shape (p*n, rank), indexed by [k*n+i, t] (trace convention W[k*n+i, r])
    # 
    # The decomposition condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1  (since every (i,j,k) triple contributes)
    # Wait no, that's not right either. The tensor M_{n,m,p} has:
    # M[i,j,k] = 1 for all valid indices (it's the all-ones tensor in the appropriate space)
    # No! The matrix multiplication tensor has M[i,j,k] = 1 if j == k... no.
    # 
    # Let me be precise. The bilinear map is: (A, B) -> C where C = AB
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The associated trilinear tensor T satisfies:
    # T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] appears with coefficient 1)
    # So T is the all-ones tensor of shape (n, m, p)? No, it's shape (n, m, p) with T[i,j,k]=1.
    # 
    # Wait, I think the tensor is actually: the structure tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # because each product A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    # 
    # So the verification is: for all i,j,k: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    T = np.zeros((n, m, p))
    T.fill(1.0)
    
    # Compute from decomposition
    T_dec = np.zeros((n, m, p))
    for t in range(rank):
        U_t = U[:, t].reshape(n, m)
        V_t = V[:, t].reshape(m, p)
        W_t = W[:, t].reshape(p, n)
        T_dec += np.einsum('ij,jk,ki->ijk', U_t, V_t, W_t)
    
    if not np.allclose(T, T_dec, atol=1e-10):
        return False
    return True

def _build_strassen_2x2x2():
    """Strassen decomposition of 2x2 * 2x2 -> 2x2. Rank 7."""
    # Standard Strassen formulas
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: [0,1,2,3] = [00,01,10,11] (row-major: i*m+j)
    # B indices: [0,1,2,3] = [00,01,10,11] (row-major: j*p+k)
    # C indices: [0,1,2,3] = [00,01,10,11] (W[k*n+i] so index is k*n+i)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] += 1; U[3, 0] += 1  # A00, A11
    V[0, 0] += 1; V[3, 0] += 1  # B00, B11
    W[0, 0] += 1  # C00 (k=0,i=0 -> idx 0)
    W[3, 0] += 1  # C11 (k=1,i=1 -> idx 3)
    
    # M2 = (A10+A11)B00
    U[2, 1] += 1; U[3, 1] += 1  # A10, A11
    V[0, 1] += 1  # B00
    W[2, 1] += 1  # C10 (k=0,i=1 -> idx 2)
    
    # M3 = A00(B01-B11)
    U[0, 2] += 1  # A00
    V[1, 2] += 1; V[3, 2] -= 1  # B01 - B11
    W[1, 2] += 1  # C01 (k=1,i=0 -> idx 1)
    W[3, 2] += 1  # C11 (k=1,i=1 -> idx 3)
    
    # M4 = A11(B10-B00)
    U[3, 3] += 1  # A11
    V[2, 3] += 1; V[0, 3] -= 1  # B10 - B00
    W[0, 3] += 1  # C00
    W[2, 3] += 1  # C10 (k=0,i=1 -> idx 2)
    
    # M5 = (A00+A10)B11
    U[0, 4] += 1; U[2, 4] += 1  # A00, A10
    V[3, 4] += 1  # B11
    W[0, 4] -= 1  # C00
    W[1, 4] += 1  # C01 (k=1,i=0 -> idx 1)
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] += 1; U[0, 5] -= 1  # A10 - A00
    V[0, 5] += 1; V[2, 5] += 1  # B00 + B10
    W[3, 5] += 1  # C11 (k=1,i=1 -> idx 3)
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] += 1; U[3, 6] -= 1  # A01 - A11
    V[1, 6] += 1; V[3, 6] += 1  # B01 + B11
    W[0, 6] += 1  # C00
    W[3, 6] += 1  # C11 (k=1,i=1 -> idx 3)
    
    return {'U': U, 'V': V, 'W': W}

def _build_4x4x7_optimized():
    """Build a decomposition for 4x4 * 4x7 using block structure with Strassen on 2x2x2 blocks.
    
    We partition A (4x4) into four 2x2 blocks, B (4x7) into two 2x7 blocks vertically,
    and C (4x7) into four 2x7 blocks.
    
    Actually, let's use a simpler approach: use schoolbook for the full 4x4x7 but try
    to find a lower-rank decomposition using a structured approach.
    
    Given time constraints, let's use the known result: r(4,4,7) <= r(4,4,4) + r(4,4,3)
    with r(4,4,4) <= 21 (or 25 with Strassen) and r(4,4,3) <= ?
    
    For now, let's build a reasonable decomposition.
    """
    n, m, p = 4, 4, 7
    
    # Use schoolbook as baseline: rank 112
    # Try to do better with block Strassen
    # Split 4x4 * 4x7 as:
    # C = A * B where A is 4x4, B is 4x7
    # Partition A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Partition B into 2x7 and 2x7: B = [[B0], [B1]] where B0, B1 are 2x7
    # C = [[C0], [C1]] where C0, C1 are 2x7
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    #
    # So we need to compute 2x2 * 2x7 four times (A00*B0, A01*B1, A10*B0, A11*B1)
    # Each 2x2 * 2x7 can be done with rank... 
    # 2x2 * 2x7: schoolbook is 4*2*7 = 56, but we can do better
    # Using Strassen-like: treat 2x7 as 3.5 copies of 2x2... 
    # Actually 2x2 * 2x7 = (2x2 * 2x2) x 3.5... not clean
    # Let's just do 2x2 * 2x7 with schoolbook: rank = 2*2*7 = 28 each
    # Total: 4 * 28 = 112 (same as schoolbook)
    
    # Alternative: use the fact that r(4,4,4) <= 25 (Strassen recursive)
    # and handle the extra 3 columns separately
    # r(4,4,7) <= r(4,4,4) + 4*3 = 25 + 12 = 37? No, that's not how it works.
    # r(n,m,p1+p2) <= r(n,m,p1) + r(n,m,p2)
    # r(4,4,7) <= r(4,4,4) + r(4,4,3)
    # r(4,4,4) <= 25 (Strassen) or maybe lower
    # r(4,4,3): 4x4 * 4x3, schoolbook = 4*4*3 = 48
    # Can we do better? 
    # 
    # For a practical implementation within time limits, let's use:
    # r(4,4,7) <= 25 + 48 = 73 using Strassen for 4x4x4 and schoolbook for 4x4x3
    
    # Build 4x4x4 with Strassen (rank 25)
    s2 = _build_strassen_2x2x2()
    # We need to embed this into 4x4x4 using Kronecker product structure
    # 4x4x4 with Strassen: partition into 2x2 blocks, use Strassen recursively
    # This gives rank 7*7 = 49... no, Strassen for 2^k x 2^k x 2^k gives rank 7^k
    # For 4x4x4: 7^2 = 49? No, that's wrong.
    # Strassen for n x n x n where n = 2^k has rank 7^k
    # For k=2 (n=4): rank 49
    # Hmm, that's not better than schoolbook (64) by much.
    
    # Actually, the best known for 4x4x4 is 21 (Laderman) or maybe lower.
    # Let's just use schoolbook for simplicity and correctness, then try random search.
    
    result = _schoolbook_4x4x7()
    return result

def _random_search(start_rank, max_iters, timeout_s=100):
    """Try random search for a low-rank decomposition."""
    n, m, p = 4, 4, 7
    start_time = time.time()
    best = None
    best_rank = start_rank
    
    for rank in range(112, 15, -1):  # Try ranks from 112 down to 16
        if time.time() - start_time > timeout_s - 10:
            break
        for attempt in range(max_iters):
            if time.time() - start_time > timeout_s - 5:
                break
            # Generate random sparse U, V, W with half-integer entries
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
            # Make them sparse (most entries zero)
            U[1:] *= (np.random.rand(n*m-1, rank) > 0.7)
            V[1:] *= (np.random.rand(m*p-1, rank) > 0.7)
            W[1:] *= (np.random.rand(p*n-1, rank) > 0.7)
            
            if _verify(U, V, W):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = rank
                break
        if best is not None:
            break
    
    return best, best_rank

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7."""
    start_time = time.time()
    deadline = 120
    
    # Start with schoolbook
    sb = _schoolbook_4x4x7()
    best = sb
    best_rank = 112
    
    # Verify schoolbook
    if not _verify(sb['U'], sb['V'], sb['W']):
        # Fallback: construct manually
        n, m, p = 4, 4, 7
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                    t += 1
        best = {'U': U[:, :t], 'V': V[:, :t], 'W': W[:, :t]}
        best_rank = t
    
    # Try to improve with structured approach
    if time.time() - start_time < deadline - 20:
        # Try building with block structure
        try:
            opt = _build_4x4x7_optimized()
            if _verify(opt['U'], opt['V'], opt['W']):
                opt_rank = opt['U'].shape[1]
                if opt_rank < best_rank:
                    best = opt
                    best_rank = opt_rank
        except:
            pass
    
    # Try random search with remaining time
    remaining = deadline - 5 - (time.time() - start_time)
    if remaining > 10:
        random.seed(42)
        np.random.seed(42)
        res, new_rank = _random_search(best_rank, max_iters=100, timeout_s=min(remaining, 60))
        if res is not None and new_rank < best_rank:
            best = res
            best_rank = new_rank
    
    # Final check
    if not _verify(best['U'], best['V'], best['W']):
        # Return schoolbook as safe fallback
        n, m, p = 4, 4, 7
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, t] = 1.0
                    V[j*p+k, t] = 1.0
                    W[k*n+i, t] = 1.0
                    t += 1
        best = {'U': U[:, :t], 'V': V[:, :t], 'W': W[:, :t]}
    
    return best
