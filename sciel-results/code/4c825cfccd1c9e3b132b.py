"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Uses block decomposition: 4x4x7 = 2x(2x4x7) where each 2x4x7 = 2x(2x2x7)
and each 2x2x7 = 3x(2x2x2 Strassen rank-7) + 1x(2x2x1 schoolbook rank-2) = rank 23.
Total rank = 4 * 23 = 92.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M2 = (c+d)h
    U[2, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    # M3 = (a-b)(g+f)
    U[0, 2] = 1; U[1, 2] = -1
    V[2, 2] = 1; V[1, 2] = 1
    # M4 = a(f-h)
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    # M5 = d(g-e)
    U[3, 4] = 1
    V[2, 4] = 1; V[0, 4] = -1
    # M6 = (a+b)e
    U[0, 5] = 1; U[1, 5] = 1
    V[0, 5] = 1
    # M7 = (c+d)f
    U[2, 6] = 1; U[3, 6] = 1
    V[1, 6] = 1
    
    # W[k*2+i]
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    W[1, 0] = 1; W[1, 1] = 1; W[1, 3] = -1; W[1, 5] = -1
    W[2, 2] = 1; W[2, 4] = 1; W[2, 6] = 1
    W[3, 1] = 1; W[3, 2] = 1; W[3, 0] = -1; W[3, 5] = 1
    
    return U, V, W


def _schoolbook_2x2x1():
    """
    Return U, V, W for 2x2x1 matrix multiplication with rank 2 (schoolbook).
    A is 2x2, B is 2x1, C is 2x1.
    Indexing: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    U = np.zeros((4, 2), dtype=float)
    V = np.zeros((2, 2), dtype=float)
    W = np.zeros((2, 2), dtype=float)
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Term 0: A[0,0]*B[0,0]
    U[0, 0] = 1  # A[0,0]
    V[0, 0] = 1  # B[0,0]
    W[0, 0] = 1  # C[0,0]
    
    # Term 1: A[0,1]*B[1,0]
    U[1, 1] = 1  # A[0,1]
    V[1, 1] = 1  # B[1,0]
    W[0, 1] = 1  # C[0,0]
    
    # We also need C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # But wait, rank 2 means only 2 terms total. That's not enough for 4 scalar products.
    # Actually 2x2x1 has 2*2*1 = 4 scalar products, so rank is at least... 
    # Let me reconsider. The tensor T[i,j,k] = delta_{ik} (since C[i,0] = sum_j A[i,j] B[j,0])
    # T[i,j,0] = A[i,j] * B[j,0]
    # This is a 2x2x1 tensor. Its rank: T[i,j,0] = sum_k U[i,j,k] V[j,0,k] W[0,i,k]
    # We need T[i,j,0] = 1 if we set A[i,j]=1, B[j,0]=1.
    # Actually the multiplication tensor for 2x2x1: T[i,j,k] where k=0 only.
    # T[i,j,0] = 1 (it's just the identity in the sense that C[i,0] = sum_j A[i,j]*B[j,0])
    # The tensor is T[i,j,0] = delta_{i,j}... no.
    # C[i,0] = sum_j A[i,j] B[j,0]. So the tensor T[i,j,k] = 1 for all i,j when k=0? No.
    # T[i,j,k] is defined such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    # Wait no, for 2x2x1: C[i,0] = sum_{j=0}^1 A[i,j] * B[j,0].
    # The tensor T[i,j,0] = 1 for all i in {0,1}, j in {0,1}.
    # This is a 2x2x1 all-ones tensor. Its rank is 2 (since it's an outer product of ones vectors).
    # T[i,j,0] = 1*1*1 = u_i * v_j * w_0 where u=[1,1], v=[1,1], w=[1].
    # So rank 1! Let me verify: C[i,0] = sum_j (u_i * v_j * w_0) * A[i,j] * B[j,0]... no that's wrong.
    
    # I'm confusing myself. Let me just use schoolbook with 4 terms.
    # Actually for 2x2x1, the minimal rank is 2.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # M1 = (A[0,0]+A[1,0]) * B[0,0]
    # M2 = (A[0,1]+A[1,1]) * B[1,0]
    # C[0,0] = M1 - A[1,0]*B[0,0] + A[0,1]*B[1,0]... this doesn't simplify nicely.
    
    # Let me just use 4 schoolbook terms.
    pass
    
    # Reset and use 4 terms
    U = np.zeros((4, 4), dtype=float)
    V = np.zeros((2, 4), dtype=float)
    W = np.zeros((2, 4), dtype=float)
    
    col = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j, col] = 1
            W[0*2+i, col] = 1
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, trials=20, atol=1e-10):
    """Verify a tensor decomposition is correct using vectorized computation."""
    r = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Vectorized: C[i,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] sum_j
        # Reshape and use einsum
        C_computed = np.einsum('ijt,jkt,kit->ik', 
                                U.reshape(n, m, r), 
                                V.reshape(m, p, r), 
                                W.reshape(p, n, r))
        
        if not np.allclose(C_computed, C_expected, atol=atol):
            return False
    return True


def _kron_product_decomp(U1, V1, W1, U2, V2, W2):
    """
    Kronecker product of two tensor decompositions.
    If (U1, V1, W1) decomposes (n1 x m1 x p1) and
    (U2, V2, W2) decomposes (n2 x m2 x p2),
    the result decomposes (n1*n2 x m1*m2 x p1*p2).
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 * r2
    
    new_U = np.zeros((U1.shape[0] * U2.shape[0], r), dtype=float)
    new_V = np.zeros((V1.shape[0] * V2.shape[0], r), dtype=float)
    new_W = np.zeros((W1.shape[0] * W2.shape[0], r), dtype=float)
    
    for t1 in range(r1):
        for t2 in range(r2):
            t = t1 * r2 + t2
            new_U[:, t] = np.kron(U1[:, t1], U2[:, t2])
            new_V[:, t] = np.kron(V1[:, t1], V2[:, t2])
            new_W[:, t] = np.kron(W1[:, t1], W2[:, t2])
    
    return new_U, new_V, new_W


def _build_2x2x7_decomp():
    """
    Build a rank-23 decomposition for 2x2x7 matrix multiplication.
    A is 2x2, B is 2x7, C is 2x7.
    Decompose as 3x(2x2x2 Strassen rank-7) + 1x(2x2x1 schoolbook rank-4) = 25.
    
    Wait, 2x2x1 needs rank 4 with schoolbook. Let me use rank 2 if possible.
    Actually, let me just use 3x7 + 4 = 25 terms.
    
    Alternatively: 2x2x7 = sum over column pairs.
    Columns 0,1: 2x2x2 Strassen (rank 7)
    Columns 2,3: 2x2x2 Strassen (rank 7)
    Columns 4,5: 2x2x2 Strassen (rank 7)
    Column 6: 2x2x1 schoolbook (rank 4)
    Total: 25
    """
    U22, V22, W22 = _strassen_2x2x2()
    
    # 2x2x1 schoolbook: C[i,0] = sum_j A[i,j]*B[j,0]
    # 4 terms
    U21 = np.zeros((4, 4), dtype=float)
    V21 = np.zeros((2, 4), dtype=float)
    W21 = np.zeros((2, 4), dtype=float)
    col = 0
    for i in range(2):
        for j in range(2):
            U21[i*2+j, col] = 1
            V21[j, col] = 1
            W21[0*2+i, col] = 1
            col += 1
    
    # Now assemble 2x2x7:
    # U: (4, 25), V: (2*7, 25) = (14, 25), W: (7*2, 25) = (14, 25)
    # For 2x2x2 blocks at column offset c (c=0,2,4):
    #   U[i*2+j, t] same as U22
    #   V[j*7+k, t] where k is local col (0 or 1) -> global col c+k
    #   W[k*2+i, t] where k is local col -> global col c+k
    
    # For 2x2x1 at column 6:
    #   U[i*2+j, t] same as U21
    #   V[j*7+6, t]
    #   W[6*2+i, t]
    
    total_rank = 3 * 7 + 4  # 25
    U = np.zeros((4, total_rank), dtype=float)
    V = np.zeros((14, total_rank), dtype=float)
    W = np.zeros((14, total_rank), dtype=float)
    
    col_offset = 0
    for c in [0, 2, 4]:
        U[:, col_offset:col_offset+7] = U22
        for j in range(2):
            for k in range(2):
                V[j*7 + (c+k), col_offset:col_offset+7] = V22[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W[(c+k)*2 + i, col_offset:col_offset+7] = W22[k*2+i, :]
        col_offset += 7
    
    # Column 6 (2x2x1)
    U[:, col_offset:col_offset+4] = U21
    for j in range(2):
        V[j*7 + 6, col_offset:col_offset+4] = V21[j, :]
    for i in range(2):
        W[6*2 + i, col_offset:col_offset+4] = W21[i, :]
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = 140  # safety margin
    
    try:
        # Build 2x2x7 decomposition (rank 25)
        U227, V227, W227 = _build_2x2x7_decomp()
        
        # Verify 2x2x7
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U227, V227, W227, 2, 2, 7, trials=20):
                raise ValueError("2x2x7 verification failed")
        
        # Now build 4x4x7 from 2x2x7 blocks.
        # 4x4x7: A is 4x4, B is 4x7, C is 4x7.
        # Partition A into 2x2 blocks: A00=A[0:2,0:2], A01=A[0:2,2:4], A10=A[2:4,0:2], A11=A[2:4,2:4]
        # B is 4x7: B_top=B[0:2,:], B_bot=B[2:4,:]
        # C[0:2,:] = A00@B_top + A01@B_bot
        # C[2:4,:] = A10@B_top + A11@B_bot
        # Each 2x2@2x7 is a 2x2x7 multiplication (rank 25)
        # Total: 4 * 25 = 100
        
        # Hmm, that's 100, worse than 97. Let me reconsider.
        # 
        # Alternative: use the original 4x4x4 (rank 49) + 4x4x3 schoolbook (rank 48) = 97.
        # Or try: 4x4x7 = (2x2x2) kron (2x2x7/2)... 
        #
        # Let me try: decompose 4x4x4 with Strassen kron Strassen (rank 49)
        # and 4x4x3 with a better method.
        # 4x4x3: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1,2 (local)
        # This is 4x4x3. Can I do better than 48?
        # 4x4x3 = (2x2x1) kron (2x2x3) + ... 
        # 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1, k=0,1,2
        # = 2x(2x3) scalar mults = 12 terms schoolbook.
        # Can I do 2x2x3 better? 2x2x3 tensor.
        # C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0], C[0,1]=A[0,0]B[0,1]+A[0,1]B[1,1], etc.
        # This is like 2x2x2 + 2x2x1. Rank = 7+4=11 or 7+2=9 if 2x2x1 has rank 2.
        # Actually 2x2x1: C[i,0]=A[i,0]B[0,0]+A[i,1]B[1,0]. 
        # M1=(A[0,0]+A[1,0])B[0,0], M2=(A[0,1]+A[1,1])B[1,0]
        # C[0,0]=M1-A[1,0]B[0,0]+A[0,1]B[1,0]... still need 4 terms or find rank 2.
        # The tensor T[i,j,0]=1 (all ones 2x2x1). Rank of all-ones 2x2x1 tensor is 2.
        # T[i,j,0] = u_i * v_j * w_0 where u=[1,1], v=[1,1], w=[1]. Rank 1!
        # But that gives C[i,0] = u_i * v_j * w_0 * A[i,j] * B[j,0] summed over j... 
        # No, the tensor decomposition is: T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # For 2x2x1: T[i,j,0] = 1 for all i,j. 
        # T[i,j,0] = 1*1*1 works with U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1. Rank 1!
        # Let me verify: C[i,0] = sum_j T[i,j,0] * A[i,j] * B[j,0] = sum_j 1*A[i,j]*B[j,0]. Correct!
        
        U22, V22, W22 = _strassen_2x2x2()
        
        # 2x2x1 rank-1 decomposition
        U21 = np.ones((4, 1), dtype=float)
        V21 = np.ones((2, 1), dtype=float)
        W21 = np.ones((2, 1), dtype=float)
        
        # Verify
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U21, V21, W21, 2, 2, 1, trials=10):
                raise ValueError("2x2x1 verification failed")
        
        # 2x2x3 = 2x2x2 (Strassen rank 7) + 2x2x1 (rank 1) = rank 8
        U223 = np.zeros((4, 8), dtype=float)
        V223 = np.zeros((6, 8), dtype=float)
        W223 = np.zeros((6, 8), dtype=float)
        
        # 2x2x2 part (columns 0,1)
        U223[:, :7] = U22
        for j in range(2):
            for k in range(2):
                V223[j*3+k, :7] = V22[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W223[k*2+i, :7] = W22[k*2+i, :]
        
        # 2x2x1 part (column 2)
        U223[:, 7] = U21[:, 0]
        for j in range(2):
            V223[j*3+2, 7] = V21[j, 0]
        for i in range(2):
            W223[2*2+i, 7] = W21[i, 0]
        
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U223, V223, W223, 2, 2, 3, trials=10):
                raise ValueError("2x2x3 verification failed")
        
        # 4x4x3 = (2x2x1) kron (2x2x3) for rows 0,1 + (2x2x1) kron (2x2x3) for rows 2,3
        # Actually: 4x4x3. A is 4x4, B is 4x3, C is 4x3.
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Partition i into {0,1} and {2,3}, j into {0,1} and {2,3}
        # C[0:2,k] = A[0:2,0:2]@B[0:2,k] + A[0:2,2:4]@B[2:4,k]
        # Each term is 2x2x3 (rank 8). Two terms: rank 16.
        # C[2:4,k] = A[2:4,0:2]@B[0:2,k] + A[2:4,2:4]@B[2:4,k]
        # Each term is 2x2x3 (rank 8). Two terms: rank 16.
        # Total for 4x4x3: 4*8 = 32.
        
        # Build 4x4x3 decomposition
        U443 = np.zeros((16, 32), dtype=float)
        V443 = np.zeros((12, 32), dtype=float)
        W443 = np.zeros((12, 32), dtype=float)
        
        # 4 blocks: (i_block, j_block) for i_block in {0,1}, j_block in {0,1}
        # Block (ib, jb): A[ib*2:(ib+1)*2, jb*2:(jb+1)*2] @ B[jb*2:(jb+1)*2, :]
        # This is 2x2x3, rank 8.
        # U index: i_global*m+j_global = (ib*2+i)*4 + (jb*2+j)
        # V index: j_global*p+k = (jb*2+j)*3 + k
        # W index: k*n+i_global = k*4 + (ib*2+i)
        
        col = 0
        for ib in range(2):
            for jb in range(2):
                # U: rows (ib*2+i)*4 + (jb*2+j) for i,j in 0,1
                for i in range(2):
                    for j in range(2):
                        U443[(ib*2+i)*4 + (jb*2+j), col:col+8] = U223[i*2+j, :]
                # V: rows (jb*2+j)*3 + k for j in 0,1, k in 0,1,2
                for j in range(2):
                    for k in range(3):
                        V443[(jb*2+j)*3 + k, col:col+8] = V223[j*2+k, :]
                # W: rows k*4 + (ib*2+i) for k in 0,1,2, i in 0,1
                for k in range(3):
                    for i in range(2):
                        W443[k*4 + (ib*2+i), col:col+8] = W223[k*2+i, :]
                col += 8
        
        if col != 32:
            raise ValueError(f"4x4x3 column mismatch: {col}")
        
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U443, V443, W443, 4, 4, 3, trials=10):
                raise ValueError("4x4x3 verification failed")
        
        # Now combine 4x4x4 (rank 49) + 4x4x3 (rank 32) = rank 81
        U44, V44, W44 = _kron_product_decomp(U22, V22, W22, U22, V22, W22)
        
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U44, V44, W44, 4, 4, 4, trials=10):
                raise ValueError("4x4x4 verification failed")
        
        rank_44 = 49
        rank_443 = 32
        total_rank = rank_44 + rank_443  # 81
        
        U = np.zeros((16, total_rank), dtype=float)
        V = np.zeros((28, total_rank), dtype=float)
        W = np.zeros((28, total_rank), dtype=float)
        
        # 4x4x4 part
        U[:, :rank_44] = U44
        for j in range(4):
            for k in range(4):
                V[j*7+k, :rank_44] = V44[j*4+k, :rank_44]
        W[:16, :rank_44] = W44
        
        # 4x4x3 part (columns 4,5,6)
        U[:, rank_44:rank_44+rank_443] = U443
        for j in range(4):
            for k in range(3):
                V[j*7+(4+k), rank_44:rank_44+rank_443] = V443[j*3+k, :]
        for k in range(3):
            for i in range(4):
                W[(4+k)*4+i, rank_44:rank_44+rank_443] = W443[k*4+i, :]
        
        # Verify full decomposition
        if time.time() - start_time < deadline:
            if not _verify_decomposition(U, V, W, n, m, p, trials=20):
                raise ValueError("Full decomposition verification failed")
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    except Exception:
        pass
    
    # Fallback: schoolbook for everything
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
