"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = U.shape[1]
    # Build the matrix multiplication tensor: T[i, j, k, l] = 1 if j==k else 0
    # In the factorization: T[i, j, k, l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we check: for all i, j, k, l: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == (1 if j==k else 0)
    # We can do this efficiently by reshaping.
    # Let's verify on a random matrix product to be safe and fast.
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C from the tensor factors
    # C[i, k] = sum_{j, l} A[i, j] * B[j, k] * (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r])
    # But the tensor entry T[i, j, k, l] should be 1 if j==k else 0.
    # So C[i, k] = sum_j A[i, j] * B[j, k] * (1 if j==k else 0) = sum_j A[i, j] * B[j, k] * delta(j,k)
    # Wait, the standard matrix multiplication is C[i, k] = sum_j A[i, j] * B[j, k].
    # The tensor for matrix multiplication is T_{i, j, k} = delta_{j, k}? No.
    # The tensor T is such that C_{i, k} = sum_{j} A_{i, j} B_{j, k}.
    # In tensor notation, the multiplication tensor has indices (i, j, k) and value 1 if the "middle" indices match?
    # Actually, the tensor M for matrix multiplication of size n x m x p is:
    # M[i, j, k] = 1 if j == k (where j is from 1..m and k is from 1..m? No, j is column of A, row of B).
    # Let's use the factorization definition:
    # C_{i, k} = sum_r W_{k, i, r} * (sum_j A_{i, j} U_{i, j, r}) * (sum_j B_{j, k} V_{j, k, r})
    # This doesn't look right.
    # Standard tensor decomposition for matmul:
    # C_{i, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # Here, the indices are:
    # U is indexed by (i, j, r) -> flattened to (i*m+j, r)
    # V is indexed by (j, k, r) -> flattened to (j*p+k, r)
    # W is indexed by (k, i, r) -> flattened to (k*n+i, r)
    # So C_{i, k} = sum_r (sum_j A_{i, j} U_{i, j, r}) (sum_j B_{j, k} V_{j, k, r}) W_{k, i, r}
    # Wait, the sum over j is separate for U and V? No.
    # Let's re-read the standard definition.
    # The matrix multiplication tensor M_{n, m, p} is a 3-tensor with indices (i, j, k) where 1<=i<=n, 1<=j<=m, 1<=k<=p.
    # M[i, j, k] = 1 if j == k (assuming m=p), else 0.
    # A rank-r decomposition is M[i, j, k] = sum_{t=1}^r U[i, j, t] V[j, k, t] W[k, i, t].
    # Here U[i, j, t] is U[i*m+j, t], V[j, k, t] is V[j*p+k, t], W[k, i, t] is W[k*n+i, t].
    # So we need to check: for all i, j, k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == (1 if j==k else 0).
    # Let's verify this condition.
    U_flat = U
    V_flat = V
    W_flat = W
    # We can check a subset or use random sampling if rank is large, but for 49 it's small.
    # Let's check all i, j, k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over t
                s = 0.0
                for t in range(rank):
                    s += U_flat[i*m+j, t] * V_flat[j*p+k, t] * W_flat[k*n+i, t]
                expected = 1.0 if j == k else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _strassen_2x2_factors():
    """Return the U, V, W factors for 2x2 matrix multiplication using Strassen's algorithm."""
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # We need to express this as U, V, W such that C[i, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t].
    # For 2x2, n=m=p=2.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}.
    # U[i*m+j, t] -> U[i*2+j, t]
    # V[j*p+k, t] -> V[j*2+k, t]
    # W[k*n+i, t] -> W[k*2+i, t]
    
    # Let's build U, V, W as 4x7 matrices.
    # t=0: M1 = (A11 + A22)(B11 + B22)
    # This contributes to C11, C22.
    # C11 = ... + M1 ...
    # C22 = M1 ...
    # So for t=0:
    # U: A11 + A22 -> U[0,0]=1, U[3,0]=1 (A11 is (0,0), A22 is (1,1) -> indices 0 and 3)
    # V: B11 + B22 -> V[0,0]=1, V[3,0]=1 (B11 is (0,0), B22 is (1,1) -> indices 0 and 3)
    # W: C11 + C22 -> W[0,0]=1, W[3,0]=1 (C11 is (0,0) -> k*n+i = 0*2+0=0, C22 is (1,1) -> 1*2+1=3)
    #
    # t=1: M2 = (A21 + A22)B11
    # Contributes to C21, C22.
    # U: A21 + A22 -> U[2,1]=1, U[3,1]=1
    # V: B11 -> V[0,1]=1
    # W: C21 + C22 -> C21 is (1,0) -> 1*2+0=2, C22 is (1,1) -> 3. So W[2,1]=1, W[3,1]=-1 (since C22 has -M2)
    # Wait, C22 = M1 - M2 + M3 + M6. So W[3,1] should be -1.
    #
    # t=2: M3 = A11(B12 - B22)
    # Contributes to C12, C22.
    # U: A11 -> U[0,2]=1
    # V: B12 - B22 -> V[1,2]=1, V[3,2]=-1
    # W: C12 + C22 -> C12 is (0,1) -> 1*2+0=2, C22 is (1,1) -> 3. So W[2,2]=1, W[3,2]=1.
    #
    # t=3: M4 = A22(B21 - B11)
    # Contributes to C11, C21.
    # U: A22 -> U[3,3]=1
    # V: B21 - B11 -> V[2,3]=1, V[0,3]=-1
    # W: C11 + C21 -> C11 is (0,0) -> 0, C21 is (1,0) -> 2. So W[0,3]=1, W[2,3]=1.
    #
    # t=4: M5 = (A11 + A12)B22
    # Contributes to C11, C12.
    # U: A11 + A12 -> U[0,4]=1, U[1,4]=1
    # V: B22 -> V[3,4]=1
    # W: C11 - C12? C11 = ... - M5 ..., C12 = ... + M5 ...
    # So W[0,4]=-1, W[2,4]=1.
    #
    # t=5: M6 = (A21 - A11)(B11 + B12)
    # Contributes to C22.
    # U: A21 - A11 -> U[2,5]=1, U[0,5]=-1
    # V: B11 + B12 -> V[0,5]=1, V[1,5]=1
    # W: C22 -> W[3,5]=1.
    #
    # t=6: M7 = (A12 - A22)(B21 + B22)
    # Contributes to C11.
    # U: A12 - A22 -> U[1,6]=1, U[3,6]=-1
    # V: B21 + B22 -> V[2,6]=1, V[3,6]=1
    # W: C11 -> W[0,6]=1.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # t=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # t=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # t=2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # t=3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # t=4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # t=5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # t=6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with the Strassen 2x2 factors
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Verify the 2x2 decomposition
    if not _verify(U2, V2, W2, 2, 2, 2):
        # Fallback to schoolbook if Strassen implementation is incorrect
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
    
    # Construct the 4x4 decomposition using Kronecker products
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    # Shapes: U2 is 4x7, so U4 is (4*4) x (7*7) = 16 x 49
    U2_np = np.array(U2, dtype=float)
    V2_np = np.array(V2, dtype=float)
    W2_np = np.array(W2, dtype=float)
    
    U4_np = np.kron(U2_np, U2_np)
    V4_np = np.kron(V2_np, V2_np)
    W4_np = np.kron(W2_np, W2_np)
    
    # Verify the 4x4 decomposition
    if not _verify(U4_np, V4_np, W4_np, 4, 4, 4):
        # Fallback to schoolbook if Kronecker construction is incorrect
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
    
    # Convert to lists of lists with integer entries (all entries are -1, 0, or 1)
    U4 = U4_np.astype(int).tolist()
    V4 = V4_np.astype(int).tolist()
    W4 = W4_np.astype(int).tolist()
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
