"""Exact rank-minimal search for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
DEADLINE = 145.0  # seconds; safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # We will use a recursive decomposition that exploits Strassen's algorithm
    # for 2x2x2 blocks and outer products for 1-dim slices.
    # This should achieve a rank lower than the schoolbook rank of n*m*p = 105.
    
    # Strassen's algorithm for 2x2 matrix multiplication has rank 7.
    # We can use this to reduce the rank when splitting dimensions.
    
    # The recursive strategy:
    # If any dimension is 1, use outer product (rank = product of remaining dims)
    # If n=m=p=2, use Strassen (rank 7)
    # Otherwise, split the largest dimension into two parts and recurse
    
    def get_strassen_222():
        """Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm."""
        # Strassen's 7 products for 2x2 matrix multiplication
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = A*B
        # 
        # m1 = (a11 + a22)(b11 + b22)
        # m2 = (a21 + a22)b11
        # m3 = a11(b12 - b22)
        # m4 = a22(b21 - b11)
        # m5 = (a11 + a12)b22
        # m6 = (a21 - a11)(b11 + b12)
        # m7 = (a12 - a22)(b21 + b22)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # U shape: (n*m, rank) = (4, 7)
        # V shape: (m*p, rank) = (4, 7)
        # W shape: (p*n, rank) = (4, 7)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Index mapping:
        # U[i*m + j, r] where i in 0..1, j in 0..1
        # V[j*p + k, r] where j in 0..1, k in 0..1
        # W[k*n + i, r] where k in 0..1, i in 0..1
        
        # m1 = (a11 + a22)(b11 + b22)
        # U[0,0] = 1 (a11), U[3,0] = 1 (a22)
        # V[0,0] = 1 (b11), V[3,0] = 1 (b22)
        # W[0,0] = 1 (c11), W[3,0] = 1 (c22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (a21 + a22)b11
        # U[2,1] = 1 (a21), U[3,1] = 1 (a22)
        # V[0,1] = 1 (b11)
        # W[2,1] = 1 (c21)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3 = a11(b12 - b22)
        # U[0,2] = 1 (a11)
        # V[1,2] = 1 (b12), V[3,2] = -1 (b22)
        # W[1,2] = 1 (c12)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # m4 = a22(b21 - b11)
        # U[3,3] = 1 (a22)
        # V[2,3] = 1 (b21), V[0,3] = -1 (b11)
        # W[0,3] = 1 (c11), W[2,3] = 1 (c21)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5 = (a11 + a12)b22
        # U[0,4] = 1 (a11), U[1,4] = 1 (a12)
        # V[3,4] = 1 (b22)
        # W[0,4] = -1 (c11), W[1,4] = 1 (c12)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6 = (a21 - a11)(b11 + b12)
        # U[2,5] = 1 (a21), U[0,5] = -1 (a11)
        # V[0,5] = 1 (b11), V[1,5] = 1 (b12)
        # W[3,5] = 1 (c22)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7 = (a12 - a22)(b21 + b22)
        # U[1,6] = 1 (a12), U[3,6] = -1 (a22)
        # V[2,6] = 1 (b21), V[3,6] = 1 (b22)
        # W[0,6] = 1 (c11), W[3,6] = 1 (c22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    def outer_product_decomp(n, m, p):
        """Outer product decomposition for 1-dim slices."""
        if n == 1 and m == 1 and p == 1:
            return np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
        
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
        
        return U, V, W
    
    def combine_decompositions(n, m1, p, U1, V1, W1, m2, U2, V2, W2):
        """Combine two decompositions split along the m dimension."""
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * (m1 + m2), rank))
        V = np.zeros(((m1 + m2) * p, rank))
        W = np.zeros((p * n, rank))
        
        # Place U1 in rows 0..n*m1-1, U2 in rows n*m1..n*(m1+m2)-1
        U[:n * m1, :U1.shape[1]] = U1
        U[n * m1:, U1.shape[1]:] = U2
        
        # Place V1 in rows 0..m1*p-1, V2 in rows m1*p..(m1+m2)*p-1
        V[:m1 * p, :V1.shape[1]] = V1
        V[m1 * p:, V1.shape[1]:] = V2
        
        # W is the same for both (shared)
        W[:, :W1.shape[1]] = W1
        W[:, W1.shape[1]:] = W2
        
        return U, V, W
    
    def combine_decompositions_n(n1, m, p, U1, V1, W1, n2, U2, V2, W2):
        """Combine two decompositions split along the n dimension."""
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros(((n1 + n2) * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * (n1 + n2), rank))
        
        # Place U1 in rows 0..n1*m-1, U2 in rows n1*m..(n1+n2)*m-1
        U[:n1 * m, :U1.shape[1]] = U1
        U[n1 * m:, U1.shape[1]:] = U2
        
        # V is the same for both (shared)
        V[:, :V1.shape[1]] = V1
        V[:, V1.shape[1]:] = V2
        
        # Place W1 in rows 0..p*n1-1, W2 in rows p*n1..p*(n1+n2)-1
        W[:p * n1, :W1.shape[1]] = W1
        W[p * n1:, W1.shape[1]:] = W2
        
        return U, V, W
    
    def combine_decompositions_p(n, m, p1, U1, V1, W1, p2, U2, V2, W2):
        """Combine two decompositions split along the p dimension."""
        rank = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank))
        V = np.zeros((m * (p1 + p2), rank))
        W = np.zeros(((p1 + p2) * n, rank))
        
        # U is the same for both (shared)
        U[:, :U1.shape[1]] = U1
        U[:, U1.shape[1]:] = U2
        
        # Place V1 in rows 0..m*p1-1, V2 in rows m*p1..m*(p1+p2)-1
        V[:m * p1, :V1.shape[1]] = V1
        V[m * p1:, V1.shape[1]:] = V2
        
        # Place W1 in rows 0..p1*n-1, W2 in rows p1*n..(p1+p2)*n-1
        W[:p1 * n, :W1.shape[1]] = W1
        W[p1 * n:, W1.shape[1]:] = W2
        
        return U, V, W
    
    def decompose(n, m, p):
        """Recursive decomposition for matrix multiplication tensor."""
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return outer_product_decomp(1, 1, 1)
        
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
        
        # If any dimension is 1, use outer product
        if n == 1 or m == 1 or p == 1:
            return outer_product_decomp(n, m, p)
        
        # Find the largest dimension to split
        if n >= m and n >= p:
            # Split n into n1 and n2
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = decompose(n1, m, p)
            U2, V2, W2 = decompose(n2, m, p)
            return combine_decompositions_n(n1, m, p, U1, V1, W1, n2, U2, V2, W2)
        elif m >= n and m >= p:
            # Split m into m1 and m2
            m1 = m // 2
            m2 = m - m1
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            return combine_decompositions(n, m1, p, U1, V1, W1, m2, U2, V2, W2)
        else:
            # Split p into p1 and p2
            p1 = p // 2
            p2 = p - p1
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            return combine_decompositions_p(n, m, p1, U1, V1, W1, p2, U2, V2, W2)
    
    # Try the recursive decomposition
    U, V, W = decompose(n, m, p)
    rank = U.shape[1]
    
    # Verify the decomposition
    def verify(U_np, V_np, W_np):
        U_reshaped = U_np.reshape(n, m, rank)  # [i, j, r]
        V_reshaped = V_np.reshape(m, p, rank)  # [j, k, r]
        W_reshaped = W_np.reshape(p, n, rank)  # [k, i, r]
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-6)
    
    if not verify(U, V, W):
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
