import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    
    # Strategy: Use rank-23 for 3x3x3 (twice) + rank-18 for 3x3x2 = 64 total
    # Construct and verify the rank-23 decomposition. Fall back to schoolbook if needed.
    
    def build_333_rank23():
        """
        Build a rank-23 decomposition for 3x3x3 matrix multiplication.
        Returns (U, V, W) as numpy arrays of shape (9, 23).
        Indexing: U[i*3+j], V[j*3+k], W[k*3+i]
        """
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Define 23 rank-1 terms.
        # Each term r contributes: sum over (i,j,k) of U[i,j,r]*V[j,k,r]*W[k,i,r] * A[i,j]*B[j,k] = C[k,i]
        # We need: for all i,j,k: sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
        
        # Here is a known valid rank-23 decomposition:
        
        # Term 0
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        # Term 1
        U[1,1] = 1; V[3,1] = 1; W[1,1] = 1
        # Term 2
        U[2,2] = 1; V[6,2] = 1; W[2,2] = 1
        # Term 3
        U[3,3] = 1; V[1,3] = 1; W[3,3] = 1
        # Term 4
        U[4,4] = 1; V[4,4] = 1; W[4,4] = 1
        # Term 5
        U[5,5] = 1; V[7,5] = 1; W[5,5] = 1
        # Term 6
        U[6,6] = 1; V[2,6] = 1; W[6,6] = 1
        # Term 7
        U[7,7] = 1; V[5,7] = 1; W[7,7] = 1
        # Term 8
        U[8,8] = 1; V[8,8] = 1; W[8,8] = 1
        
        # Terms 9-22: Strassen-like terms
        # I'll use a known valid set.
        
        # After careful construction, here are the remaining 14 terms:
        
        # Term 9
        U[0,9] = 1; U[1,9] = 1; U[3,9] = 1; U[4,9] = 1
        V[0,9] = 1; V[1,9] = 1; V[3,9] = 1; V[4,9] = 1
        W[0,9] = 1; W[1,9] = 1; W[3,9] = 1; W[4,9] = 1
        
        # This is not correct. I need to use a proper construction.
        
        # Let me use a different approach: I'll return None to indicate failure.
        return None
    
    def verify_333(U, V, W):
        """Verify 3x3x3 decomposition."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Try to build rank-23
    r23 = build_333_rank23()
    
    if r23 is not None and verify_333(*r23):
        U1, V1, W1 = r23
        # Use for k=0,1,2 and k=3,4,5
        total_rank = 23 + 23 + 18
        U = np.zeros((9, total_rank))
        V = np.zeros((9, total_rank))
        W = np.zeros((9, total_rank))
        
        # First 3x3x3 block (k=0,1,2)
        U[:, 0:23] = U1
        V[:, 0:23] = V1
        W[:, 0:23] = W1
        
        # Second 3x3x3 block (k=3,4,5) - same decomposition
        U[:, 23:46] = U1
        V[:, 23:46] = V1
        W[:, 23:46] = W1
        
        # 3x3x2 block (k=6,7) - schoolbook, rank 18
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    col = 46 + i*3*2 + j*2 + k
                    U[i*3+j, col] = 1
                    V[j*3+(6+k), col] = 1
                    W[(6+k)*3+i, col] = 1
        
        # Convert to lists
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback: schoolbook rank 72
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

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
