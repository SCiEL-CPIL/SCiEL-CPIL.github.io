import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 2, 3, 4
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # T should equal 1 for all (i,j,k) in range(n,m,p)
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check if T matches the identity tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i,j,k] - 1.0) > 1e-10:
                        return False
        return True
    
    def make_schoolbook(rank=None):
        """Return the schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def make_rank20():
        """
        Construct a rank-20 decomposition using two 2x3x2 blocks.
        Each 2x3x2 block uses a rank-10 decomposition.
        
        For 2x3x2 (A: 2x3, B: 3x2, C: 2x2):
        We use the following 10 multiplications:
        m1 = A[0,0]*B[0,0]
        m2 = A[0,1]*B[1,0]
        m3 = A[0,2]*B[2,0]
        m4 = A[1,0]*B[0,0]
        m5 = A[1,1]*B[1,0]
        m6 = A[1,2]*B[2,0]
        m7 = (A[0,0]+A[0,1]+A[0,2])*(B[0,1]+B[1,1]+B[2,1])
        m8 = (A[1,0]+A[1,1]+A[1,2])*(B[0,1]+B[1,1]+B[2,1])
        m9 = (A[0,0]+A[0,1])*(B[0,1]+B[1,1])
        m10 = (A[1,0]+A[1,1])*(B[0,1]+B[1,1])
        
        C[0,0] = m1+m2+m3
        C[1,0] = m4+m5+m6
        C[0,1] = m7 - m9
        C[1,1] = m8 - m10
        
        Wait, let me verify C[0,1]:
        m7 = A[0,0]*B[0,1] + A[0,0]*B[1,1] + A[0,0]*B[2,1] +
             A[0,1]*B[0,1] + A[0,1]*B[1,1] + A[0,1]*B[2,1] +
             A[0,2]*B[0,1] + A[0,2]*B[1,1] + A[0,2]*B[2,1]
        m9 = A[0,0]*B[0,1] + A[0,0]*B[1,1] + A[0,1]*B[0,1] + A[0,1]*B[1,1]
        m7-m9 = A[0,0]*B[2,1] + A[0,1]*B[2,1] + A[0,2]*B[0,1] + A[0,2]*B[1,1] + A[0,2]*B[2,1]
        But we want C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        
        This doesn't match. Let me use a different approach.
        
        Actually, for 2x3x2, the rank is indeed 10, but the decomposition is non-trivial.
        Let me use a known valid one from the literature.
        
        Alternative: use schoolbook for each 2x3x1 column of B.
        For B col 0: 6 mults (2x3x1)
        For B col 1: 6 mults (2x3x1)
        Total: 12 per 2x3x2 block, 24 total.
        
        To get rank 20, I need to find a rank-10 decomposition for 2x3x2.
        
        Let me try a different set of 10 mults:
        m1 = A[0,0]*B[0,0]
        m2 = A[0,1]*B[1,0]
        m3 = A[0,2]*B[2,0]
        m4 = A[1,0]*B[0,0]
        m5 = A[1,1]*B[1,0]
        m6 = A[1,2]*B[2,0]
        m7 = (A[0,0]+A[0,2])*(B[0,1]+B[2,1])
        m8 = (A[1,0]+A[1,2])*(B[0,1]+B[2,1])
        m9 = (A[0,1]+A[0,2])*(B[1,1]+B[2,1])
        m10 = (A[1,1]+A[1,2])*(B[1,1]+B[2,1])
        
        C[0,0] = m1+m2+m3
        C[1,0] = m4+m5+m6
        C[0,1] = m7 - A[0,0]*B[2,1] - A[0,2]*B[0,1]  -- still has extra terms
        
        This is getting complicated. Let me just use the schoolbook rank 24 for now,
        but try to find a better decomposition via search if time permits.
        """
        # For now, return schoolbook 24
        return make_schoolbook()
    
    # Try to find a better decomposition
    best_result = make_schoolbook()
    
    # Try rank 20 with a systematic approach
    # The 2x3x4 tensor has a natural block structure
    # We can try to find a decomposition that exploits this
    
    # For now, just return the schoolbook decomposition
    # which is guaranteed correct
    return best_result
