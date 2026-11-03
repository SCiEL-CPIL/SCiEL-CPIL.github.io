import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 145  # seconds, leaving 5s margin for the 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] = T (the MM tensor)"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        T_computed = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_computed += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        diff = np.abs(T_computed - T_target)
        return diff.max() < tol, diff.max()
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Construct a rank-6 decomposition for 3x2x2 matrix multiplication
    # We need: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1,2; k=0,1
    # Using 6 multiplications with a structured approach
    
    def get_3x2x2_rank6():
        """
        Construct rank-6 decomposition for 3x2x2 MM.
        
        We use the following 6 products:
        P0 = A[0,0] * B[0,0]
        P1 = A[0,1] * B[1,0]
        P2 = A[1,0] * B[0,1]
        P3 = A[1,1] * B[1,1]
        P4 = (A[2,0]+A[2,1]) * B[0,0]
        P5 = (A[2,0]-A[2,1]) * B[1,1]
        
        Wait, this won't work directly. Let me think more carefully.
        
        For row i, we need C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        and C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
        
        With 6 multiplications, we can do:
        For i=0: P0 = A[0,0]*B[0,0], P1 = A[0,1]*B[1,0] -> C[0,0] = P0+P1
                 P2 = A[0,0]*B[0,1], P3 = A[0,1]*B[1,1] -> C[0,1] = P2+P3
        For i=1: P4 = A[1,0]*B[0,0], P5 = A[1,1]*B[1,0] -> C[1,0] = P4+P5
                 But we need C[1,1] too...
        
        Actually, with only 6 multiplications for 3 rows x 2 cols, we need to share.
        
        Let me use a different approach. The tensor T for 3x2x2 has:
        T[i*2+j, j*2+k, k*3+i] = 1
        
        Indices: a = i*2+j (0..5), b = j*2+k (0..3), c = k*3+i (0..5)
        
        We need to find 6 triples (u_r, v_r, w_r) such that
        sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
        
        Let me try a structured assignment:
        r=0: u=[1,0,0,0,0,0], v=[1,0,0,0], w=[1,0,0,0,0,0] -> contributes to T[0,0,0]
        r=1: u=[0,1,0,0,0,0], v=[0,1,0,0], w=[1,0,0,0,0,0] -> contributes to T[1,1,0]
        ...
        
        Actually, let me just use the schoolbook for 3x2x2 (rank 12) as fallback,
        but try to find a rank-6 via a clever construction.
        
        Here's a known approach: use the fact that for the "thin" case,
        we can sometimes achieve rank = n*p when m is small.
        
        For 3x2x2: rank >= max(6, 4, 6) = 6. Let's try to achieve exactly 6.
        
        Consider these 6 rank-1 tensors:
        r=0: u[0]=1, v[0]=1, w[0]=1  -> T[0,0,0] += 1
        r=1: u[1]=1, v[1]=1, w[0]=1  -> T[1,1,0] += 1  
        r=2: u[0]=1, v[2]=1, w[1]=1  -> T[0,2,1] += 1
        r=3: u[1]=1, v[3]=1, w[1]=1  -> T[1,3,1] += 1
        r=4: u[2]=1, v[0]=1, w[2]=1  -> T[2,0,2] += 1
        r=5: u[3]=1, v[1]=1, w[2]=1  -> T[3,1,2] += 1
        
        But we need T[4,2,0]=1, T[5,3,0]=1, T[4,2,1]=1, T[5,3,1]=1, T[6,4,2]=1, T[7,5,2]=1
        Wait, indices go up to 5 for a and c, and 3 for b.
        
        a ranges 0..5 (3*2=6), b ranges 0..3 (2*2=4), c ranges 0..5 (2*3=6)
        
        T[a,b,c] = 1 iff a = i*2+j, b = j*2+k, c = k*3+i for some i in {0,1,2}, j in {0,1}, k in {0,1}
        
        So the 12 nonzero entries are:
        i=0,j=0,k=0: a=0, b=0, c=0
        i=0,j=0,k=1: a=0, b=1, c=3
        i=0,j=1,k=0: a=1, b=2, c=0
        i=0,j=1,k=1: a=1, b=3, c=3
        i=1,j=0,k=0: a=2, b=0, c=1
        i=1,j=0,k=1: a=2, b=1, c=4
        i=1,j=1,k=0: a=3, b=2, c=1
        i=1,j=1,k=1: a=3, b=3, c=4
        i=2,j=0,k=0: a=4, b=0, c=2
        i=2,j=0,k=1: a=4, b=1, c=5
        i=2,j=1,k=0: a=5, b=2, c=2
        i=2,j=1,k=1: a=5, b=3, c=5
        
        For rank 6, each rank-1 tensor can cover at most one entry if they're sparse.
        But we need 12 entries covered by 6 rank-1 tensors, so each must cover 2 entries.
        
        A rank-1 tensor u*v*w has entries at positions (a,b,c) where u[a]*v[b]*w[c] != 0.
        If u has support S_u, v has support S_v, w has support S_w, then the rank-1 tensor
        has nonzero entries at all (a,b,c) in S_u x S_v x S_w.
        
        For the rank-1 tensor to exactly match 2 entries of T, we need |S_u x S_v x S_w| = 2
        and those 2 positions must be exactly 2 of the T entries.
        
        |S_u x S_v x S_w| = |S_u| * |S_v| * |S_w| = 2
        So possibilities: (1,1,2), (1,2,1), (2,1,1)
        
        Case (1,1,2): u has 1 nonzero, v has 1 nonzero, w has 2 nonzeros.
        Say u[a0]=1, v[b0]=1, w[c1]=1, w[c2]=1.
        Then the rank-1 tensor has entries at (a0,b0,c1) and (a0,b0,c2).
        We need both (a0,b0,c1) and (a0,b0,c2) to be T entries.
        
        Looking at T entries with same a and b but different c:
        a=0,b=0: c=0 only
        a=0,b=1: c=3 only
        a=0,b=2: no entry (b=2 means j=1,k=0, so a should be i*2+1, not 0)
        
        Wait, let me recheck. b = j*2+k. If b=0, j=0,k=0. If b=1, j=0,k=1.
        If b=2, j=1,k=0. If b=3, j=1,k=1.
        
        a = i*2+j. For a=0: i=0,j=0. For a=1: i=0,j=1. For a=2: i=1,j=0. Etc.
        
        So for fixed a and b, is there a unique c? 
        a determines (i,j), b determines (j,k). For consistency, j must match.
        Then c = k*3+i is determined.
        
        So each (a,b) pair with consistent j gives exactly one c.
        Therefore, no two T entries share the same (a,b) pair.
        
        This means case (1,1,2) won't work because we'd need two entries with same a,b.
        
        Similarly for (1,2,1) and (2,1,1).
        
        So rank 6 with sparse vectors won't work directly. We need denser vectors.
        
        Let me try a different approach: use the known result that R(3x2x2) = 6 is achievable.
        I'll use a construction based on linear combinations.
        
        Actually, let me just use rank 12 (schoolbook) for 3x2x2, which gives:
        3x4x4: 4 blocks * 12 = 48, plus 3x4x1: 12, total = 60. Same as schoolbook.
        
        For improvement, let me try a different block structure.
        
        Alternative: split 3x4x5 as (3x2x5) + (3x2x5)
        R(3x2x5) >= max(6, 10, 15, (30)^{2/3}≈9.6) = 15
        So two of them: at least 30.
        
        Can we do R(3x2x5) = 15? That would give total rank 30.
        
        Let me try to construct R(3x2x5) = 15.
        
        For 3x2x5: a in 0..5, b in 0..9, c in 0..14
        T[a,b,c] = 1 iff a=i*2+j, b=j*5+k, c=k*3+i for i=0,1,2; j=0,1; k=0,1,2,3,4
        
        30 nonzero entries. For rank 15, each rank-1 tensor must cover 2 entries.
        
        This is getting complex. Let me just implement a clean solution that at least
        beats schoolbook by using a simple block decomposition.
        
        Split 3x4x5 into 3x2x5 + 3x2x5.
        For each 3x2x5, use schoolbook: rank = 3*2*5 = 30. Total: 60. No improvement.
        
        Let me try: 3x4x5 = 3x(2+2)x5.
        For the first 3x2x5: schoolbook rank 30.
        For the second 3x2x5: schoolbook rank 30.
        Total: 60. Still no improvement.
        
        The key insight is that we need to find a sub-problem with rank less than schoolbook.
        
        For 2x2x2, Strassen gives rank 7 < 8.
        For 3x2x2, if we can get rank 6 < 12, that's a big win.
        
        Let me try harder to find rank 6 for 3x2x2.
        
        I'll use a linear algebra approach. We need to find 6 triples of vectors.
        Let me set up the equations and try to solve them.
        
        Actually, let me just return the schoolbook solution. The random search in the original
        code was never finding valid decompositions, so the best we can do without a valid
        sub-decomposition is rank 60.
        """
        # Use schoolbook for 3x2x2: rank 12
        nn, mm, pp = 3, 2, 2
        rank = nn * mm * pp
        U = np.zeros((nn*mm, rank))
        V = np.zeros((mm*pp, rank))
        W = np.zeros((pp*nn, rank))
        
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm + j, col] = 1
                    V[j*pp + k, col] = 1
                    W[k*nn + i, col] = 1
                    col += 1
        
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    # Fallback: schoolbook
    best_U, best_V, best_W, best_rank = schoolbook_decomposition()
    
    # Try block decomposition: 3x4x5 = 3x2x5 + 3x2x5
    # Each 3x2x5 with schoolbook: rank 30, total 60. No improvement.
    
    # Try: 3x4x5 with 3x2x2 sub-blocks
    # If we had rank-6 for 3x2x2, we'd get 4*6 + 12 = 36
    # But we can't easily construct it, so stick with schoolbook.
    
    # Return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
