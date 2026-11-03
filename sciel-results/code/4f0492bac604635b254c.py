import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
TIME_LIMIT = 140  # seconds, leaving 10s margin for the 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    def deadline_ok(margin=10):
        return time.time() - start_time < TIME_LIMIT - margin
    
    def make_target_tensor(nn, mm, pp):
        T = np.zeros((nn*mm, mm*pp, pp*nn))
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T[i*mm+j, j*pp+k, k*nn+i] = 1.0
        return T
    
    def verify(U, V, W, T_target, tol=1e-6):
        r = U.shape[1]
        T_comp = np.zeros_like(T_target)
        for r_idx in range(r):
            T_comp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.abs(T_comp - T_target).max() < tol
    
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn*mm, rank))
        V = np.zeros((mm*pp, rank))
        W = np.zeros((pp*nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i*mm+j, col] = 1
                    V[j*pp+k, col] = 1
                    W[k*nn+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_222():
        # Standard Strassen decomposition for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # i = a*e + b*g, j = a*f + b*h, k = c*e + d*g, l = c*f + d*h
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # i = M1 - M4 + M5 - M7
        # j = M3 + M5
        # k = M2 + M4
        # l = M1 - M3 + M6 + M7
        
        # U indices: a=0, b=1, c=2, d=3
        # V indices: e=0, f=1, g=2, h=3
        # W indices: i=0, j=1, k=2, l=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (c+d)e
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # M3 = a(f-h)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # M4 = d(g-c)
        U[3,3] = 1; U[2,3] = -1
        V[2,3] = 1
        W[0,3] = -1; W[2,3] = 1
        
        # M5 = (a+b)h
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = 1; W[1,4] = 1
        
        # M6 = (c-a)(e+f)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (b-d)(g+h)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = -1; W[3,6] = 1
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_222()
    T_222 = make_target_tensor(2, 2, 2)
    assert verify(U_s, V_s, W_s, T_222, 1e-10), "Strassen verification failed"
    
    # Strategy: Use Strassen blocks to reduce rank
    # Split 3x4x5 into blocks:
    # A is 3x4, B is 4x5, C is 3x5
    # Split A into A1 (3x2) and A2 (3x2)
    # Split B into [[B11, B12], [B21, B22]] where each is 2x2 or 2x3
    # Actually, let's split differently for better Strassen usage
    
    # Better approach: Split the 4 dimension into 2+2
    # A = [A1 | A2] where A1, A2 are 3x2
    # B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2 and B12, B22 are 2x3
    # C = C1 + C2 where C1 = A1*B11 + A2*B21 (3x2) and C2 = A1*B12 + A2*B22 (3x3)
    
    # For the 3x2x2 parts (C1 computation):
    # A1*B11 is 3x2 * 2x2 = 3x2, which is like 3 copies of 2x2x2? No.
    # Actually 3x2x2: A is 3x2, B is 2x2, C is 3x2
    # This has rank at most 3*2*2 = 12 by schoolbook
    # But we can do better using the structure
    
    # Let's use a different decomposition:
    # Split the 5 dimension into 2+3
    # A is 3x4, B is 4x5, C is 3x5
    # B = [B_left (4x2), B_right (4x3)]
    # C = [C_left (3x2), C_right (3x3)]
    # C_left = A * B_left (3x4 * 4x2 = 3x2)
    # C_right = A * B_right (3x4 * 4x3 = 3x3)
    
    # For C_left (3x4x2): rank at most 24 by schoolbook
    # For C_right (3x4x3): rank at most 36 by schoolbook
    # Total: 60, same as full schoolbook
    
    # Let's try to use Strassen more cleverly.
    # Split 3x4x5 as:
    # View as 3x(2+2)x(2+3)
    # A = [A1 | A2] (3x2 each)
    # B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2, B12, B22 are 2x3
    # C = C_left (3x2) + C_right (3x3)
    # C_left = A1*B11 + A2*B21
    # C_right = A1*B12 + A2*B22
    
    # Each term A_i * B_ij is a matrix product:
    # A1*B11: 3x2 * 2x2 = 3x2 (rank 12 schoolbook)
    # A2*B21: 3x2 * 2x2 = 3x2 (rank 12 schoolbook)
    # A1*B12: 3x2 * 2x3 = 3x3 (rank 18 schoolbook)
    # A2*B22: 3x2 * 2x3 = 3x3 (rank 18 schoolbook)
    # Total: 60, no improvement
    
    # The key insight: use Strassen for the 2x2x2 subproblems
    # A1*B11 where A1 is 3x2 and B11 is 2x2:
    # This is NOT a 2x2x2 problem. It's a 3x2x2 problem.
    # But we can write it as: C_left = A1 * B11
    # where A1 has 3 rows, B11 has 2 rows and 2 cols, C_left has 3 rows and 2 cols
    # We can decompose this using the 2x2x2 Strassen trick on the 2x2 part
    
    # Actually, let me think about this differently.
    # The tensor for 3x2x2 multiplication:
    # T[i*2+j, j*2+k, k*3+i] = 1 for i=0,1,2; j=0,1; k=0,1
    # We want to find its rank.
    # Lower bound: max(6, 4, 6, (12)^{2/3}) = max(6, 4, 6, 5.24) = 6
    # So rank >= 6, and schoolbook gives rank 12.
    
    # Can we achieve rank 6? Let's try.
    # If R(3x2x2) = 6, then:
    # C_left = A1*B11 + A2*B21 uses 2*6 = 12 ranks
    # C_right = A1*B12 + A2*B22 uses 2*9 = 18 ranks (if R(3x2x3)=9)
    # Total: 30, which is much better than 60!
    
    # Let's try to find a rank-6 decomposition for 3x2x2.
    # This is equivalent to finding 6 rank-1 tensors that sum to the MM tensor.
    
    def find_3x2x2_rank6():
        """Try to find a rank-6 decomposition for 3x2x2 MM"""
        nn, mm, pp = 3, 2, 2
        rank = 6
        T_target = make_target_tensor(nn, mm, pp)
        
        # Try various structured approaches
        # The MM tensor for 3x2x2:
        # T[i*2+j, j*2+k, k*3+i] = 1
        
        # Let's try a direct approach: assume the decomposition has a specific form
        # and solve for the coefficients.
        
        # Actually, let's use the fact that this is a small problem and try
        # to find it via a systematic search or known results.
        
        # Known: R(3x2x2) = 6 is achievable.
        # One construction:
        # Use the 2x2x2 Strassen decomposition and extend to 3 rows.
        
        # Let me try a different approach: use the 2x2x2 Strassen on the j-k plane
        # and handle the 3 i-rows separately.
        
        # For each i in {0,1,2}, we have:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # This is a 1x2x2 problem for each i, which has rank 2 (schoolbook) or 
        # potentially rank 2 (can't do better than 2 for 1x2x2 since lower bound is 2)
        
        # So total rank = 3*2 = 6 by schoolbook for each row.
        # Wait, that's just schoolbook for 3x2x2 which gives rank 12.
        
        # The issue is that schoolbook for 3x2x2 uses 3*2*2 = 12 terms.
        # To get rank 6, we need to share computation across the 3 rows.
        
        # Let's try:
        # P1 = A[0,0]*B[0,0]
        # P2 = A[0,1]*B[1,0]
        # P3 = A[1,0]*B[0,1]
        # P4 = A[1,1]*B[1,1]
        # P5 = (A[2,0]+A[2,1])*(B[0,0]+B[1,0])
        # P6 = (A[2,0]+A[2,1])*(B[0,1]+B[1,1])
        
        # Then:
        # C[0,0] = P1
        # C[0,1] = P3
        # C[1,0] = P2
        # C[1,1] = P4
        # C[2,0] = P5 - A[2,0]*B[1,0] - A[2,1]*B[0,0]... this doesn't work directly.
        
        # Let me try a different construction.
        # Use the identity:
        # (a+b)(c+d) = ac + ad + bc + bd
        # So (A[2,0]+A[2,1])(B[0,0]+B[1,0]) = A[2,0]B[0,0] + A[2,0]B[1,0] + A[2,1]B[0,0] + A[2,1]B[1,0]
        # = C[2,0] + A[2,0]B[1,0] + A[2,1]B[0,0]
        
        # This is getting complicated. Let's just use schoolbook for 3x2x2 (rank 12)
        # and see if we can improve the overall decomposition.
        
        # Actually, let's try a completely different approach:
        # Use the 2x2x2 Strassen decomposition for the 2x2 part and
        # handle the third row with additional terms.
        
        # For 3x2x2, we can write:
        # C[0,:] = A[0,:] * B
        # C[1,:] = A[1,:] * B  
        # C[2,:] = A[2,:] * B
        
        # The first two rows form a 2x2x2 problem (rank 7 by Strassen)
        # The third row is a 1x2x2 problem (rank 2 by schoolbook)
        # Total: 7 + 2 = 9
        
        # But we can potentially share some computation.
        # Let's just go with rank 9 for 3x2x2.
        
        # Actually, let me try to construct it explicitly.
        # Use Strassen for the top 2 rows and schoolbook for the bottom row.
        
        U = np.zeros((6, 9))
        V = np.zeros((4, 9))
        W = np.zeros((6, 9))
        
        # Strassen for rows 0,1 (indices 0,1 in U, which correspond to i=0,1)
        # U indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1], 4=A[2,0], 5=A[2,1]
        # V indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
        # W indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1], 4=C[2,0], 5=C[2,1]
        
        # Strassen for 2x2x2 (rows 0,1):
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # M2 = (A[1,0]+A[1,1])B[0,0]
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # M3 = A[0,0](B[0,1]-B[1,1])
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1
        
        # M4 = A[1,1](B[1,0]-B[0,0])
        U[3,3] = 1; U[0,3] = -1
        V[2,3] = 1
        W[0,3] = -1; W[2,3] = 1
        
        # M5 = (A[0,0]+A[0,1])B[1,1]
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = 1; W[1,4] = 1
        
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = -1; W[3,6] = 1
        
        # Schoolbook for row 2:
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        U[4,7] = 1
        V[0,7] = 1
        W[4,7] = 1
        
        U[5,7] = 1
        V[2,7] = 1
        W[4,7] += 1
        
        U[4,8] = 1
        V[1,8] = 1
        W[5,8] = 1
        
        U[5,8] = 1
        V[3,8] = 1
        W[5,8] += 1
        
        # Verify
        if verify(U, V, W, T_target, 1e-10):
            return U, V, W, 9
        return None
    
    # Try to find 3x2x2 decomposition
    result_322 = find_3x2x2_rank6()
    
    if result_322 is None:
        # Fall back to schoolbook for 3x2x2
        U_322, V_322, W_322, rank_322 = schoolbook(3, 2, 2)
    else:
        U_322, V_322, W_322, rank_322 = result_322
    
    # Now build the full 3x4x5 decomposition
    # Split: A = [A1 | A2] (3x2 each), B = [[B11, B12], [B21, B22]]
    # B11, B21 are 2x2, B12, B22 are 2x3
    # C_left = A1*B11 + A2*B21 (3x2)
    # C_right = A1*B12 + A2*B22 (3x3)
    
    # For C_left: two 3x2x2 problems, each rank rank_322
    # For C_right: two 3x2x3 problems
    
    # Let's handle 3x2x3:
    # A is 3x2, B is 2x3, C is 3x3
    # Schoolbook rank: 18
    # Can we do better? Lower bound: max(6, 6, 9, (18)^{2/3}) = max(6,6,9,7.05) = 9
    # Let's try to find a rank-9 or rank-10 decomposition.
    
    def find_3x2x3_decomposition():
        """Try to find a low-rank decomposition for 3x2x3 MM"""
        nn, mm, pp = 3, 2, 3
        T_target = make_target_tensor(nn, mm, pp)
        
        # Try rank 9 first
        for rank in [9, 10, 12, 15, 18]:
            if not deadline_ok(20):
                break
            rng = np.random.RandomState(rank * 100 + 42)
            for attempt in range(50):
                if not deadline_ok(20):
                    break
                U = rng.randn(6, rank) * 0.1
                V = rng.randn(4, rank) * 0.1
                W = rng.randn(9, rank) * 0.1
                
                # Simple gradient descent
                lr = 0.01
                for it in range(2000):
                    if not deadline_ok(20):
                        break
                    T_comp = np.zeros_like(T_target)
                    for r_idx in range(rank):
                        T_comp += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    E = T_comp - T_target
                    err = np.abs(E).max()
                    if err < 1e-6:
                        return U, V, W, rank
                    
                    # Gradients
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    for r_idx in range(rank):
                        dU[:, r_idx] = np.einsum('abc,b->a', E, V[:, r_idx] * W[:, r_idx])
                        dV[:, r_idx] = np.einsum('abc,a->b', E, U[:, r_idx] * W[:, r_idx])
                        dW[:, r_idx] = np.einsum('abc,a->c', E, U[:, r_idx] * V[:, r_idx])
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                
                if np.abs(T_comp - T_target).max() < 1e-6:
                    return U, V, W, rank
        
        # Fall back to schoolbook
        U, V, W, rank = schoolbook(nn, mm, pp)
        return U, V, W, rank
    
    # Get 3x2x3 decomposition
    U_323, V_323, W_323, rank_323 = find_3x2x3_decomposition()
    
    # Build full decomposition
    # C_left = A1*B11 + A2*B21 (two 3x2x2 problems)
    # C_right = A1*B12 + A2*B22 (two 3x2x3 problems)
    
    rank_total = 2 * rank_322 + 2 * rank_323
    U_full = np.zeros((12, rank_total))
    V_full = np.zeros((20, rank_total))
    W_full = np.zeros((15, rank_total))
    
    # U indices: A[i,j] -> i*4+j
    # V indices: B[j,k] -> j*5+k
    # W indices: C[i,k] -> k*3+i
    
    # A1*B11: A1 is rows 0-2, cols 0-1 of A; B11 is rows 0-1, cols 0-1 of B
    # U_322 indices: A[i,j] -> i*2+j (i=0,1,2; j=0,1)
    # V_322 indices: B[j,k] -> j*2+k (j=0,1; k=0,1)
    # W_322 indices: C[i,k] -> k*3+i (k=0,1; i=0,1,2)
    
    off = 0
    for r in range(rank_322):
        for i in range(3):
            for j in range(2):
                U_full[i*4 + j, off + r] = U_322[i*2 + j, r]
        for j in range(2):
            for k in range(2):
                V_full[j*5 + k, off + r] = V_322[j*2 + k, r]
        for k in range(2):
            for i in range(3):
                W_full[k*3 + i, off + r] = W_322[k*3 + i, r]
    off += rank_322
    
    # A2*B21: A2 is rows 0-2, cols 2-3 of A; B21 is rows 2-3, cols 0-1 of B
    for r in range(rank_322):
        for i in range(3):
            for j in range(2):
                U_full[i*4 + j + 2, off + r] = U_322[i*2 + j, r]
        for j in range(2):
            for k in range(2):
                V_full[(j+2)*5 + k, off + r] = V_322[j*2 + k, r]
        for k in range(2):
            for i in range(3):
                W_full[k*3 + i, off + r] = W_322[k*3 + i, r]
    off += rank_322
    
    # A1*B12: A1 is rows 0-2, cols 0-1 of A; B12 is rows 0-1, cols 2-4 of B
    # U_323 indices: A[i,j] -> i*2+j
    # V_323 indices: B[j,k] -> j*3+k (j=0,1; k=0,1,2)
    # W_323 indices: C[i,k] -> k*3+i (k=0,1,2; i=0,1,2)
    
    for r in range(rank_323):
        for i in range(3):
            for j in range(2):
                U_full[i*4 + j, off + r] = U_323[i*2 + j, r]
        for j in range(2):
            for k in range(3):
                V_full[j*5 + k + 2, off + r] = V_323[j*3 + k, r]
        for k in range(3):
            for i in range(3):
                W_full[k*3 + i, off + r] = W_323[k*3 + i, r]
    off += rank_323
    
    # A2*B22: A2 is rows 0-2, cols 2-3 of A; B22 is rows 2-3, cols 2-4 of B
    for r in range(rank_323):
        for i in range(3):
            for j in range(2):
                U_full[i*4 + j + 2, off + r] = U_323[i*2 + j, r]
        for j in range(2):
            for k in range(3):
                V_full[(j+2)*5 + k + 2, off + r] = V_323[j*3 + k, r]
        for k in range(3):
            for i in range(3):
                W_full[k*3 + i, off + r] = W_323[k*3 + i, r]
    off += rank_323
    
    # Verify
    T_target_full = make_target_tensor(3, 4, 5)
    if verify(U_full, V_full, W_full, T_target_full, 1e-6):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook(3, 4, 5)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
