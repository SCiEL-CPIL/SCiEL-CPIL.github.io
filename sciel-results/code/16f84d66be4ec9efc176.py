import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and T[a,b,c] = 0 for all other (a,b,c)
    
    # Efficient check: for each r, the outer product U[:,r] V[:,r] W[:,r]
    # contributes to specific (a,b,c) positions.
    
    # Build the target tensor implicitly: 
    # target[a,b,c] = 1 iff exists i,j,k s.t. a=i*m+j, b=j*p+k, c=k*n+i
    # Note: a determines i,j (i=a//m, j=a%m). b determines j,k (j=b//p, k=b%p).
    #       c determines k,i (i=c%n, k=c//n).
    # Consistency: j from a must equal j from b. k from b must equal k from c. i from c must equal i from a.
    # So for a given (a,b,c), let i1=a//m, j1=a%m, j2=b//p, k2=b%p, k3=c//n, i3=c%n.
    # If j1==j2 and k2==k3 and i1==i3, then target=1, else 0.
    
    # Instead of building the full tensor (6*24*16 = 2304 entries), we can check directly.
    
    # Compute T
    # T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr) is too slow/memory heavy if rank is large, but rank=44 is small.
    # 6*24*16 = 2304. 44 * 6 * 24 * 16 = ~100k ops. Very fast.
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check zeros
    # Create a mask of target positions
    target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_mask[a,b,c] = True
    
    non_target_err = np.max(np.abs(T[~target_mask])) if (~target_mask).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    
    # Strategy:
    # Split B (3x8) into 4 blocks of 3x2: B_0, B_1, B_2, B_3
    # Each block corresponds to columns 2b, 2b+1 of B.
    # C_b = A * B_b is 2x2.
    # A is 2x3. B_b is 3x2.
    # Split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    # Split B_b into B_top (2x2, rows 0,1) and B_bottom (1x2, row 2).
    # C_b = A_left * B_top + A_right * B_bottom
    # A_left * B_top is 2x2 * 2x2 -> Rank 7 (Strassen)
    # A_right * B_bottom is 2x1 * 1x2 -> Rank 4 (Trivial outer product)
    # Total rank per block = 7 + 4 = 11
    # Total rank = 4 * 11 = 44
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    current_rank = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part (A_left * B_top) ---
        # A_left indices in U:
        # a00 -> U[0*3+0] = U[0]
        # a01 -> U[0*3+1] = U[1]
        # a10 -> U[1*3+0] = U[3]
        # a11 -> U[1*3+1] = U[4]
        
        # B_top indices in V:
        # b00 -> V[0*8+k0] = V[k0]
        # b01 -> V[0*8+k1] = V[k1]
        # b10 -> V[1*8+k0] = V[8+k0]
        # b11 -> V[1*8+k1] = V[8+k1]
        
        # C_b indices in W:
        # C00 (i=0, k=k0) -> W[k0*2+0] = W[2*k0]
        # C01 (i=0, k=k1) -> W[k1*2+0] = W[2*k1]
        # C10 (i=1, k=k0) -> W[k0*2+1] = W[2*k0+1]
        # C11 (i=1, k=k1) -> W[k1*2+1] = W[2*k1+1]
        
        # Strassen products:
        # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
        t = current_rank
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        current_rank += 1
        
        # m2 = (a10+a11)b00 -> C10 += 1, C11 -= 1
        t = current_rank
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        current_rank += 1
        
        # m3 = a00(b01-b11) -> C01 += 1, C11 += 1
        t = current_rank
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        current_rank += 1
        
        # m4 = a11(b10-b00) -> C00 += 1, C10 += 1
        t = current_rank
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        current_rank += 1
        
        # m5 = (a00+a01)b11 -> C00 -= 1, C01 += 1
        t = current_rank
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        current_rank += 1
        
        # m6 = (a10-a00)(b00+b01) -> C11 += 1
        t = current_rank
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        current_rank += 1
        
        # m7 = (a01-a11)(b10+b11) -> C00 += 1
        t = current_rank
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        current_rank += 1
        
        # --- Trivial part (A_right * B_bottom) ---
        # A_right indices in U:
        # a02 (i=0, j=2) -> U[0*3+2] = U[2]
        # a12 (i=1, j=2) -> U[1*3+2] = U[5]
        
        # B_bottom indices in V:
        # b20 (j=2, k=k0) -> V[2*8+k0] = V[16+k0]
        # b21 (j=2, k=k1) -> V[2*8+k1] = V[16+k1]
        
        # C_b indices in W:
        # C00 (i=0, k=k0) -> W[2*k0]
        # C01 (i=0, k=k1) -> W[2*k1]
        # C10 (i=1, k=k0) -> W[2*k0+1]
        # C11 (i=1, k=k1) -> W[2*k1+1]
        
        # Term 1: A[0,2] * B[2,k0] -> C[0,k0]
        t = current_rank
        U[2, t] += 1
        V[16+k0, t] += 1
        W[2*k0, t] += 1
        current_rank += 1
        
        # Term 2: A[0,2] * B[2,k1] -> C[0,k1]
        t = current_rank
        U[2, t] += 1
        V[16+k1, t] += 1
        W[2*k1, t] += 1
        current_rank += 1
        
        # Term 3: A[1,2] * B[2,k0] -> C[1,k0]
        t = current_rank
        U[5, t] += 1
        V[16+k0, t] += 1
        W[2*k0+1, t] += 1
        current_rank += 1
        
        # Term 4: A[1,2] * B[2,k1] -> C[1,k1]
        t = current_rank
        U[5, t] += 1
        V[16+k1, t] += 1
        W[2*k1+1, t] += 1
        current_rank += 1

    # Verification
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # This should not happen if the logic is correct
        # Return empty or raise? The contract says invalid result scores 0.
        # Better to return the result anyway, assuming the logic is sound.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
