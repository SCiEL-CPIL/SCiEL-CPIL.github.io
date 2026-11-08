import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8  # A: n×m, B: m×p, C: n×p
    start_time = time.time()
    deadline = 140  # safety margin from 150s
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2x2: C[i,k] = sum_j A[i,j]B[j,k]
    # Strassen factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # r=0: M1
    strassen_U = np.zeros((4, 7))
    strassen_V = np.zeros((4, 7))
    strassen_W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11) -> U: A00=1,A11=1; V: B00=1,B11=1; W: C00+=1,C11+=1
    strassen_U[0,0] = 1; strassen_U[3,0] = 1  # A00, A11
    strassen_V[0,0] = 1; strassen_V[3,0] = 1  # B00, B11
    strassen_W[0,0] = 1; strassen_W[3,0] = 1  # C00, C11
    
    # r=1: (A10+A11)B00 -> U: A10=1,A11=1; V: B00=1; W: C10+=1
    strassen_U[2,1] = 1; strassen_U[3,1] = 1
    strassen_V[0,1] = 1
    strassen_W[2,1] = 1  # C10
    
    # r=2: A00(B10-B11) -> U: A00=1; V: B10=1,B11=-1; W: C01+=1
    strassen_U[0,2] = 1
    strassen_V[2,2] = 1; strassen_V[3,2] = -1
    strassen_W[1,2] = 1  # C01
    
    # r=3: A11(B01-B10) -> U: A11=1; V: B01=1,B10=-1; W: C10+=1
    strassen_U[3,3] = 1
    strassen_V[1,3] = 1; strassen_V[2,3] = -1
    strassen_W[2,3] = 1  # C10
    
    # r=4: (A00+A01)B11 -> U: A00=1,A01=1; V: B11=1; W: C00-=1
    strassen_U[0,4] = 1; strassen_U[1,4] = 1
    strassen_V[3,4] = 1
    strassen_W[0,4] = -1  # C00
    
    # r=5: (A10-A00)(B00+B10) -> U: A10=1,A00=-1; V: B00=1,B10=1; W: C11+=1
    strassen_U[2,5] = 1; strassen_U[0,5] = -1
    strassen_V[0,5] = 1; strassen_V[2,5] = 1
    strassen_W[3,5] = 1  # C11
    
    # r=6: (A01-A11)(B01+B11) -> U: A01=1,A11=-1; V: B01=1,B11=1; W: C00+=1
    strassen_U[1,6] = 1; strassen_U[3,6] = -1
    strassen_V[1,6] = 1; strassen_V[3,6] = 1
    strassen_W[0,6] = 1  # C00
    
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Build rank-56 solution using 8 Strassen blocks
    def build_strassen_56():
        U = np.zeros((n*m, 56))
        V = np.zeros((m*p, 56))
        W = np.zeros((p*n, 56))
        
        for b in range(8):
            # b encodes (j_group, k_group): j_group in {0,1}, k_group in {0,1,2,3}
            j_off = 2 * (b // 4)  # 0 or 2
            k_off = 2 * (b % 4)   # 0,2,4,6
            base = 7 * b
            
            for r in range(7):
                cr = base + r
                # U: indices i*2+local_j, where local_j in {0,1}, global j = local_j + j_off
                for local_idx in range(4):
                    val = strassen_U[local_idx, r]
                    if val != 0:
                        li, lj = divmod(local_idx, 2)
                        U[li*m + (lj + j_off), cr] += val
                
                # V: indices local_j*2+local_k, global j = local_j + j_off, global k = local_k + k_off
                for local_idx in range(4):
                    val = strassen_V[local_idx, r]
                    if val != 0:
                        lj, lk = divmod(local_idx, 2)
                        V[(lj + j_off)*p + (lk + k_off), cr] += val
                
                # W: indices local_k*2+local_i, global k = local_k + k_off, global i = local_i
                for local_idx in range(4):
                    val = strassen_W[local_idx, r]
                    if val != 0:
                        lk, li = divmod(local_idx, 2)
                        W[(lk + k_off)*n + li, cr] += val
        
        return U, V, W
    
    U56, V56, W56 = build_strassen_56()
    valid56 = verify(U56, V56, W56)
    
    if not valid56:
        # Fallback: rank 64 (one per i,j,k triple)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 56
    best_U, best_V, best_W = U56.copy(), V56.copy(), W56.copy()
    
    # Try to reduce rank by finding linear dependencies among rank terms
    # For a given rank-r decomposition, if the columns of the "flattened" tensor are linearly dependent,
    # we might be able to reduce rank.
    
    # Alternative: try rank 48, 40, etc. with optimization
    # Set up: minimize ||sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1||^2 subject to integer/half-integer constraint
    
    # Let's try a simpler approach: check if we can merge some Strassen terms
    # The 8 blocks are independent (different j,k ranges), so we can't merge across blocks.
    # Within a block, Strassen rank 7 is optimal for 2x2x2.
    
    # So 56 seems to be the natural upper bound from this construction.
    
    # Try: can we use a different decomposition for 2x4x8 that's not block-diagonal?
    
    # Idea: Use rank-7 Strassen on 2x2x2 for the "core" and handle the extra dimensions differently.
    
    # Let's try: split j into {0,1,2,3} and use a 4-term decomposition where each term
    # handles one j but with Strassen-like sharing in the i,k space.
    
    # For each j, we need: sum_r U[i*4+j,r] V[j*8+k,r] W[k*2+i,r] contributes to the total.
    # If we use 4 "base" terms (one per j) plus correction terms...
    
    # Actually, let me try a direct optimization approach for rank < 56.
    
    def try_optimize_rank(target_rank, max_iters=2000, lr=0.01):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # f = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1
                        f = 0.0
                        for r in range(target_rank):
                            f += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        loss += (f - 1.0)**2
                        
                        for r in range(target_rank):
                            uf = U[i*m+j, r]
                            vf = V[j*p+k, r]
                            wf = W[k*n+i, r]
                            # d/dU = 2*(f-1)*V*W
                            grad_U[i*m+j, r] += 2*(f-1.0)*vf*wf
                            grad_V[j*p+k, r] += 2*(f-1.0)*uf*wf
                            grad_W[k*n+i, r] += 2*(f-1.0)*uf*vf
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        # Check if solution is close enough
        if best_loss < 1e-10:
            # Round to nearest half-integer
            best_U = np.round(best_U * 2) / 2
            best_V = np.round(best_V * 2) / 2
            best_W = np.round(best_W * 2) / 2
            
            if verify(best_U, best_V, best_W):
                return best_U, best_V, best_W, best_loss
        
        return None, None, None, best_loss
    
    # Try ranks below 56
    for target_rank in range(48, 55):
        if time.time() - start_time > deadline - 10:
            break
        
        U_t, V_t, W_t, loss_t = try_optimize_rank(target_rank, max_iters=500, lr=0.05)
        
        if U_t is not None:
            best_rank = target_rank
            best_U, best_V, best_W = U_t, V_t, W_t
            break
    
    # Also try rank 48 with more iterations
    if best_rank == 56 and time.time() - start_time < deadline - 20:
        U_t, V_t, W_t, loss_t = try_optimize_rank(48, max_iters=2000, lr=0.02)
        if U_t is not None:
            best_rank = 48
            best_U, best_V, best_W = U_t, V_t, W_t
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Return the verified rank-56 solution
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
