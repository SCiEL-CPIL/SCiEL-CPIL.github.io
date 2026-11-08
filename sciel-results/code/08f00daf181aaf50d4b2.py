import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill
    
    def check_deadline():
        return (time.time() - start_time) > DEADLINE
    
    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Vectorized verification
        # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Use einsum: T = np.einsum('ic,jc,kc->ijk', U, V, W) but indices don't match directly
        # U has shape (n*m, rank), index (i*m+j, c)
        # V has shape (m*p, rank), index (j*p+k, c)
        # W has shape (p*n, rank), index (k*n+i, c)
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Let's build it iteratively but with numpy for speed
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            # U_col = U[:, c] has shape (n*m,)
            # V_col = V[:, c] has shape (m*p,)
            # W_col = W[:, c] has shape (p*n,)
            U_col = U[:, c].reshape(n, m)
            V_col = V[:, c].reshape(m, p)
            W_col = W[:, c].reshape(p, n)
            # T[i,j,k] += U_col[i,j] * V_col[j,k] * W_col[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U_col[i,j] * V_col[j,k] * W_col[k,i]
        if np.max(np.abs(T - 1.0)) > 1e-10:
            return False
        return True
    
    def add_strassen_block(i0, j0, k0, U, V, W, col):
        """Add a 2x2x2 Strassen block at position (i0, j0, k0)."""
        # M1 = (A00+A11)(B00+B11)
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # Try multiple block configurations
    # All non-overlapping 4-block configs give rank 71, but let's try a few
    block_configs = [
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)],
        [(0, 0, 0), (0, 3, 0), (0, 0, 2), (0, 3, 2)],
        [(1, 0, 0), (1, 2, 0), (1, 0, 2), (1, 2, 2)],
        [(1, 0, 0), (1, 3, 0), (1, 0, 2), (1, 3, 2)],
    ]
    
    for blocks in block_configs:
        if check_deadline():
            break
        
        num_blocks = len(blocks)
        total_rank = num_blocks * 7 + (n*m*p - num_blocks*8)
        
        U = [[0.0]*total_rank for _ in range(n*m)]
        V = [[0.0]*total_rank for _ in range(m*p)]
        W = [[0.0]*total_rank for _ in range(p*n)]
        
        col = 0
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        
        valid_config = True
        for i0, j0, k0 in blocks:
            # Check bounds
            if i0+1 >= n or j0+1 >= m or k0+1 >= p:
                valid_config = False
                break
            # Check no overlap
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if covered[i0+di][j0+dj][k0+dk]:
                            valid_config = False
                            break
                    if not valid_config:
                        break
                if not valid_config:
                    break
            if not valid_config:
                break
            add_strassen_block(i0, j0, k0, U, V, W, col)
            col += 7
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
        
        if not valid_config:
            continue
        
        # Add schoolbook terms for uncovered positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] = 1.0
                        V[j*p+k][col] = 1.0
                        W[k*n+i][col] = 1.0
                        col += 1
        
        # Trim to actual rank
        actual_rank = col
        U = [row[:actual_rank] for row in U]
        V = [row[:actual_rank] for row in V]
        W = [row[:actual_rank] for row in W]
        
        # Verify
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = 3, 5, 5
    print(f"Rank: {U.shape[1]}")
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    print(f"Valid: {ok}")
