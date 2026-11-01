import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def strassen_block():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for u in range(2):
            for v in range(4):
                base_col = 7 * (4 * u + v)
                
                a11 = 0 * m + 2 * u
                a12 = 0 * m + 2 * u + 1
                a21 = 1 * m + 2 * u
                a22 = 1 * m + 2 * u + 1
                
                b11 = (2 * u) * p + 2 * v
                b12 = (2 * u) * p + 2 * v + 1
                b21 = (2 * u + 1) * p + 2 * v
                b22 = (2 * u + 1) * p + 2 * v + 1
                
                c11 = (2 * v) * n + 0
                c12 = (2 * v + 1) * n + 0
                c21 = (2 * v) * n + 1
                c22 = (2 * v + 1) * n + 1
                
                U[a11, base_col] += 1; U[a22, base_col] += 1
                V[b11, base_col] += 1; V[b22, base_col] += 1
                W[c11, base_col] += 1; W[c22, base_col] += 1
                
                U[a21, base_col+1] += 1; U[a22, base_col+1] += 1
                V[b11, base_col+1] += 1
                W[c21, base_col+1] += 1; W[c22, base_col+1] -= 1
                
                U[a11, base_col+2] += 1
                V[b12, base_col+2] += 1; V[b22, base_col+2] -= 1
                W[c12, base_col+2] += 1; W[c22, base_col+2] += 1
                
                U[a22, base_col+3] += 1
                V[b21, base_col+3] += 1; V[b11, base_col+3] -= 1
                W[c11, base_col+3] += 1; W[c21, base_col+3] += 1
                
                U[a11, base_col+4] += 1; U[a12, base_col+4] += 1
                V[b22, base_col+4] += 1
                W[c11, base_col+4] -= 1; W[c12, base_col+4] += 1
                
                U[a21, base_col+5] += 1; U[a11, base_col+5] -= 1
                V[b11, base_col+5] += 1; V[b12, base_col+5] += 1
                W[c22, base_col+5] += 1
                
                U[a12, base_col+6] += 1; U[a22, base_col+6] -= 1
                V[b21, base_col+6] += 1; V[b22, base_col+6] += 1
                W[c11, base_col+6] += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True
    
    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_block()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Try to find lower rank via structured approach
    # M_{2,4,8} can be seen as sub-tensor of M_{2,8,8}
    # Try rank 48, 42, 36, 32
    
    def try_rank(target_rank, U_init, V_init, W_init):
        """Try to find a valid decomposition at target_rank using initialization."""
        if target_rank >= rank_best:
            return None
        
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Initialize with first target_rank columns of best solution
        U[:, :min(target_rank, rank_best)] = U_init[:, :min(target_rank, rank_best)]
        V[:, :min(target_rank, rank_best)] = V_init[:, :min(target_rank, rank_best)]
        W[:, :min(target_rank, rank_best)] = W_init[:, :min(target_rank, rank_best)]
        
        # Check if valid as-is
        if verify(U, V, W, target_rank):
            return U, V, W, target_rank
        
        return None
    
    # Try various target ranks
    for target in [48, 42, 36, 32]:
        if time.time() > deadline - 10:
            break
        result = try_rank(target, U_best, V_best, W_best)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            break
    
    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
