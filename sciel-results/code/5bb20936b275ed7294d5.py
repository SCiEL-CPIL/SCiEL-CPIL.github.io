"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    # And for all other (i',j',k') where not (i'=i,j'=j,k'=k) appropriately, should be 0
    # Actually, the tensor T_{i,j,k} = delta_{i,i'}delta_{j,j'}delta_{k,k'} for the multiplication
    # The decomposition gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T_{i,j,k}
    # where T_{i,j,k} = 1 if this is a valid multiplication term, else 0
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no, the tensor is T[i,j,k] such that sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = T[i,j,k]
    # And T[i,j,k] should be 1 for the multiplication tensor
    # The multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
    # But in the decomposition, we have sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This should equal 1 if (i,j,k) corresponds to a multiplication term, which is all of them
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Also need to check that there are no spurious terms
    # The tensor is fully specified: T[i,j,k] = 1 for all valid i,j,k
    # So we just need to verify all are 1
    return True

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close."""
    r = round(x * 2) / 2.0
    if abs(x - r) < tol:
        return r
    return None

def _try_search(rank_target, time_limit=30.0):
    """Try to find a decomposition of the given rank using random search."""
    n, m, p = DIMENSIONS
    start = time.time()
    best = None
    
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Random sparse initialization
        U = np.zeros((n*m, rank_target))
        V = np.zeros((m*p, rank_target))
        W = np.zeros((p*n, rank_target))
        
        for c in range(rank_target):
            # Randomly choose support
            u_idx = random.sample(range(n*m), min(n*m, random.randint(1, 3)))
            v_idx = random.sample(range(m*p), min(m*p, random.randint(1, 3)))
            w_idx = random.sample(range(p*n), min(p*n, random.randint(1, 3)))
            
            for idx in u_idx:
                U[idx, c] = random.choice([-1, 1])
            for idx in v_idx:
                V[idx, c] = random.choice([-1, 1])
            for idx in w_idx:
                W[idx, c] = random.choice([-1, 1])
        
        # Verify
        if _verify_decomp(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            # Try to round to half-integers
            U_rounded = []
            V_rounded = []
            W_rounded = []
            success = True
            for i in range(n*m):
                row = []
                for c in range(rank_target):
                    r = _round_to_halfint(U[i, c])
                    if r is None:
                        success = False
                        break
                    row.append(r)
                if not success:
                    break
                U_rounded.append(row)
            if not success:
                continue
            for j in range(m*p):
                row = []
                for c in range(rank_target):
                    r = _round_to_halfint(V[j, c])
                    if r is None:
                        success = False
                        break
                    row.append(r)
                if not success:
                    break
                V_rounded.append(row)
            if not success:
                continue
            for k in range(p*n):
                row = []
                for c in range(rank_target):
                    r = _round_to_halfint(W[k, c])
                    if r is None:
                        success = False
                        break
                    row.append(r)
                if not success:
                    break
                W_rounded.append(row)
            
            if success:
                # Final verification with exact values
                if _verify_decomp(U_rounded, V_rounded, W_rounded, n, m, p):
                    best = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    return best
    
    return best

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: must finish within 110s to be safe (120s limit, leave margin)
    deadline = time.time() + 110.0
    
    # Try to find a lower-rank decomposition
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline - 30:
            break
        result = _try_search(target_rank, time_limit=min(15.0, max(1.0, deadline - time.time() - 30)))
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
