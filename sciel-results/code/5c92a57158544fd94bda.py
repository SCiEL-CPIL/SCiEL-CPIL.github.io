"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best = (rank_sb, U_sb, V_sb, W_sb)
    
    # Try structured decompositions using Winograd-like schemes
    # For (3,5,7), try to use block decomposition
    
    # Approach: Use the identity (a+b)(c+d) = ac + ad + bc + bd
    # to reduce rank by combining terms
    
    # Try various structured approaches
    candidates = []
    
    # Approach 1: Partition middle dimension 5 = 2+3
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # Rank = rank(3,2,7) + rank(3,3,7)
    
    # Approach 2: Use Winograd for sub-blocks
    # For 2x2 blocks, use rank-7 Winograd
    
    # Let's try a direct search for small rank reductions
    # Start from schoolbook and try to merge terms
    
    # Actually, let's use a smarter approach: 
    # The tensor T_{(i,j),(j,k),(k,i)} = 1
    # We can try to find a low-rank approximation that is exact
    
    # Use integer programming via exhaustive search for small ranks
    # This is only feasible for small ranks, so let's be strategic
    
    # Known: rank(2,2,2) = 7, rank(2,3,2) = 11, etc.
    # For (3,5,7), lower bound is max(15,35,21) = 35, but likely much higher
    
    # Try to find a decomposition with rank < 105 using iterative methods
    for target_rank in range(80, rank_sb + 1):
        if time.time() > deadline:
            break
        result = _search_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            best = (target_rank, U, V, W)
            break
    
    rank, U, V, W = best
    return {"U": U, "V": V, "W": W}


def _search_decomposition(n, m, p, rank, deadline):
    """Search for a rank-rank decomposition."""
    # Build constraint matrix
    # Each (i,j,k) gives a constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a trilinear system, hard to solve directly
    # Use alternating minimization
    
    np.random.seed(42)
    
    for trial in range(10):
        if time.time() > deadline:
            return None
        
        # Initialize with random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n * m, rank))
        V = np.random.randint(-1, 2, size=(m * p, rank))
        W = np.random.randint(-1, 2, size=(p * n, rank))
        
        # Alternating minimization
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Check validity
            if _is_valid(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
            
            # Update U: fix V, W, solve for U
            # For each row a of U, the constraints involving U[a,:] are:
            # sum_r U[a,r] * V[b,r] * W[c,r] = 1 for all (b,c) where (a,b,c) is valid
            # Let M_a[b,c] = V[b,r] * W[c,r] summed over r... no, it's per r
            # Actually: for fixed V,W, the equation is linear in U[a,:]
            # sum_r U[a,r] * (V[b,r] * W[c,r]) = 1
            # So we have a linear system: U[a,:] * (V[b,:]*W[c,:]).T = 1
            # But we have multiple (b,c) for each a
            
            # For each a, collect all (b,c) pairs
            for a in range(n * m):
                # Find all (b,c) such that there exists (i,j,k) with i*m+j=a, j*p+k=b, k*n+i=c
                i = a // m
                j = a % m
                constraints = []
                rhs = []
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    constraints.append(V[b] * W[c])
                    rhs.append(1)
                if constraints:
                    A_mat = np.array(constraints)  # (p, rank)
                    b_vec = np.array(rhs)  # (p,)
                    # Solve A_mat.T @ U[a,:] = b_vec
                    # A_mat is (p, rank), so A_mat.T is (rank, p)
                    # We want U[a,:] (rank,) such that A_mat.T @ U[a,:] = b_vec (p,)
                    # This is underdetermined if rank > p, overdetermined if rank < p
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat.T, b_vec, rcond=None)
                        U[a] = np.round(sol)
                    except:
                        pass
            
            # Similarly update V and W
            for b in range(m * p):
                j = b // p
                k = b % p
                constraints = []
                rhs = []
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    constraints.append(U[a] * W[c])
                    rhs.append(1)
                if constraints:
                    A_mat = np.array(constraints)
                    b_vec = np.array(rhs)
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat.T, b_vec, rcond=None)
                        V[b] = np.round(sol)
                    except:
                        pass
            
            for c in range(p * n):
                k = c // n
                i = c % n
                constraints = []
                rhs = []
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    constraints.append(U[a] * V[b])
                    rhs.append(1)
                if constraints:
                    A_mat = np.array(constraints)
                    b_vec = np.array(rhs)
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A_mat.T, b_vec, rcond=None)
                        W[c] = np.round(sol)
                    except:
                        pass
        
        if _is_valid(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _is_valid(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if np.sum(U[a] * V[b] * W[c]) != 1:
                    return False
    return True
# EVOLVE-BLOCK-END
