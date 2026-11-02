"""Structural change: rank-reduction search on Strassen decomposition for 2x4x4."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U (4x7), V (4x7), W (4x7) for the rank-7 2x2x2 Strassen product."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    def setu(i, j, c, val):
        U[i*2 + j][c] = val
    def setv(j, k, c, val):
        V[j*2 + k][c] = val
    def setw(k, i, c, val):
        W[k*2 + i][c] = val

    setu(0,0,0,1); setu(1,1,0,1)
    setv(0,0,0,1); setv(1,1,0,1)
    setw(0,0,0,1); setw(1,1,0,1)
    setu(1,0,1,1); setu(1,1,1,1)
    setv(0,0,1,1)
    setw(0,1,1,1); setw(1,1,1,1)
    setu(0,0,2,1)
    setv(1,0,2,1); setv(1,1,2,-1)
    setw(0,0,2,1); setw(1,0,2,-1)
    setu(1,1,3,1)
    setv(0,1,3,1); setv(0,0,3,-1)
    setw(1,0,3,1); setw(1,1,3,-1)
    setu(0,0,4,1); setu(0,1,4,1)
    setv(1,1,4,1)
    setw(0,0,4,1); setw(0,1,4,1)
    setu(1,0,5,1); setu(0,0,5,-1)
    setv(0,0,5,1); setv(0,1,5,1)
    setw(0,0,5,1); setw(1,0,5,1)
    setu(0,1,6,1); setu(1,1,6,-1)
    setv(1,0,6,1); setv(1,1,6,1)
    setw(0,1,6,1); setw(1,1,6,1)

    return U, V, W


def _build_block_decomposition():
    """4 independent 2x2x2 Strassen blocks -> rank 28 for 2x4x4."""
    n, m, p = 2, 4, 4
    rank = 4 * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    su, sv, sw = _strassen_2x2x2()

    for a in range(2):
        for b in range(2):
            base = (a*2 + b) * 7
            for c in range(7):
                col = base + c
                for i in range(2):
                    for jb in range(2):
                        uval = su[i*2 + jb][c]
                        if uval:
                            U[i*m + (a*2 + jb)][col] += uval
                for jb in range(2):
                    for kb in range(2):
                        vval = sv[jb*2 + kb][c]
                        if vval:
                            V[(a*2 + jb)*p + (b*2 + kb)][col] += vval
                for i in range(2):
                    for kb in range(2):
                        wval = sw[kb*2 + i][c]
                        if wval:
                            W[(b*2 + kb)*n + i][col] += wval

    return {"U": U, "V": V, "W": W}


def _tensor_components(U, V, W, n, m, p):
    """Return list of rank-1 tensor components as vectors in R^(n*m*p)."""
    rank = len(U[0])
    comps = []
    for c in range(rank):
        comp = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    comp.append(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c])
        comps.append(comp)
    return comps


def _try_rank_reduction(result, n, m, p, deadline, max_iter=5000):
    """Search for linear dependencies among rank-1 tensors to reduce rank.
    
    Strategy: randomly project the tensor components into lower dimensions
    and check for rank deficiency. If we find a null space vector, we can
    eliminate one rank term.
    """
    U = result["U"]; V = result["V"]; W = result["W"]
    rank = len(U[0])
    best_result = result
    best_rank = rank
    
    # Get tensor components
    comps = _tensor_components(U, V, W, n, m, p)
    # Each comp is a vector of length n*m*p = 32
    # We have `rank` such vectors; check if they're linearly dependent
    
    # Build matrix where each column is a rank-1 tensor component
    # Shape: (n*m*p, rank) = (32, 28)
    comp_matrix = np.array(comps).T  # (32, rank)
    
    # Check rank of this matrix
    # If rank < number of columns, we have dependencies
    for attempt in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        # Try to find null space vectors
        # Use SVD to find small singular values
        try:
            _, s, vh = np.linalg.svd(comp_matrix, full_matrices=False)
        except Exception:
            break
        
        # Find singular values close to zero
        tol = 1e-10
        null_indices = [i for i in range(len(s)) if s[i] < tol]
        
        if len(null_indices) > 0:
            # We have null space vectors; try to use them to reduce rank
            # For each null vector, we can eliminate one component
            for null_idx in null_indices:
                # vh[null_idx] gives the null space vector
                # This means sum_c vh[null_idx][c] * comp_c = 0
                # We can drop the component with largest |vh[null_idx][c]|
                null_vec = vh[null_idx]
                # Find which component to drop
                abs_vals = [abs(null_vec[c]) for c in range(rank)]
                drop_c = max(range(rank), key=lambda c: abs_vals[c])
                
                # Remove component drop_c
                new_U = [U[i][:drop_c] + U[i][drop_c+1:] for i in range(len(U))]
                new_V = [V[i][:drop_c] + V[i][drop_c+1:] for i in range(len(V))]
                new_W = [W[i][:drop_c] + W[i][drop_c+1:] for i in range(len(W))]
                
                new_result = {"U": new_U, "V": new_V, "W": new_W}
                if _verify(new_result):
                    best_result = new_result
                    best_rank = rank - 1
                    # Update comp_matrix
                    comp_matrix = np.delete(comp_matrix, drop_c, axis=1)
                    rank = best_rank
                    break
        else:
            # No exact null space; try perturbation-based search
            # Add random small perturbations and re-check
            pass
    
    return best_result


def _verify(result):
    """Verify the decomposition against the multiplication tensor."""
    U = result["U"]; V = result["V"]; W = result["W"]
    n, m, p = 2, 4, 4
    rank = len(U[0])
    for _ in range(3):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j]*B[j][k]
                C[i][k] = s
        R = [[0]*p for _ in range(n)]
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += U[i*m + j][c] * V[j*p + k][c]
                    R[i][k] += s * W[k*n + i][c]
        for i in range(n):
            for k in range(p):
                if R[i][k] != C[i][k]:
                    return False
    return True


def _construct_alternative_blocks():
    """Try alternative block decompositions that may have lower rank.
    
    Idea: Instead of 4 independent 2x2x2 Strassen blocks, use a mixed
    decomposition where some blocks share computation.
    """
    n, m, p = 2, 4, 4
    
    # Alternative: Use 2x4x4 as a single unit with custom rank-? decomposition
    # Try: split into (2x2x4) and (2x2x4) with shared Strassen structure
    
    # Approach: C[i,k] = sum_j A[i,j] B[j,k]
    # Split j into j0 (0,1) and j1 (2,3)
    # C[i,k] = sum_{j0} A[i,j0] B[j0,k] + sum_{j1} A[i,j0] B[j0,k]
    
    # For the first sum (j in {0,1}): this is 2x2x4
    # For the second sum (j in {2,3}): this is 2x2x4
    
    # 2x2x4 can be done with rank 2*4 = 8 schoolbook, or better?
    # Actually 2x2x4: A is 2x2, B is 2x4, C is 2x4
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is just 2 dot products per row, so rank 2*4 = 8 schoolbook
    # But we can do better: for each k, C[:,k] = A * B[:,k]
    # This is 4 separate 2x2x1 products, each rank 2, total rank 8
    # Or: C = A * B where A is 2x2, B is 2x4
    # Rank of 2x2x4 tensor: at most 2*4 = 8, at least max(2*2, 2*4, 4*2)/... 
    # Actually for matrix multiplication m x n x p, rank is between max(mn,np,pm) and mnp
    # For 2x2x4: between max(4,8,4)=8 and 16. So rank 8 is optimal!
    
    # So 2x4x4 = two 2x2x4 products, each rank 8, total rank 16?
    # But that's not right because the two 2x2x4 products share the same A rows.
    
    # Let me think again. C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # = (A[i,0]B[0,k] + A[i,1]B[1,k]) + (A[i,2]B[2,k] + A[i,3]B[3,k])
    
    # First term: for fixed i, sum over j in {0,1}: A[i,j] B[j,k]
    # This is like: for each k, [A[i,0], A[i,1]] dot [B[0,k], B[1,k]]
    
    # Can we do the first term with rank < 4 per k? No, because it's a 2x1x4 tensor
    # Actually the first term is a 2x2x4 tensor: T1[i,j,k] = A[i,j] B[j,k] for j in {0,1}
    # This has rank at most 2*4 = 8 (schoolbook), and we showed rank 8 is optimal.
    
    # So total rank for 2x4x4 is at most 8 + 8 = 16? But that contradicts Strassen giving 28.
    # Wait, I think I made an error. Let me reconsider.
    
    # The tensor for 2x4x4 matrix multiplication has 2*4*4 = 32 entries.
    # The rank is the minimum number of rank-1 tensors needed.
    # Schoolbook gives rank 32.
    # Strassen-based gives rank 28.
    # My analysis above is wrong because I'm confusing the structure.
    
    # Let me just return the Strassen-based result as the best known.
    return _build_block_decomposition()


def construct():
    start = time.time()
    deadline = start + 120.0
    
    # Start with rank-28 Strassen decomposition
    result = _build_block_decomposition()
    if not _verify(result):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try rank reduction
    n, m, p = 2, 4, 4
    result = _try_rank_reduction(result, n, m, p, deadline)
    
    return result
# EVOLVE-BLOCK-END
