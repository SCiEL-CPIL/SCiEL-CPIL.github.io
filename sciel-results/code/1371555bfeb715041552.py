import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT - 10:
        # Return a valid schoolbook solution if time is critical
        U = [[0.0] * (n*m*p) for _ in range(n*m)]
        V = [[0.0] * (n*m*p) for _ in range(m*p)]
        W = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Use multiple overlapping Strassen-like blocks and then attempt
    # to find linear dependencies to reduce rank further.
    # 
    # The 3x5x5 tensor T has entries T[i,j,k] = 1 (it's the multiplication tensor).
    # We want to find U (15 x r), V (25 x r), W (15 x r) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # Start with a base decomposition and try to reduce rank by finding
    # linear combinations of the rank-1 terms that can be merged.
    
    # First, build a decomposition using Strassen on all possible 2x2x2 subcubes
    # and schoolbook for the rest, then attempt rank reduction.
    
    # All possible 2x2x2 blocks (i0, j0, k0) where i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # But we need to be careful about overlaps.
    
    # Let's use a greedy approach: cover the 3x5x5 grid with 2x2x2 Strassen blocks
    # to minimize the total rank.
    
    # For each 2x2x2 block, Strassen uses 7 multiplications instead of 8.
    # Net saving per block: 1 multiplication.
    
    # Maximum number of non-overlapping 2x2x2 blocks in 3x5x5:
    # In dimension i (size 3): at most 1 block (covers i=0,1)
    # In dimension j (size 5): at most 2 blocks (covers j=0,1 and j=2,3)
    # In dimension k (size 5): at most 2 blocks (covers k=0,1 and k=2,3)
    # So max 1*2*2 = 4 blocks, saving 4 multiplications.
    # This gives rank 75 - 4 = 71.
    
    # But with overlapping blocks, we might do better. Let's try a different approach:
    # Use the fact that the tensor has structure and try to find a lower rank decomposition
    # by solving a system of equations.
    
    # Actually, let's try a more systematic approach. We know that:
    # - The tensor T_{ijk} for 3x5x5 multiplication is the "multiplication tensor"
    # - We can try to find a rank-r decomposition by solving:
    #   sum_{c=1}^{r} U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c} = delta_{i,j,k} (in the appropriate sense)
    
    # For matrix multiplication, the tensor is:
    # T[i, j, k] = 1 if this term A[i,j] * B[j,k] contributes to C[i,k]
    # Actually, the multiplication tensor M_{ijk} satisfies:
    # C[i,k] = sum_j A[i,j] * M[i,j,k] * B[j,k]
    # where M[i,j,k] = 1 for all i,j,k (it's just the identity in the middle index)
    
    # Wait, I need to be more careful. The tensor for matrix multiplication is:
    # T[i, j, k] = 1 for all valid i, j, k
    # because C[i,k] = sum_j A[i,j] B[j,k]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of 75 equations (3*5*5) in the unknowns U, V, W.
    
    # Let's try rank 70 first (saving 5 from schoolbook).
    
    # Use a base decomposition and then try to eliminate one term at a time.
    
    # Start with 4 Strassen blocks (rank 71) as in the current solution.
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    base_rank = total_strassen_rank + schoolbook_terms_count  # 71
    
    U = [[0.0] * base_rank for _ in range(n * m)]
    V = [[0.0] * base_rank for _ in range(m * p)]
    W = [[0.0] * base_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Now try to reduce rank by finding linear dependencies.
    # Convert to numpy arrays for easier manipulation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Verify the base decomposition
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U_arr, V_arr, W_arr):
        # Base decomposition is invalid, return schoolbook
        U = [[0.0] * (n*m*p) for _ in range(n*m)]
        V = [[0.0] * (n*m*p) for _ in range(m*p)]
        W = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by one: can we express one column as a linear combination
    # of the others, maintaining the decomposition?
    # This is complex, so let's try a simpler approach:
    # Look for pairs of columns that can be merged.
    
    # Actually, let's try a different strategy: use a larger Strassen-like block.
    # For a 2x3x3 block, can we do better than 2*3*3 = 18?
    # Strassen for 2x2x2 gives 7. For 2x3x3, we can split into 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1
    # That's 7 + 4 + 4 + 2 = 17. Saving 1.
    
    # Or use the fact that the 5x5 multiplication can be done in rank 19 (known result).
    # For 3x5x5, if we treat it as 3 separate 1x5x5 multiplications, each can be done
    # in rank 19, giving total rank 3*19 = 57. But that's not right because the 
    # middle dimension is shared.
    
    # Actually, the correct approach is:
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0,1,2 and k=0,1,2,3,4
    # This is a bilinear map from A (3x5) and B (5x5) to C (3x5).
    # The tensor rank of this map is what we're looking for.
    
    # Known results: The rank of 5x5 matrix multiplication is at most 19 (from Strassen-like algorithms).
    # For 3x5x5, we can potentially do better than 75.
    
    # Let's try to use a 5x5x5 Strassen-like decomposition on the B side.
    # If we can decompose the 5x5 multiplication in rank 19, then for each row i of A,
    # we compute C[i,:] = A[i,:] * B using rank 19.
    # But the A[i,:] is different for each i, so we can't directly reuse.
    
    # However, we can write:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we have a rank-19 decomposition of the 5x5 multiplication:
    # B[j,k] = sum_{c=1}^{19} V[j,c] * W[k,c] (this is not quite right)
    
    # Actually, the 5x5 multiplication tensor T5[j,k,l] satisfies:
    # (B*B')[k,l] = sum_j B[j,k] * T5[j,k,l] * B'[j,l]
    # Wait, I'm confusing myself.
    
    # Let me restart with a cleaner approach.
    
    # The 3x5x5 multiplication tensor is T[i,j,k] where:
    # C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
    # and T[i,j,k] = 1 for all valid i,j,k.
    
    # We want to find U (15 x r), V (25 x r), W (15 x r) such that:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the tensor T.
    
    # Let's try to use the known rank-19 decomposition of 5x5 matrix multiplication.
    # The 5x5 multiplication can be done in rank 19 using a Strassen-like algorithm.
    
    # For the 3x5x5 case, we can potentially share some of the rank-19 terms across rows.
    
    # Here's an idea: Use the rank-19 decomposition of 5x5 for the B matrix,
    # and for each row i of A, compute the dot product with the appropriate coefficients.
    
    # Let the rank-19 decomposition of 5x5 multiplication be:
    # (B * B')[k, l] = sum_{c=1}^{19} P_c(B) * Q_c(B')
    # where P_c and Q_c are bilinear forms.
    
    # For our case, B' is not the same as B, so this doesn't directly apply.
    
    # Let me try a different approach: just try to find a rank-70 decomposition
    # by solving a linear system.
    
    # If we fix U and V, then W is determined by:
    # W[k*n+i, c] = T[i,j,k] / (U[i*m+j,c] * V[j*p+k,c])
    # But this only works if U[i*m+j,c] * V[j*p+k,c] != 0.
    
    # This is getting too complex. Let's just return the rank-71 solution.
    # It's already an improvement over schoolbook (75).
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify correctness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                if abs(val - 1.0) > 1e-10:
                    print(f"ERROR: T[{i},{j},{k}] = {val}, expected 1.0")
                    break
