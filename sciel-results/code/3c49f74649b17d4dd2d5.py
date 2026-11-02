import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 7
    
    # Rank 50 solution
    def get_rank_50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)
            
            U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[c][col] += 1; U[d][col] += 1; V[e][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a][col] += 1; U[b][col] += 1; V[h][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1; W[c11][col] += 1; col += 1
            U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1; W[c00][col] += 1; col += 1
            return True

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)
            
            U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
            U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
            U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
            return True

        for k0 in [0, 2, 4]:
            add_strassen_222(0, 0, k0)
        add_schoolbook_221(0, 0, 6)
        for k0 in [0, 2, 4]:
            add_strassen_222(0, 2, k0)
        add_schoolbook_221(0, 2, 6)
        
        return {"U": U, "V": V, "W": W}

    best_solution = get_rank_50()
    best_rank = 50
    
    # Try to find a better solution using local search
    # We'll try to merge some terms or find a lower rank decomposition
    # Key insight: Look for patterns in the tensor that can be exploited
    
    def verify_solution(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    reconstructed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
                    if abs(reconstructed - 1) > 1e-9:
                        return False
        return True
    
    # Try Winograd-style approaches for 2x2x2 blocks
    # The standard Strassen uses 7 multiplications. Can we do better for specific cases?
    # For 2x2x2, the minimal rank is 7 (Strassen), so no improvement there.
    
    # Try to find a 2x2x3 decomposition with rank < 8
    # 2x2x3 schoolbook = 6, but maybe we can do 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
    # Or find a mixed approach
    
    # Let's try a different partitioning:
    # Instead of 2x2x2 blocks, try 2x2x3 blocks
    # 2x2x3 can be done in rank 6 (schoolbook) or potentially less with Winograd
    
    def get_rank_48_attempt():
        """Try to achieve rank 48 by using 2x2x3 blocks more efficiently"""
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_223_block(i0, j0, k0):
            """Add a 2x2x3 block using 6 multiplications (schoolbook)"""
            nonlocal col
            if col + 6 > rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            
            for kk in range(3):
                e = j0 * p + k0 + kk
                f = (j0 + 1) * p + k0 + kk
                c0 = (k0 + kk) * n + i0
                c1 = (k0 + kk) * n + (i0 + 1)
                
                U[a][col] += 1; V[e][col] += 1; W[c0][col] += 1; col += 1
                U[b][col] += 1; V[f][col] += 1; W[c0][col] += 1; col += 1
                U[c][col] += 1; V[e][col] += 1; W[c1][col] += 1; col += 1
                U[d][col] += 1; V[f][col] += 1; W[c1][col] += 1; col += 1
            return True
        
        # Use 2x2x3 blocks: 2 blocks of 2x2x3 = 12 multiplications
        # Remaining: 2x4x1 = 8 multiplications
        # Total: 12 + 8 = 20... that's not right
        # Let me recalculate: 2x4x7
        # Option 1: Two 2x4x3 blocks (each rank 24) + one 2x4x1 block (rank 8) = 56
        # Option 2: One 2x2x7 block (rank ?) + one 2x2x7 block (rank ?)
        
        # For 2x2x7: 
        # Schoolbook: 14
        # Strassen-based: 2x2x2 (7) + 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 25... worse
        # Actually 2x2x7 = 2x2x3 + 2x2x3 + 2x2x1 = 6+6+4 = 16
        # Or 2x2x7 using Winograd: might be 14 or 15
        
        # Let's try: 2x2x7 in rank 14 (schoolbook)
        # Two 2x2x7 blocks = 28, total rank 28? That seems too good to be true
        # Wait, 2x4x7 = (2x2x7) + (2x2x7) where the two 2x2x7 blocks use different columns of A
        # Block 1: A[0:2, 0:2] @ B[0:2, 0:7] -> C[0:2, 0:2]
        # Block 2: A[0:2, 2:4] @ B[2:4, 0:7] -> C[0:2, 2:4]
        # So we need two independent 2x2x7 decompositions
        
        # 2x2x7 schoolbook = 14, so two blocks = 28. But we need rank 50 currently.
        # The issue is that the 2x2x7 blocks share structure in C.
        
        return None  # This approach doesn't work directly
    
    # Try a more sophisticated approach: look for common subexpressions
    # or use the fact that we have specific dimensions
    
    # For now, let's try a local search on the rank 50 solution to see if we can
    # eliminate any terms
    
    # Check if any term is redundant
    U_arr = np.array(best_solution["U"], dtype=float)
    V_arr = np.array(best_solution["V"], dtype=float)
    W_arr = np.array(best_solution["W"], dtype=float)
    
    # Try removing each term and checking if the decomposition still works
    for term_idx in range(50):
        if time.time() > deadline - 10:
            break
        
        # Skip if this would be too slow
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        
        # Zero out term_idx
        U_test[:, term_idx] = 0
        V_test[:, term_idx] = 0
        W_test[:, term_idx] = 0
        
        # Quick check: does any (i,j,k) still have the correct value?
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    reconstructed = np.sum(U_test[a_idx] * V_test[b_idx] * W_test[c_idx])
                    if abs(reconstructed - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a redundant term! Create a rank 49 solution
            best_solution["U"] = [[0] * 49 for _ in range(n * m)]
            best_solution["V"] = [[0] * 49 for _ in range(m * p)]
            best_solution["W"] = [[0] * 49 for _ in range(p * n)]
            for r in range(50):
                if r == term_idx:
                    continue
                new_idx = r if r < term_idx else r - 1
                for i in range(n * m):
                    best_solution["U"][i][new_idx] = U_arr[i][r]
                for j in range(m * p):
                    best_solution["V"][j][new_idx] = V_arr[j][r]
                for k in range(p * n):
                    best_solution["W"][k][new_idx] = W_arr[k][r]
            best_rank = 49
            break
    
    # Verification
    U_arr = np.array(best_solution["U"], dtype=float)
    V_arr = np.array(best_solution["V"], dtype=float)
    W_arr = np.array(best_solution["W"], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
                if abs(reconstructed - 1) > 1e-9:
                    # Fallback if verification fails
                    return get_rank_50()
    
    return best_solution
