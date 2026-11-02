"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        if not U:
            return False
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition."""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1; W[3][6]=1
        return U, V, W
    
    def build_strassen_block():
        """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
        U_list = []
        V_list = []
        W_list = []
        sU, sV, sW = strassen_2x2()
        
        def add_strassen(a_map, b_map, c_map):
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
        
        def add_schoolbook_2x2x2x3(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(a_row_start+i)*4 + (a_col_start+j)] = 1
                        V_new[(b_row_start+j)*5 + (b_col_start+k)] = 1
                        W_new[(c_col_start+k)*4 + (c_row_start+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
        
        add_strassen([0,1,4,5], [0,1,5,6], [0,1,4,5])
        add_schoolbook_2x2x2x3(0, 0, 0, 2, 0, 2)
        add_strassen([8,9,12,13], [0,1,5,6], [2,3,6,7])
        add_schoolbook_2x2x2x3(2, 0, 0, 2, 2, 2)
        add_strassen([2,3,6,7], [10,11,15,16], [0,1,4,5])
        add_schoolbook_2x2x2x3(0, 2, 2, 2, 0, 2)
        add_strassen([10,11,14,15], [10,11,15,16], [2,3,6,7])
        add_schoolbook_2x2x2x3(2, 2, 2, 2, 2, 2)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Start with Strassen block decomposition (rank 64)
    best = build_strassen_block()
    best_rank = len(best["U"])
    
    if not verify(best["U"], best["V"], best["W"]):
        best = schoolbook()
        best_rank = n * m * p
    
    # Core algorithm: iterative rank reduction via null-space elimination
    def reduce_rank(U, V, W, max_iterations=200):
        """Iteratively eliminate terms that are linear combinations of others."""
        U = [list(row) for row in U]
        V = [list(row) for row in V]
        W = [list(row) for row in W]
        rank = len(U[0])
        
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline - 10:
                break
            if rank <= 1:
                break
            
            eliminated = False
            # Try to eliminate each term
            for elim_idx in range(rank):
                if time.time() - start_time > deadline - 10:
                    break
                if eliminated:
                    break
                
                # Build linear system: express term elim_idx as combination of others
                # A * coeffs = b
                # A[r, c] = sum of U[r]*V[r]*W[r] for remaining terms
                # b[r] = U[elim]*V[elim]*W[elim] for each r in [16*20*20]
                
                remaining = [i for i in range(rank) if i != elim_idx]
                n_remaining = len(remaining)
                
                # Compute b vector
                b = np.zeros(n * m * p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = (i * m + j) * (m * p) + (j * p + k)
                            b[idx] = U[i*m+j][elim_idx] * V[j*p+k][elim_idx] * W[k*n+i][elim_idx]
                
                # Check if b is zero (term contributes nothing)
                if np.allclose(b, 0, atol=1e-12):
                    # Remove this term
                    for r in range(rank):
                        U[r].pop(elim_idx)
                        V[r].pop(elim_idx)
                        W[r].pop(elim_idx)
                    rank -= 1
                    eliminated = True
                    continue
                
                # Build A matrix: A[eq, col] = product for remaining term col
                A = np.zeros((n * m * p, n_remaining))
                for ci, ri in enumerate(remaining):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = (i * m + j) * (m * p) + (j * p + k)
                                A[idx, ci] = U[i*m+j][ri] * V[j*p+k][ri] * W[k*n+i][ri]
                
                # Solve A * coeffs = b
                try:
                    coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                except Exception:
                    continue
                
                # Check if solution is exact (residual is near zero)
                residual = np.linalg.norm(A @ coeffs - b)
                if residual < 1e-9:
                    # Verify the solution coefficients are "simple" (rational-like)
                    # For now, just check if it works
                    # Remove term elim_idx and add coeffs * remaining terms
                    for r in range(rank):
                        for ci, ri in enumerate(remaining):
                            if abs(coeffs[ci]) > 1e-12:
                                U[r][ri] += coeffs[ci] * U[r][elim_idx]
                                V[r][ri] += coeffs[ci] * V[r][elim_idx]
                                W[r][ri] += coeffs[ci] * W[r][elim_idx]
                        U[r].pop(elim_idx)
                        V[r].pop(elim_idx)
                        W[r].pop(elim_idx)
                    rank -= 1
                    eliminated = True
                    # Verify after modification
                    if not verify(U, V, W, tol=1e-8):
                        # Rollback: this is complex, so we'll just stop
                        return U, V, W, rank
                    break  # Restart the elimination loop with new rank
        
        return U, V, W, rank
    
    # Apply rank reduction
    U_red, V_red, W_red, new_rank = reduce_rank(best["U"], best["V"], best["W"])
    
    if new_rank < best_rank and verify(U_red, V_red, W_red):
        best = {"U": U_red, "V": V_red, "W": W_red}
        best_rank = new_rank
    
    # Additional: try to find better block decompositions
    # Try different block partitions
    def try_block_partition(n_blocks_a, n_blocks_b, n_blocks_c):
        """Try a block decomposition with given partition."""
        if time.time() - start_time > deadline - 20:
            return None
        
        # For 4x4 * 4x5, try 2x2 blocks on A, 2x2 on B, 2x2 on C
        # This is what build_strassen_block does
        return None
    
    # Try to merge identical terms
    def merge_terms(U, V, W):
        term_map = defaultdict(lambda: [0]*16, [0]*20, [0]*20)
        for r in range(len(U[0])):
            u_tuple = tuple(U[i][r] for i in range(16))
            v_tuple = tuple(V[i][r] for i in range(20))
            w_tuple = tuple(W[i][r] for i in range(20))
            key = (u_tuple, v_tuple, w_tuple)
            if key not in term_map:
                term_map[key] = [list(U[i][r] for i in range(16)), 
                               list(V[i][r] for i in range(20)), 
                               list(W[i][r] for i in range(20))]
            else:
                # Add the term to existing
                for i in range(16):
                    term_map[key][0][i] += U[i][r]
                for i in range(20):
                    term_map[key][1][i] += V[i][r]
                for i in range(20):
                    term_map[key][2][i] += W[i][r]
        
        U_merged = []
        V_merged = []
        W_merged = []
        for u_list, v_list, w_list in term_map.values():
            if all(x == 0 for x in u_list) and all(x == 0 for x in v_list) and all(x == 0 for x in w_list):
                continue
            U_merged.append(u_list)
            V_merged.append(v_list)
            W_merged.append(w_list)
        
        return {"U": U_merged, "V": V_merged, "W": W_merged}
    
    if best_rank > 0:
        merged = merge_terms(best["U"], best["V"], best["W"])
        if merged["U"] and verify(merged["U"], merged["V"], merged["W"]):
            if len(merged["U"]) < best_rank:
                best = merged
                best_rank = len(merged["U"])
    
    # Final verification
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        best = schoolbook()
        best_rank = n * m * p
    
    return best

# EVOLVE-BLOCK-END
