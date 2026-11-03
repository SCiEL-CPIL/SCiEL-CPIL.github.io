import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 8]

def _validate_numpy(U, V, W, n, m, p):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _validate(U, V, W, n, m, p):
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    return _validate_numpy(U, V, W, n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
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

def _strassen_2x2_2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)"""
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen's 7 multiplications
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U: A factors (2x2 = 4 entries, rank 7)
    # U[i,j,r] where i,j in {0,1}, r in {0..6}
    # M1: A11+A22 -> U[0,0,0]=1, U[1,1,0]=1
    # M2: A21+A22 -> U[1,0,1]=1, U[1,1,1]=1
    # M3: A11 -> U[0,0,2]=1
    # M4: A22 -> U[1,1,3]=1
    # M5: A11+A12 -> U[0,0,4]=1, U[0,1,4]=1
    # M6: A21-A11 -> U[1,0,5]=1, U[0,0,5]=-1
    # M7: A12-A22 -> U[0,1,6]=1, U[1,1,6]=-1
    
    U[0][0] = 1; U[1][0] = 1  # M1: A11+A22
    U[1][1] = 1; U[1][0] = 1  # M2: A21+A22 (A21 is [1,0], A22 is [1,1])
    U[0][2] = 1                # M3: A11
    U[1][3] = 1                # M4: A22
    U[0][4] = 1; U[0][4] = 1   # M5: A11+A12
    U[1][5] = 1; U[0][5] = -1  # M6: A21-A11
    U[0][6] = 1; U[1][6] = -1  # M7: A12-A22
    
    # V: B factors
    # M1: B11+B22 -> V[0,0,0]=1, V[1,1,0]=1
    # M2: B11 -> V[0,0,1]=1
    # M3: B12-B22 -> V[0,1,2]=1, V[1,1,2]=-1
    # M4: B21-B11 -> V[1,0,3]=1, V[0,0,3]=-1
    # M5: B22 -> V[1,1,4]=1
    # M6: B11+B12 -> V[0,0,5]=1, V[0,1,5]=1
    # M7: B21+B22 -> V[1,0,6]=1, V[1,1,6]=1
    
    V[0][0] = 1; V[1][0] = 1  # M1: B11+B22
    V[0][1] = 1                # M2: B11
    V[0][2] = 1; V[1][2] = -1  # M3: B12-B22
    V[1][3] = 1; V[0][3] = -1  # M4: B21-B11
    V[1][4] = 1                # M5: B22
    V[0][5] = 1; V[0][5] = 1   # M6: B11+B12
    V[1][6] = 1; V[1][6] = 1   # M7: B21+B22
    
    # W: C = M1 - M5 + M7, etc.
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M4
    
    W[0][0] = 1; W[0][4] = -1; W[0][6] = 1  # C11 = M1 - M5 + M7
    W[1][0] = 1; W[1][4] = 1                # C12 = M3 + M5
    W[0][1] = 1; W[0][3] = 1                # C21 = M2 + M4
    W[1][1] = 1; W[1][1] = -1; W[1][3] = 1  # C22 = M1 - M2 + M4
    
    return U, V, W

def _build_3x3x2_decomposition():
    """
    Build a 3x3x2 decomposition using Strassen blocks.
    We can embed 2x2 Strassen blocks in the 3x3 matrix.
    """
    n, m, p = 3, 3, 2
    # Start with schoolbook for 3x3x2 (rank 18) and try to reduce
    # Use overlapping 2x2 blocks
    
    # For now, use schoolbook as baseline
    return _schoolbook(n, m, p)

def _construct_from_strassen_blocks(n, m, p, block_positions):
    """
    Construct decomposition using Strassen blocks at given positions.
    block_positions: list of (i_offset, j_offset, k_offset) for 2x2x2 blocks
    """
    rank = 7 * len(block_positions)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_s, V_s, W_s = _strassen_2x2_2()
    
    for block_idx, (i_off, j_off, k_off) in enumerate(block_positions):
        base_rank = block_idx * 7
        
        # Add U block
        for i in range(2):
            for j in range(2):
                if i_off + i < n and j_off + j < m:
                    for r in range(7):
                        U[(i_off + i) * m + (j_off + j)][base_rank + r] += U_s[i * 2 + j][r]
        
        # Add V block
        for j in range(2):
            for k in range(2):
                if j_off + j < m and k_off + k < p:
                    for r in range(7):
                        V[(j_off + j) * p + (k_off + k)][base_rank + r] += V_s[j * 2 + k][r]
        
        # Add W block
        for k in range(2):
            for i in range(2):
                if k_off + k < p and i_off + i < n:
                    for r in range(7):
                        W[(k_off + k) * n + (i_off + i)][base_rank + r] += W_s[k * 2 + i][r]
    
    return {"U": U, "V": V, "W": W}

def _try_strassen_placements(n, m, p, time_limit=30.0):
    """
    Try different placements of 2x2x2 Strassen blocks to cover the 3x3x8 problem.
    """
    start_time = time.time()
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Generate all possible 2x2x2 block positions
    possible_i = list(range(n - 1))  # 0, 1
    possible_j = list(range(m - 1))  # 0, 1
    possible_k = list(range(p - 1))  # 0, 1, ..., 6
    
    all_positions = [(i, j, k) for i in possible_i for j in possible_j for k in possible_k]
    
    # Try single blocks first
    for pos in all_positions:
        if time.time() - start_time > time_limit:
            break
        result = _construct_from_strassen_blocks(n, m, p, [pos])
        if _validate(result["U"], result["V"], result["W"], n, m, p):
            rank = len(result["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
    
    # Try pairs of non-overlapping blocks
    for i in range(len(all_positions)):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, len(all_positions)):
            pos1 = all_positions[i]
            pos2 = all_positions[j]
            
            # Check if blocks overlap
            overlap = (abs(pos1[0] - pos2[0]) < 2 and 
                      abs(pos1[1] - pos2[1]) < 2 and 
                      abs(pos1[2] - pos2[2]) < 2)
            if overlap:
                continue
            
            result = _construct_from_strassen_blocks(n, m, p, [pos1, pos2])
            if _validate(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
    
    # Try triples
    for i in range(len(all_positions)):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, len(all_positions)):
            if time.time() - start_time > time_limit:
                break
            for k in range(j + 1, len(all_positions)):
                pos1 = all_positions[i]
                pos2 = all_positions[j]
                pos3 = all_positions[k]
                
                # Check overlaps
                overlap12 = (abs(pos1[0] - pos2[0]) < 2 and 
                           abs(pos1[1] - pos2[1]) < 2 and 
                           abs(pos1[2] - pos2[2]) < 2)
                overlap13 = (abs(pos1[0] - pos3[0]) < 2 and 
                           abs(pos1[1] - pos3[1]) < 2 and 
                           abs(pos1[2] - pos3[2]) < 2)
                overlap23 = (abs(pos2[0] - pos3[0]) < 2 and 
                           abs(pos2[1] - pos3[1]) < 2 and 
                           abs(pos2[2] - pos3[2]) < 2)
                if overlap12 or overlap13 or overlap23:
                    continue
                
                result = _construct_from_strassen_blocks(n, m, p, [pos1, pos2, pos3])
                if _validate(result["U"], result["V"], result["W"], n, m, p):
                    rank = len(result["U"][0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
    
    return best_result, best_rank

def _find_332(rank, time_limit=10.0):
    n, m, p = 3, 3, 2
    start_time = time.time()
    
    # Number of equations: n*m*p = 18
    # Unknowns in W: p*n*rank = 6*rank
    num_eqs = n * m * p
    
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        # Random U, V
        # Use small integers to keep numbers manageable
        U = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)], dtype=np.float64)
        V = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)], dtype=np.float64)
        
        # Construct A matrix for W
        # A is (num_eqs, p*n*rank)
        A = np.zeros((num_eqs, p * n * rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        # Coefficient for W[k, i, r]
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        col_idx = k * n * rank + i * rank + r
                        A[eq_idx, col_idx] = coeff
        
        b = np.ones(num_eqs)
        
        # Solve least squares
        W_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Round to nearest half-integer
        W_rounded = np.round(W_sol * 2) / 2
        
        # Check if exact
        if np.allclose(A @ W_rounded, b, atol=1e-9):
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_rounded.reshape(p * n, rank).tolist()
            
            # Final validation
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
                
    return None

def _construct_from_332(U_332, V_332, W_332):
    n, m, p = 3, 3, 8
    rank_332 = len(U_332[0])
    total_rank = 4 * rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_332):
            for block in range(4):
                U[i][block * rank_332 + r] = U_332[i][r]
                
    for j in range(m):
        for k in range(p):
            block = k // 2
            k_local = k % 2
            for r in range(rank_332):
                V[j * p + k][block * rank_332 + r] = V_332[j * 2 + k_local][r]
                
    for k in range(p):
        block = k // 2
        k_local = k % 2
        for i in range(n):
            for r in range(rank_332):
                W[k * n + i][block * rank_332 + r] = W_332[k_local * 3 + i][r]
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    schoolbook = _schoolbook(n, m, p)
    
    # Try Strassen block placements
    strassen_result, strassen_rank = _try_strassen_placements(n, m, p, time_limit=60.0)
    if strassen_result and strassen_rank < n * m * p:
        return strassen_result
    
    # Try to find better 3x3x2 decomposition
    # Start from 17 down to 14
    for r_332 in range(17, 13, -1):
        if time.time() > deadline - 20:
            break
        result_332 = _find_332(r_332, time_limit=15)
        if result_332:
            U, V, W = result_332
            decomp = _construct_from_332(U, V, W)
            if _validate(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                return decomp
                
    # Fallback
    return schoolbook
