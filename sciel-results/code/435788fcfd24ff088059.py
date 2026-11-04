"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72
    
    def build_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    def check_decomposition(U, V, W):
        """Check if decomposition exactly represents matrix multiplication."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    def try_rank_reduction(current_U, current_V, current_W, target_rank, max_attempts=500):
        """Try to reduce rank by randomly zeroing out columns and re-solving."""
        U = [row[:] for row in current_U]
        V = [row[:] for row in current_V]
        W = [row[:] for row in current_W]
        current_rank = len(U[0])
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            if current_rank <= target_rank:
                break
            
            # Pick a random column to try to eliminate
            col_to_eliminate = random.randint(0, current_rank - 1)
            
            # Try to absorb its contribution into other columns
            # For each (i,j,k) triple that this column contributes to,
            # we need to adjust other columns to compensate
            
            # First, check what this column contributes
            U_col = [U[i * m + j][col_to_eliminate] for i in range(n) for j in range(m)]
            V_col = [V[j * p + k][col_to_eliminate] for j in range(m) for k in range(p)]
            W_col = [W[k * n + i][col_to_eliminate] for k in range(p) for i in range(n)]
            
            # Find all (i,j,k) where this column has nonzero product
            affected = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i * m + j][col_to_eliminate] * V[j * p + k][col_to_eliminate] * W[k * n + i][col_to_eliminate]
                        if prod != 0:
                            affected.append((i, j, k, prod))
            
            if not affected:
                # Column is zero, just remove it
                U = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in U]
                V = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in V]
                W = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in W]
                current_rank -= 1
                continue
            
            # Try to find another column that can absorb this contribution
            # This is complex; instead, try a simpler approach:
            # Randomly perturb and check
            
            # Alternative: try swapping this column's role with another
            other_col = random.randint(0, current_rank - 1)
            if other_col == col_to_eliminate:
                continue
            
            # Try to see if we can modify other_col to also cover col_to_eliminate's contribution
            # This requires solving a system; for now, just try random modifications
            
            # Modify U, V, W at other_col to try to cover the gap
            success = False
            for _ in range(20):
                U_test = [row[:] for row in U]
                V_test = [row[:] for row in V]
                W_test = [row[:] for row in W]
                
                # Zero out col_to_eliminate
                for idx in range(n * m):
                    U_test[idx][col_to_eliminate] = 0
                for idx in range(m * p):
                    V_test[idx][col_to_eliminate] = 0
                for idx in range(p * n):
                    W_test[idx][col_to_eliminate] = 0
                
                # Try to add the missing contributions to other_col
                # For each affected (i,j,k), we need to add prod to the sum at other_col
                # U_test[i*m+j][other_col] * V_test[j*p+k][other_col] * W_test[k*n+i][other_col]
                # should gain 'prod'
                
                # This is a nonlinear system; try random small adjustments
                for (i, j, k, prod) in affected:
                    # Randomly adjust one of the three factors
                    choice = random.randint(0, 2)
                    if choice == 0:
                        if U_test[i * m + j][other_col] != 0:
                            U_test[i * m + j][other_col] += prod / (V_test[j * p + k][other_col] * W_test[k * n + i][other_col])
                        elif V_test[j * p + k][other_col] != 0:
                            U_test[i * m + j][other_col] = 1
                            U_test[i * m + j][other_col] += prod / (V_test[j * p + k][other_col] * W_test[k * n + i][other_col])
                    elif choice == 1:
                        if V_test[j * p + k][other_col] != 0:
                            V_test[j * p + k][other_col] += prod / (U_test[i * m + j][other_col] * W_test[k * n + i][other_col])
                        elif U_test[i * m + j][other_col] != 0:
                            V_test[j * p + k][other_col] = 1
                            V_test[j * p + k][other_col] += prod / (U_test[i * m + j][other_col] * W_test[k * n + i][other_col])
                    else:
                        if W_test[k * n + i][other_col] != 0:
                            W_test[k * n + i][other_col] += prod / (U_test[i * m + j][other_col] * V_test[j * p + k][other_col])
                        elif U_test[i * m + j][other_col] != 0:
                            W_test[k * n + i][other_col] = 1
                            W_test[k * n + i][other_col] += prod / (U_test[i * m + j][other_col] * V_test[j * p + k][other_col])
                
                # Check if this works
                if check_decomposition(U_test, V_test, W_test):
                    # Remove the eliminated column
                    U = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in U_test]
                    V = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in V_test]
                    W = [row[:col_to_eliminate] + row[col_to_eliminate+1:] for row in W_test]
                    current_rank -= 1
                    success = True
                    break
            
            if success:
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_result = {"U": U, "V": V, "W": W}
        
        return U, V, W, current_rank
    
    # Try to reduce from schoolbook
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    
    # Attempt rank reduction
    U, V, W, final_rank = try_rank_reduction(U, V, W, schoolbook_rank - 1, max_attempts=200)
    
    if final_rank < best_rank:
        best_rank = final_rank
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
