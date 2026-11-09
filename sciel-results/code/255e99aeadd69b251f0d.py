"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Schoolbook solution (always valid)
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Fast verification using numpy
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by testing if any column can be removed
    def try_remove_columns(U, V, W, max_attempts=50):
        """Iteratively try to remove columns that are redundant."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for _ in range(max_attempts):
            if time_left() < 3:
                break
            found = False
            for c in range(rank):
                if time_left() < 3:
                    break
                # Try removing column c
                u_col = [U[i][c] for i in range(len(U))]
                v_col = [V[i][c] for i in range(len(V))]
                w_col = [W[i][c] for i in range(len(W))]
                
                for i in range(len(U)):
                    U[i].pop(c)
                for i in range(len(V)):
                    V[i].pop(c)
                for i in range(len(W)):
                    W[i].pop(c)
                rank -= 1
                
                if verify_np(U, V, W):
                    found = True
                    break
                else:
                    for i in range(len(U)):
                        U[i].insert(c, u_col[i])
                    for i in range(len(V)):
                        V[i].insert(c, v_col[i])
                    for i in range(len(W)):
                        W[i].insert(c, w_col[i])
                    rank += 1
            
            if not found:
                break
        
        return U, V, W, rank
    
    # Try merging two columns: if col_a + col_b can replace both
    def try_merge_columns(U, V, W, max_attempts=30):
        """Try to merge pairs of columns to reduce rank."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for _ in range(max_attempts):
            if time_left() < 5:
                break
            found = False
            for c1 in range(rank):
                if time_left() < 5:
                    break
                for c2 in range(c1 + 1, rank):
                    if time_left() < 5:
                        break
                    # Try replacing cols c1,c2 with a single new column
                    # New column values: we need to find x,y,z such that
                    # for all (i,j,k): sum over remaining + x_i*y_j*z_k = 1
                    
                    # This is complex; skip for now and use random perturbation
                    # Instead, try small integer perturbations
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dz in [-1, 0, 1]:
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                # Create modified version
                                U_mod = [row[:] for row in U]
                                V_mod = [row[:] for row in V]
                                W_mod = [row[:] for row in W]
                                
                                # Remove col c2, modify col c1
                                for i in range(len(U_mod)):
                                    U_mod[i].pop(c2)
                                for i in range(len(V_mod)):
                                    V_mod[i].pop(c2)
                                for i in range(len(W_mod)):
                                    W_mod[i].pop(c2)
                                
                                # Add dx, dy, dz to col c1
                                for i in range(len(U_mod)):
                                    U_mod[i][c1] += dx
                                for i in range(len(V_mod)):
                                    V_mod[i][c1] += dy
                                for i in range(len(W_mod)):
                                    W_mod[i][c1] += dz
                                
                                if verify_np(U_mod, V_mod, W_mod):
                                    U, V, W = U_mod, V_mod, W_mod
                                    rank -= 1
                                    found = True
                                    break
                                if time_left() < 5:
                                    break
                            if found or time_left() < 5:
                                break
                        if found or time_left() < 5:
                            break
                    if found or time_left() < 5:
                        break
                if found or time_left() < 5:
                    break
            if not found:
                break
        
        return U, V, W, rank
    
    # Strassen-style block decomposition for 3x4 * 4x6
    # Split 4x4 middle into 2x2 blocks of 2x2
    # A is 3x4: split into top 2 rows and bottom 1 row
    # B is 4x6: split into left 2 cols and right 4 cols? No, need compatible splits
    # Better: treat 4 as 2+2, so A becomes [A_top(2x4), A_bot(1x4)]
    # B becomes [B_left(4x2), B_right(4x4)]? That doesn't work for Strassen
    # 
    # Alternative: Use the fact that 4 = 2*2 and apply Strassen to each 2x2 block
    # of the middle dimension. Split A into 3 blocks of height 2 (padding last to 2x2)
    # Actually, let's use a different approach: direct tensor decomposition search
    # with structured initialization based on block patterns
    
    # Create structured initial solution using block patterns
    def make_structured_init():
        """Create a structured initial decomposition using block patterns."""
        # We'll use a pattern where we exploit the 2x2 structure of the middle dimension
        # Initialize with schoolbook, then try to find linear dependencies
        
        # Start with a smaller rank guess based on theoretical bounds
        # For 3x4x6, lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
        # Upper bound from schoolbook is 72
        # Try to find something in between using structured patterns
        
        rank = 36  # Start with half of schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Use a pattern based on 2x2 Strassen blocks
        # Split indices: A[i,j] where i in [0,3), j in [0,4)
        # B[j,k] where j in [0,4), k in [0,6)
        # C[i,k] where i in [0,3), k in [0,6)
        
        # Create 7 Strassen-like terms for each 2x2 block pair
        # But we have 3 rows in A and 6 cols in B, so we need to handle this carefully
        
        # For now, use a random structured initialization
        for r in range(rank):
            # Assign non-zero values with some structure
            pattern = random.choice([0, 1, 2, 3])
            for i in range(n):
                for j in range(m):
                    if (i + j + r) % (pattern + 2) == 0:
                        U[i * m + j][r] = random.choice([-1, 1])
            
            for j in range(m):
                for k in range(p):
                    if (j + k + r) % (pattern + 2) == 0:
                        V[j * p + k][r] = random.choice([-1, 1])
            
            for k in range(p):
                for i in range(n):
                    if (k + i + r) % (pattern + 2) == 0:
                        W[k * n + i][r] = random.choice([-1, 1])
        
        return U, V, W, rank
    
    # Greedy construction: build decomposition term by term
    def greedy_build(target_rank):
        """Build a decomposition greedily, adding one term at a time."""
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Track which (i,j,k) triples are satisfied
        satisfied = set()
        all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for r in range(target_rank):
            if time_left() < 2:
                break
            
            # Find unsatisfied triples
            unsatisfied = [t for t in all_triples if t not in satisfied]
            if not unsatisfied:
                break
            
            # Try to find a rank-1 term that satisfies many unsatisfied triples
            best_term = None
            best_count = 0
            
            # Sample some candidate terms
            for _ in range(100):
                # Generate random rank-1 term
                u_vec = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                v_vec = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                w_vec = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                
                # Count how many unsatisfied triples this term would satisfy
                count = 0
                for (i, j, k) in unsatisfied:
                    ui = u_vec[i * m + j]
                    vj = v_vec[j * p + k]
                    wk = w_vec[k * n + i]
                    if ui * vj * wk == 1:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_term = (u_vec, v_vec, w_vec)
            
            if best_term and best_count > 0:
                u_vec, v_vec, w_vec = best_term
                for idx in range(n * m):
                    U[idx][r] = u_vec[idx]
                for idx in range(m * p):
                    V[idx][r] = v_vec[idx]
                for idx in range(p * n):
                    W[idx][r] = w_vec[idx]
                
                # Update satisfied set
                for (i, j, k) in all_triples:
                    ui = U[i * m + j][r]
                    vj = V[j * p + k][r]
                    wk = W[k * n + i][r]
                    if ui * vj * wk == 1:
                        satisfied.add((i, j, k))
        
        return U, V, W, len(satisfied)
    
    # Start from schoolbook and try reductions
    U_cur, V_cur, W_cur = (
        best_result["U"][:], best_result["V"][:], best_result["W"][:]
    )
    U_cur, V_cur, W_cur, rank_cur = try_remove_columns(U_cur, V_cur, W_cur, max_attempts=30)
    if rank_cur < best_rank and verify_np(U_cur, V_cur, W_cur):
        best_rank = rank_cur
        best_result = {"U": U_cur, "V": V_cur, "W": W_cur}
    
    U_cur, V_cur, W_cur = (
        best_result["U"][:], best_result["V"][:], best_result["W"][:]
    )
    U_cur, V_cur, W_cur, rank_cur = try_merge_columns(U_cur, V_cur, W_cur, max_attempts=20)
    if rank_cur < best_rank and verify_np(U_cur, V_cur, W_cur):
        best_rank = rank_cur
        best_result = {"U": U_cur, "V": V_cur, "W": W_cur}
    
    # Try greedy construction with various target ranks
    if time_left() > 20:
        for target_rank in [48, 42, 36, 30, 27]:
            if time_left() < 10:
                break
            if target_rank >= best_rank:
                continue
            
            U_g, V_g, W_g, sat_count = greedy_build(target_rank)
            if verify_np(U_g, V_g, W_g):
                best_rank = target_rank
                best_result = {"U": U_g, "V": V_g, "W": W_g}
                break
    
    # Random search with structured bias
    if time_left() > 15:
        for target_rank in range(max(24, best_rank - 10), best_rank):
            if time_left() < 8:
                break
            
            attempts = 0
            max_attempts = 30
            while time_left() > 5 and attempts < max_attempts:
                attempts += 1
                
                # Use sparse random matrices with small integers
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                # Fill with sparse random values
                for _ in range(target_rank * 3):
                    r = random.randint(0, target_rank - 1)
                    ui = random.randint(0, n * m - 1)
                    U[ui][r] = random.choice([-1, 1, 2, -2])
                    
                    vj = random.randint(0, m * p - 1)
                    V[vj][r] = random.choice([-1, 1, 2, -2])
                    
                    wk = random.randint(0, p * n - 1)
                    W[wk][r] = random.choice([-1, 1, 2, -2])
                
                if verify_np(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    # Final verification
    if not verify_np(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
