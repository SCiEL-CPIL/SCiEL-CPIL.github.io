import numpy as np
import time
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 4, 4
    # Target tensor T
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    # x in 0..7, y in 0..15, z in 0..7
    T = np.zeros((8, 16, 8), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Known solution with rank 28
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(rank):
            T_est += np.einsum('r,s,t->rst', U_np[:, r], V_np[:, r], W_np[:, r])
        return np.allclose(T_est, T, atol=1e-9)

    def compute_catalecticant_ranks(T_tensor):
        """
        Compute ranks of catalecticant (flattening) matrices to get lower bounds on tensor rank.
        For a tensor of shape (n*m, m*p, p*n) = (8, 16, 8):
        - Flatten along axis 0: matrix of shape (8, 16*8) -> rank gives lower bound
        - Flatten along axis 1: matrix of shape (16, 8*8) -> rank gives lower bound  
        - Flatten along axis 2: matrix of shape (8, 8*16) -> rank gives lower bound
        """
        # Flatten along first axis: (8, 16*8)
        flat0 = T_tensor.reshape(8, -1)
        rank0 = np.linalg.matrix_rank(flat0)
        
        # Flatten along second axis: (16, 8*8)
        flat1 = T_tensor.transpose(1, 0, 2).reshape(16, -1)
        rank1 = np.linalg.matrix_rank(flat1)
        
        # Flatten along third axis: (8, 8*16)
        flat2 = T_tensor.transpose(2, 0, 1).reshape(8, -1)
        rank2 = np.linalg.matrix_rank(flat2)
        
        # The tensor rank is at least the max of these catalecticant ranks
        lower_bound = max(rank0, rank1, rank2)
        return lower_bound, rank0, rank1, rank2

    # Compute the lower bound
    lb, r0, r1, r2 = compute_catalecticant_ranks(T)
    print(f"Catalecticant ranks: {r0}, {r1}, {r2}; Lower bound: {lb}")

    # Start with the known rank 28 solution
    best_rank = rank_known
    best_U = [row[:] for row in U_known]
    best_V = [row[:] for row in V_known]
    best_W = [row[:] for row in W_known]

    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # If the lower bound is less than 28, try to find a lower rank solution
    # Use the structure of the problem to guide the search
    U_np = np.array(U_known, dtype=float)
    V_np = np.array(V_known, dtype=float)
    W_np = np.array(W_known, dtype=float)
    current_rank = rank_known

    # Try to find linear dependencies among the rank-1 components
    # Each component r contributes: u_r ⊗ v_r ⊗ w_r
    # We can try to express one component as a linear combination of others
    
    # For efficiency, work with flattened vectors
    def flatten_component(u, v, w):
        return np.outer(np.outer(u, v), w).flatten()

    # Build matrix of all components
    components = []
    for r in range(current_rank):
        comp = flatten_component(U_np[:, r], V_np[:, r], W_np[:, r])
        components.append(comp)
    
    components_np = np.array(components)  # shape (rank, 8*16*8)

    # Check if any component is in the span of the others
    # This is equivalent to checking if the rank of the full matrix equals
    # the rank when we remove one column
    
    improved = False
    for attempt in range(current_rank):
        if time.time() > deadline:
            break
        
        # Try removing component 'attempt'
        other_indices = [i for i in range(current_rank) if i != attempt]
        if len(other_indices) == 0:
            continue
        
        others = components_np[other_indices]
        target = components_np[attempt]
        
        # Check if target is in the column space of others
        try:
            # Use SVD to check
            U_svd, s, Vt = np.linalg.svd(others, full_matrices=False)
            # Project target onto the row space of others
            proj = U_svd @ (U_svd.T @ target)
            residual = np.linalg.norm(target - proj)
            
            # Normalize by the norm of target for relative check
            target_norm = np.linalg.norm(target)
            if target_norm > 1e-10 and residual / target_norm < 1e-8:
                # Component 'attempt' is redundant!
                new_indices = [i for i in range(current_rank) if i != attempt]
                
                new_U = U_np[:, new_indices]
                new_V = V_np[:, new_indices]
                new_W = W_np[:, new_indices]
                
                # Verify the solution
                T_est = np.zeros((8, 16, 8), dtype=float)
                for r in range(current_rank - 1):
                    T_est += np.einsum('r,s,t->rst', new_U[:, r], new_V[:, r], new_W[:, r])
                
                if np.allclose(T_est, T, atol=1e-9):
                    # Success! We reduced the rank
                    current_rank -= 1
                    U_np = new_U
                    V_np = new_V
                    W_np = new_W
                    
                    best_rank = current_rank
                    best_U = new_U.tolist()
                    best_V = new_V.tolist()
                    best_W = new_W.tolist()
                    improved = True
                    
                    # Rebuild components for next iteration
                    components = []
                    for r in range(current_rank):
                        comp = flatten_component(U_np[:, r], V_np[:, r], W_np[:, r])
                        components.append(comp)
                    components_np = np.array(components)
                    
                    # Continue trying to reduce further
                    attempt = 0  # Reset to check from beginning again
                    continue
        except:
            pass
    
    # If no reduction found, return the best solution
    # Round to nearest 0.5 to ensure exact integer/half-integer entries
    U_final = np.round(np.array(best_U) * 2) / 2
    V_final = np.round(np.array(best_V) * 2) / 2
    W_final = np.round(np.array(best_W) * 2) / 2
    
    # Final verification
    T_est = np.zeros((8, 16, 8), dtype=float)
    for r in range(best_rank):
        T_est += np.einsum('r,s,t->rst', U_final[:, r], V_final[:, r], W_final[:, r])
    
    if not np.allclose(T_est, T, atol=1e-9):
        # If rounding broke the solution, use the original
        U_final = np.array(best_U, dtype=float)
        V_final = np.array(best_V, dtype=float)
        W_final = np.array(best_W, dtype=float)
        # Re-verify
        T_est = np.zeros((8, 16, 8), dtype=float)
        for r in range(best_rank):
            T_est += np.einsum('r,s,t->rst', U_final[:, r], V_final[:, r], W_final[:, r])
        if not np.allclose(T_est, T, atol=1e-9):
            # Fall back to the known good solution
            U_final = np.array(U_known, dtype=float)
            V_final = np.array(V_known, dtype=float)
            W_final = np.array(W_known, dtype=float)
            best_rank = rank_known

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
