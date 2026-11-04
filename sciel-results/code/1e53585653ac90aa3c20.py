"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build rank-50 Strassen-based decomposition
    rank_init = 50
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1

        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1

        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1

        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 6 Strassen blocks for k=0..5
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    assert col == rank_init, f"Expected {rank_init}, got {col}"

    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify using einsum
    recon = np.einsum('ir,jr,kr->ijk', U_np.reshape(n, m, rank_init), 
                       V_np.reshape(m, p, rank_init), 
                       W_np.reshape(p, n, rank_init))
    if not np.allclose(recon, 1.0, atol=1e-10):
        return {"U": U, "V": V, "W": W}

    # Build joint matrix: J[(i,j,k), r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need J @ ones = ones (each row sums to 1)
    J = np.zeros((n * m * p, rank_init), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank_init):
                    J[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                idx += 1

    target = np.ones(n * m * p)
    
    # Check if current decomposition is valid
    if not np.allclose(J.sum(axis=1), target, atol=1e-10):
        return {"U": U, "V": V, "W": W}

    # Strategy: find the null space of J and use it to eliminate terms
    # If v is in null(J), then J @ v = 0, meaning the "weighted sum" of columns
    # with weights v gives zero contribution. We can't directly use this to reduce rank,
    # but we can check if removing a set S of columns and adjusting weights works.
    
    # Better: try to find a subset of columns whose sum equals the target vector.
    # This is a combinatorial problem, but we can use linear algebra heuristics.
    
    # Approach 1: Check if any single column can be removed (sum of rest = target)
    best_rank = rank_init
    best_indices = list(range(rank_init))
    
    # Try removing 1 column
    for r_remove in range(rank_init):
        remaining = [r for r in range(rank_init) if r != r_remove]
        col_sum = J[:, remaining].sum(axis=1)
        if np.allclose(col_sum, target, atol=1e-10):
            best_rank = rank_init - 1
            best_indices = remaining
            break
    
    # Try removing 2 columns
    if best_rank == rank_init:
        for r1 in range(rank_init):
            if time.time() - start_time > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, rank_init):
                remaining = [r for r in range(rank_init) if r != r1 and r != r2]
                col_sum = J[:, remaining].sum(axis=1)
                if np.allclose(col_sum, target, atol=1e-10):
                    best_rank = rank_init - 2
                    best_indices = remaining
                    break
            if best_rank < rank_init:
                break
    
    # Try removing 3 columns
    if best_rank == rank_init:
        for r1 in range(rank_init):
            if time.time() - start_time > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, rank_init):
                for r3 in range(r2 + 1, rank_init):
                    remaining = [r for r in range(rank_init) if r not in (r1, r2, r3)]
                    col_sum = J[:, remaining].sum(axis=1)
                    if np.allclose(col_sum, target, atol=1e-10):
                        best_rank = rank_init - 3
                        best_indices = remaining
                        break
                if best_rank < rank_init:
                    break
            if best_rank < rank_init:
                break
    
    # If no simple removal works, try a more advanced approach:
    # Find if there's a linear combination of columns that equals the target
    # with fewer non-zero coefficients. Use L1-minimization via greedy.
    
    if best_rank == rank_init and time.time() - start_time < DEADLINE - 20:
        # Try to find a sparse solution to J @ x = target where x has 0/1 entries
        # Use greedy forward selection
        selected = []
        residual = target.copy()
        J_cols = J  # J[:, r] is column r
        
        for _ in range(rank_init):
            if time.time() - start_time > DEADLINE - 10:
                break
            # Find the column that best reduces the residual
            best_r = -1
            best_norm = np.inf
            for r in range(rank_init):
                if r in selected:
                    continue
                # Check if adding column r reduces the residual
                new_residual = residual - J[:, r]
                norm = np.linalg.norm(new_residual)
                if norm < best_norm:
                    best_norm = norm
                    best_r = r
                # Also check if removing from residual works (negative weight)
                new_residual2 = residual + J[:, r]
                norm2 = np.linalg.norm(new_residual2)
                if norm2 < best_norm:
                    best_norm = norm2
                    best_r = -r - rank_init  # encode negative
            if best_r >= 0:
                selected.append(best_r)
                residual -= J[:, best_r]
            elif best_r < 0:
                actual_r = -best_r - rank_init
                selected.append(-actual_r)  # negative weight
                residual += J[:, actual_r]
            if best_norm < 1e-10:
                break
        
        # Check if we found a valid sparse solution
        if np.linalg.norm(residual) < 1e-10:
            # Extract positive and negative indices
            pos = [r for r in selected if r >= 0]
            neg = [-r for r in selected if r < 0]
            # Verify: sum of pos columns - sum of neg columns = target
            if pos and not neg:
                col_sum = J[:, pos].sum(axis=1)
                if np.allclose(col_sum, target, atol=1e-10):
                    if len(pos) < best_rank:
                        best_rank = len(pos)
                        best_indices = pos
    
    # Reconstruct with reduced rank if found
    if best_rank < rank_init:
        new_U = U_np[:, best_indices]
        new_V = V_np[:, best_indices]
        new_W = W_np[:, best_indices]
        
        # Verify
        recon2 = np.einsum('ir,jr,kr->ijk', new_U.reshape(n, m, best_rank), 
                           new_V.reshape(m, p, best_rank), 
                           new_W.reshape(p, n, best_rank))
        if np.allclose(recon2, 1.0, atol=1e-10):
            # Convert back to lists and clean up
            U_out = new_U.tolist()
            V_out = new_V.tolist()
            W_out = new_W.tolist()
            for mat in [U_out, V_out, W_out]:
                for row in mat:
                    for idx2 in range(len(row)):
                        if abs(row[idx2] - round(row[idx2])) < 1e-10:
                            row[idx2] = round(row[idx2])
            return {"U": U_out, "V": V_out, "W": W_out}

    # Final verification of the rank-50 decomposition
    recon_final = np.einsum('ir,jr,kr->ijk', U_np.reshape(n, m, rank_init), 
                            V_np.reshape(m, p, rank_init), 
                            W_np.reshape(p, n, rank_init))
    if not np.allclose(recon_final, 1.0, atol=1e-9):
        # Fallback: return as-is
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
