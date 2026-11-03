import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank, tol=1e-9):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all (i, j, k) tuples
        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        # Sum over t: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, rank, tol=1e-9):
        """Faster verification using vectorized operations."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all (i, j, k) tuples
        # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        # Sum over t: U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        num_blocks = len(j_pairs) * len(k_pairs)
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for _ in range(10):  # Multiple passes
            if time_left() < 5:
                break
            progress = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Check if term is redundant
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
                    progress = True
            if not progress:
                break

        return U_list, V_list, W_list, rank

    def try_pairwise_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate two terms at once by adding/subtracting them."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 10:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10:
                    return U_list, V_list, W_list, rank
                
                # Try replacing term idx2 with a linear combination
                # Or simply check if we can merge them
                # This is computationally expensive, so we limit attempts
                
                # Simple check: if we zero out both, does it still work?
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][idx1] = 0
                    U_new[i][idx2] = 0
                for j in range(m*p):
                    V_new[j][idx1] = 0
                    V_new[j][idx2] = 0
                for k in range(p*n):
                    W_new[k][idx1] = 0
                    W_new[k][idx2] = 0
                    
                if verify(U_new, V_new, W_new, rank):
                    # Remove both columns
                    keep = [j for j in range(rank) if j != idx1 and j != idx2]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    rank -= 2
                    # Restart loop to find more
                    return try_pairwise_elimination(U_list, V_list, W_list, rank)
                    
        return U_list, V_list, W_list, rank

    def try_triple_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate three terms at once."""
        if time_left() < 15:
            return U_list, V_list, W_list, rank
            
        # Only try if rank is small enough to make this feasible
        if rank > 40:
            return U_list, V_list, W_list, rank
            
        for idx1 in range(rank):
            if time_left() < 15:
                return U_list, V_list, W_list, rank
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 15:
                    return U_list, V_list, W_list, rank
                for idx3 in range(idx2 + 1, rank):
                    if time_left() < 15:
                        return U_list, V_list, W_list, rank
                    
                    U_new = [row[:] for row in U_list]
                    V_new = [row[:] for row in V_list]
                    W_new = [row[:] for row in W_list]
                    
                    for i in range(n*m):
                        U_new[i][idx1] = 0
                        U_new[i][idx2] = 0
                        U_new[i][idx3] = 0
                    for j in range(m*p):
                        V_new[j][idx1] = 0
                        V_new[j][idx2] = 0
                        V_new[j][idx3] = 0
                    for k in range(p*n):
                        W_new[k][idx1] = 0
                        W_new[k][idx2] = 0
                        W_new[k][idx3] = 0
                        
                    if verify(U_new, V_new, W_new, rank):
                        # Remove all three columns
                        keep = [j for j in range(rank) if j != idx1 and j != idx2 and j != idx3]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        rank -= 3
                        # Restart loop to find more
                        return try_triple_elimination(U_list, V_list, W_list, rank)
                        
        return U_list, V_list, W_list, rank

    def try_linear_combination_elimination(U_list, V_list, W_list, rank):
        """Try to eliminate a term by expressing it as a linear combination of others."""
        if time_left() < 20:
            return U_list, V_list, W_list, rank
            
        # For each term, try to see if it can be expressed as a linear combination of others
        # This is more complex, so we limit to small ranks
        if rank > 35:
            return U_list, V_list, W_list, rank
            
        for term_idx in range(rank - 1, -1, -1):
            if time_left() < 20:
                break
            
            # Try to find coefficients c_t such that:
            # U[:, term_idx] = sum_{t != term_idx} c_t * U[:, t]
            # V[:, term_idx] = sum_{t != term_idx} c_t * V[:, t]
            # W[:, term_idx] = sum_{t != term_idx} c_t * W[:, t]
            
            # This is a linear system. We can try to solve it.
            # But this is expensive, so we only try for a few terms
            
            # Build the linear system
            # For U: (n*m) equations, (rank-1) unknowns
            # For V: (m*p) equations, (rank-1) unknowns
            # For W: (p*n) equations, (rank-1) unknowns
            
            # Combine all equations
            all_rows = []
            all_rhs = []
            
            # U equations
            for i in range(n*m):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(U_list[i][t])
                all_rows.append(row)
                all_rhs.append(U_list[i][term_idx])
            
            # V equations
            for j in range(m*p):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(V_list[j][t])
                all_rows.append(row)
                all_rhs.append(V_list[j][term_idx])
            
            # W equations
            for k in range(p*n):
                row = []
                for t in range(rank):
                    if t != term_idx:
                        row.append(W_list[k][t])
                all_rows.append(row)
                all_rhs.append(W_list[k][term_idx])
            
            # Solve the linear system
            A = np.array(all_rows, dtype=float)
            b = np.array(all_rhs, dtype=float)
            
            try:
                # Use least squares
                coeffs, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is good
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-8:
                        continue
                else:
                    # Check residual
                    residual = np.linalg.norm(A @ coeffs - b)
                    if residual > 1e-8:
                        continue
                
                # Check if coefficients are close to integers or half-integers
                coeffs_rounded = np.round(coeffs * 2) / 2
                if np.linalg.norm(coeffs - coeffs_rounded) > 1e-6:
                    continue
                
                # Verify that the linear combination works
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0
                
                if verify(U_new, V_new, W_new, rank):
                    # Remove the column
                    U_list = [[row[j] for j in range(rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(rank) if j != term_idx] for row in W_list]
                    rank -= 1
            except:
                continue

        return U_list, V_list, W_list, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, k_pairs in configurations:
        if time_left() < 15:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
        # Try pairwise elimination if time permits
        if time_left() > 25:
            U_list, V_list, W_list, new_rank = try_pairwise_elimination(U_list, V_list, W_list, new_rank)
        
        # Try triple elimination if time permits
        if time_left() > 35:
            U_list, V_list, W_list, new_rank = try_triple_elimination(U_list, V_list, W_list, new_rank)
        
        # Try linear combination elimination if time permits
        if time_left() > 45:
            U_list, V_list, W_list, new_rank = try_linear_combination_elimination(U_list, V_list, W_list, new_rank)
            
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
