import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition exactly using integer arithmetic."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, j_singles, k_pairs):
        """
        Build decomposition using Strassen 2x2x2 blocks for paired j,k
        and standard terms for single j with paired k.
        
        j_pairs: list of (j0, j1) pairs
        j_singles: list of single j indices
        k_pairs: list of (k0, k1) pairs
        """
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        
        # Ensure all j's are covered
        all_j = set(range(m))
        covered_j = used_j | set(j_singles)
        assert covered_j == all_j, f"j coverage mismatch: {covered_j} vs {all_j}"
        
        # Count rank
        num_j_pairs = len(j_pairs)
        num_k_pairs = len(k_pairs)
        num_j_singles = len(j_singles)
        
        # Each j_pair x k_pair: 7 terms (Strassen)
        # Each j_single x k_pair: 4 terms (2 i's x 2 k's)
        rank = 7 * num_j_pairs * num_k_pairs + 4 * num_j_singles * num_k_pairs
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for 2x2x2 block (i=0,1; j=j0,j1; k=k0,k1)
                # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                
                # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                
                # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                
                # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                
                # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                
                # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                
                # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        # Single j with k_pairs: standard 2x1x2 blocks
        for j in j_singles:
            for k0, k1 in k_pairs:
                # For i=0: A[0,j]*B[j,k0] and A[0,j]*B[j,k1]
                U[0*m+j, col] += 1
                V[j*p+k0, col] += 1
                W[k0*n+0, col] += 1
                col += 1
                
                U[0*m+j, col] += 1
                V[j*p+k1, col] += 1
                W[k1*n+0, col] += 1
                col += 1
                
                # For i=1: A[1,j]*B[j,k0] and A[1,j]*B[j,k1]
                U[1*m+j, col] += 1
                V[j*p+k0, col] += 1
                W[k0*n+1, col] += 1
                col += 1
                
                U[1*m+j, col] += 1
                V[j*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank, max_iterations=200):
        """Try to eliminate redundant terms by checking if removing a column still verifies."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        iterations = 0
        while iterations < max_iterations:
            if time_left() < 10:
                break
            iterations += 1
            found = False
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
                    found = True
                    break
            
            if not found:
                break
        
        return U_list, V_list, W_list, rank

    # Try different configurations
    # j dimension: 5 elements, k dimension: 6 elements
    configurations = [
        # 2 j-pairs, 1 j-single; 3 k-pairs
        ([(0,1),(2,3)], [4], [(0,1),(2,3),(4,5)]),
        ([(0,1),(3,4)], [2], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [0], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [4], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [2], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [2], [(0,1),(2,3),(4,5)]),
        # Different k pairings
        ([(0,1),(2,3)], [4], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [4], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,3)], [4], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [4], [(0,5),(1,2),(3,4)]),
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for j_pairs, j_singles, k_pairs in configurations:
        if time_left() < 15:
            break
        
        U, V, W, rank = build_strassen_decomp(j_pairs, j_singles, k_pairs)
        
        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue
        
        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)
        
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
