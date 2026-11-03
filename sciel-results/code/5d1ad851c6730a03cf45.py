import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
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

    def build_decomp(j_pairs, k_pairs):
        """Build decomposition using Strassen for pairs and standard for singletons."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        used_k = set()
        for k0, k1 in k_pairs:
            used_k.add(k0)
            used_k.add(k1)
        remaining_k = [k for k in range(p) if k not in used_k]
        
        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(remaining_j) * p
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

    def eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms using linear algebra."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        for term_idx in range(rank - 1, -1, -1):
            if time_left() < 5:
                break
            
            # Check if term is redundant by solving for it in terms of others
            # This is a simplified check: if the term's contribution can be zeroed out
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
        
        return U_list, V_list, W_list, rank

    # Generate all possible pairings for j and k
    def generate_pairings(n):
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((i, j))
        return pairs

    j_pairs_options = generate_pairings(m)
    k_pairs_options = generate_pairings(p)
    
    best_result = None
    best_rank = float('inf')
    
    # Try combinations of j and k pairs
    for j_pairs in [j_pairs_options[:i] for i in range(1, min(3, len(j_pairs_options))+1)]:
        for k_pairs in [k_pairs_options[:i] for i in range(1, min(3, len(k_pairs_options))+1)]:
            if time_left() < 10:
                break
            U, V, W, rank = build_decomp(j_pairs, k_pairs)
            
            if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
                continue
            
            # Try to eliminate terms
            U_list, V_list, W_list, new_rank = eliminate_terms(U, V, W, rank)
            
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
