"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

_start_time = time.time()
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    def build_block_decomposition():
        """Build the block decomposition with Strassen for 2x2x2 blocks."""
        terms = []
        
        def add_strassen(a_rows, c_cols, d_cols):
            a0, a1 = a_rows
            c0, c1 = c_cols
            d0, d1 = d_cols
            
            def U(la, lc): return a_rows[la]*m + c_cols[lc]
            def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
            def W(ld, la): return d_cols[ld]*n + a_rows[la]
            
            # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            terms.append((
                [(U(0,0), 1), (U(1,1), 1)],
                [(V(0,0), 1), (V(1,1), 1)],
                [(W(0,0), 1), (W(1,1), 1)]
            ))
            
            # M2 = (A[1,0]+A[1,1])*B[0,0]
            terms.append((
                [(U(1,0), 1), (U(1,1), 1)],
                [(V(0,0), 1)],
                [(W(0,1), 1), (W(1,1), -1)]
            ))
            
            # M3 = A[0,0]*(B[0,1]-B[1,1])
            terms.append((
                [(U(0,0), 1)],
                [(V(0,1), 1), (V(1,1), -1)],
                [(W(1,0), 1), (W(1,1), 1)]
            ))
            
            # M4 = A[1,1]*(B[1,0]-B[0,0])
            terms.append((
                [(U(1,1), 1)],
                [(V(1,0), 1), (V(0,0), -1)],
                [(W(0,0), 1), (W(0,1), 1)]
            ))
            
            # M5 = (A[0,0]+A[0,1])*B[1,1]
            terms.append((
                [(U(0,0), 1), (U(0,1), 1)],
                [(V(1,1), 1)],
                [(W(0,0), -1), (W(1,0), 1)]
            ))
            
            # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            terms.append((
                [(U(1,0), 1), (U(0,0), -1)],
                [(V(0,0), 1), (V(0,1), 1)],
                [(W(1,1), 1)]
            ))
            
            # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            terms.append((
                [(U(0,1), 1), (U(1,1), -1)],
                [(V(1,0), 1), (V(1,1), 1)],
                [(W(0,0), 1)]
            ))
        
        def add_schoolbook(a_rows, c_cols, d_cols):
            """Add schoolbook terms for any block size."""
            for i in a_rows:
                for j in c_cols:
                    for k in d_cols:
                        terms.append((
                            [(i*m + j, 1)],
                            [(j*p + k, 1)],
                            [(k*n + i, 1)]
                        ))
        
        n_block_ranges = [(0, 2), (2, 3)]
        m_block_ranges = [(0, 2), (2, 4), (4, 5)]
        p_block_ranges = [(0, 2), (2, 4), (4, 6)]
        
        for ni, (ns, ne) in enumerate(n_block_ranges):
            for mi, (ms, me) in enumerate(m_block_ranges):
                for pi, (ps, pe) in enumerate(p_block_ranges):
                    n_size = ne - ns
                    m_size = me - ms
                    p_size = pe - ps
                    
                    a_rows = list(range(ns, ne))
                    c_cols = list(range(ms, me))
                    d_cols = list(range(ps, pe))
                    
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        add_strassen(a_rows, c_cols, d_cols)
                    else:
                        add_schoolbook(a_rows, c_cols, d_cols)
        
        return terms
    
    def terms_to_matrices(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row][ci] += coeff
            for row, coeff in v_entries:
                V[row][ci] += coeff
            for row, coeff in w_entries:
                W[row][ci] += coeff
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        rank = len(W[0]) if W else 0
        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
        # where delta = 1 if this is a valid MM tensor entry, 0 otherwise
        # The MM tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def verify_numeric(U, V, W):
        """Fast numeric verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Build the tensor and check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Build initial decomposition
    terms = build_block_decomposition()
    U, V, W = terms_to_matrices(terms)
    
    # Verify
    if not verify_numeric(U, V, W):
        # Fallback to pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        [(i*m + j, 1)],
                        [(j*p + k, 1)],
                        [(k*n + i, 1)]
                    ))
        U, V, W = terms_to_matrices(terms)
    
    # Check deadline
    if time.time() > DEADLINE - 20:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by removing redundant columns
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Try to identify and remove zero columns or linearly dependent columns
    # First, check for columns that are all zeros
    nonzero_cols = []
    for r in range(rank):
        if (U_np[:, r] != 0).any() and (V_np[:, r] != 0).any() and (W_np[:, r] != 0).any():
            nonzero_cols.append(r)
    
    if len(nonzero_cols) < rank:
        U_np = U_np[:, nonzero_cols]
        V_np = V_np[:, nonzero_cols]
        W_np = W_np[:, nonzero_cols]
    
    # Convert back
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    if time.time() > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    # Try local search: see if we can merge any two terms
    # Two terms r1, r2 can be merged if there exists a scalar c such that
    # U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = (1/c)*W[:,r2]
    # This would allow us to eliminate one term.
    
    current_rank = len(W[0]) if W else 0
    improved = True
    while improved and time.time() < DEADLINE - 15:
        improved = False
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        
        for a in range(r):
            if time.time() > DEADLINE - 15:
                break
            for b in range(a+1, r):
                # Check if term b is a scalar multiple of term a in U and V
                # and reciprocal in W
                u_a = U_np[:, a]
                u_b = U_np[:, b]
                v_a = V_np[:, a]
                v_b = V_np[:, b]
                w_a = W_np[:, a]
                w_b = W_np[:, b]
                
                # Find scalar c from U
                nonzero_u_a = np.nonzero(u_a)[0]
                if len(nonzero_u_a) == 0:
                    continue
                idx = nonzero_u_a[0]
                if u_a[idx] == 0:
                    continue
                c = u_b[idx] / u_a[idx]
                
                # Verify c works for all of U
                if not np.allclose(u_b, c * u_a, atol=1e-10):
                    continue
                # Verify c works for all of V
                if not np.allclose(v_b, c * v_a, atol=1e-10):
                    continue
                # Verify 1/c works for W
                if c == 0:
                    continue
                if not np.allclose(w_b, w_a / c, atol=1e-10):
                    continue
                
                # Merge: remove column b, scale column a by (1 + something)
                # Actually, if U_b = c*U_a, V_b = c*V_a, W_b = W_a/c
                # Then term a + term b contributes:
                # U_a * V_a * W_a + c*U_a * c*V_a * W_a/c = U_a*V_a*W_a + c*U_a*V_a*W_a = (1+c)*U_a*V_a*W_a
                # So we can replace term a with scaled version and remove term b
                # New U_a' = (1+c) * U_a, V_a' = V_a, W_a' = W_a
                # Wait, that doesn't work. Let me think again.
                # sum_r U[i,r]*V[j,r]*W[k,r]
                # Term a: U_a[i]*V_a[j]*W_a[k]
                # Term b: c*U_a[i]*c*V_a[j]*(1/c)*W_a[k] = c*U_a[i]*V_a[j]*W_a[k]
                # Total: (1+c)*U_a[i]*V_a[j]*W_a[k]
                # So we need a single term with coefficient (1+c)
                # Set U_a_new = (1+c)*U_a, V_a_new = V_a, W_a_new = W_a
                U_np[:, a] = (1 + c) * u_a
                V_np[:, a] = v_a
                W_np[:, a] = w_a
                # Remove column b
                U_np = np.delete(U_np, b, axis=1)
                V_np = np.delete(V_np, b, axis=1)
                W_np = np.delete(W_np, b, axis=1)
                
                # Verify
                U_check = U_np.tolist()
                V_check = V_np.tolist()
                W_check = W_np.tolist()
                if verify_numeric(U_check, V_check, W_check):
                    U, V, W = U_check, V_check, W_check
                    improved = True
                    break
                else:
                    # Revert
                    U_np = np.array(U, dtype=float)
                    V_np = np.array(V, dtype=float)
                    W_np = np.array(W, dtype=float)
        
        if time.time() > DEADLINE - 15:
            break
    
    # Final conversion ensuring integer/half-integer entries
    U_final = []
    for row in U:
        U_final.append([int(round(x * 2)) / 2 for x in row])
    V_final = []
    for row in V:
        V_final.append([int(round(x * 2)) / 2 for x in row])
    W_final = []
    for row in W:
        W_final.append([int(round(x * 2)) / 2 for x in row])
    
    # Final verification
    if not verify_numeric(U_final, V_final, W_final):
        # Rebuild from original terms
        terms = build_block_decomposition()
        U, V, W = terms_to_matrices(terms)
        U_final = [[float(x) for x in row] for row in U]
        V_final = [[float(x) for x in row] for row in V]
        W_final = [[float(x) for x in row] for row in W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
