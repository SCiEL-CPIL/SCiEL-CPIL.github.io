"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
_start_time = time.time()

def construct():
    if time.time() - _start_time > 145:
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Build the matrix multiplication tensor and try to find a low-rank decomposition
    # The tensor T[i,j,k] = 1 if i*m+j == k*n (i.e., A[i,j]*B[j,k] contributes to C[i,k])
    # Actually: T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # So T[i,j,k] = 1 if j is a valid index (0<=j<m) and i<n, k<p
    # The tensor has shape (n, m, p) and T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # This is always true for valid i,j,k, so T[i,j,k] = 1 for all valid indices? No.
    # Actually the matrix multiplication tensor is: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right either. The tensor is defined such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if j is the same index in A's columns and B's rows
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k where j is shared
    
    # Actually, the standard matrix multiplication tensor M_{n,m,p} has:
    # (M_{n,m,p})_{i,j,k} = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) of C
    # This means (M_{n,m,p})_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait - that would make it all ones. Let me reconsider.
    
    # The matrix multiplication tensor is: T[i,j,k] = 1 if j is a valid middle index
    # For C = A*B where A is n×m and B is m×p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k in valid ranges
    # This is correct - it's just the all-ones tensor of shape (n,m,p)
    
    # Wait, that can't be right. Let me think again.
    # The tensor decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For standard matmul, T[i,j,k] = 1 (since every A[i,j]*B[j,k] contributes to C[i,k])
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the matmul tensor.
    # The schoolbook decomposition has rank n*m*p (each term is one product).
    # Strassen reduces this for 2×2×2 blocks.
    
    # Let me build the schoolbook decomposition and verify, then apply Strassen.
    
    terms = []
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook_block(a_rows, c_cols, d_cols):
        """Add schoolbook terms for a block."""
        for la, a in enumerate(a_rows):
            for lc, c in enumerate(c_cols):
                for ld, d in enumerate(d_cols):
                    u_idx = a * m + c
                    v_idx = c * p + d
                    w_idx = d * n + a
                    terms.append((
                        [(u_idx, 1)],
                        [(v_idx, 1)],
                        [(w_idx, 1)]
                    ))
    
    def add_laderman(a_rows, c_cols, d_cols):
        """Add 23 Laderman terms for a 3x3x3 block."""
        a = a_rows
        c = c_cols
        d = d_cols
        
        def U(i, j): return a[i]*m + c[j]
        def V(j, k): return c[j]*p + d[k]
        def W(k, i): return d[k]*n + a[i]
        
        # Laderman's 23-term algorithm for 3x3x3
        # Each term: (U_factors, V_factors, W_factors)
        # M1 = (A00+A22)*(B00+B22)
        terms.append(([(U(0,0),1),(U(2,2),1)], [(V(0,0),1),(V(2,2),1)], [(W(0,0),1),(W(2,2),1)]))
        # M2 = (A01+A11+A21)*B00
        terms.append(([(U(0,1),1),(U(1,1),1),(U(2,1),1)], [(V(0,0),1)], [(W(0,0),-1),(W(0,1),1),(W(1,1),1),(W(2,1),1)]))
        # M3 = (A10+A20)*(B10+B20)
        terms.append(([(U(1,0),1),(U(2,0),1)], [(V(1,0),1),(V(2,0),1)], [(W(0,1),-1),(W(0,2),1),(W(1,2),1)]))
        # M4 = A00*(B01-B11)
        terms.append(([(U(0,0),1)], [(V(0,1),1),(V(1,1),-1)], [(W(1,0),1)]))
        # M5 = (A12+A22)*(B11+B21)
        terms.append(([(U(1,2),1),(U(2,2),1)], [(V(1,1),1),(V(2,1),1)], [(W(1,1),1),(W(1,2),-1),(W(2,2),1)]))
        # M6 = A22*(B02-B12)
        terms.append(([(U(2,2),1)], [(V(0,2),1),(V(1,2),-1)], [(W(2,0),1)]))
        # M7 = (A02+A12)*(B20+B22)
        terms.append(([(U(0,2),1),(U(1,2),1)], [(V(2,0),1),(V(2,2),1)], [(W(0,2),-1),(W(1,2),1),(W(2,2),1)]))
        # M8 = A10*(B12-B22)
        terms.append(([(U(1,0),1)], [(V(1,2),1),(V(2,2),-1)], [(W(2,1),1)]))
        # M9 = A20*(B01-B11)
        terms.append(([(U(2,0),1)], [(V(0,1),1),(V(1,1),-1)], [(W(1,0),1)]))
        # M10 = (A01+A21)*B12
        terms.append(([(U(0,1),1),(U(2,1),1)], [(V(1,2),1)], [(W(0,1),1),(W(2,1),-1)]))
        # M11 = A11*(B20-B00)
        terms.append(([(U(1,1),1)], [(V(2,0),1),(V(0,0),-1)], [(W(0,0),1)]))
        # M12 = A21*(B22-B02)
        terms.append(([(U(2,1),1)], [(V(2,2),1),(V(0,2),-1)], [(W(2,0),1)]))
        # M13 = (A00+A01+A02)*B21
        terms.append(([(U(0,0),1),(U(0,1),1),(U(0,2),1)], [(V(2,1),1)], [(W(0,0),1),(W(0,1),-1),(W(0,2),1)]))
        # M14 = (A10+A11+A12)*B02
        terms.append(([(U(1,0),1),(U(1,1),1),(U(1,2),1)], [(V(0,2),1)], [(W(0,1),1),(W(0,2),-1),(W(1,2),1)]))
        # M15 = (A20+A21+A22)*B10
        terms.append(([(U(2,0),1),(U(2,1),1),(U(2,2),1)], [(V(1,0),1)], [(W(0,1),1),(W(0,2),-1),(W(2,2),1)]))
        # M16 = A02*B10
        terms.append(([(U(0,2),1)], [(V(1,0),1)], [(W(0,2),1)]))
        # M17 = A12*B20
        terms.append(([(U(1,2),1)], [(V(2,0),1)], [(W(1,2),1)]))
        # M18 = A20*B01
        terms.append(([(U(2,0),1)], [(V(0,1),1)], [(W(0,1),1)]))
        # M19 = A01*B20
        terms.append(([(U(0,1),1)], [(V(2,0),1)], [(W(0,0),1)]))
        # M20 = A11*B02
        terms.append(([(U(1,1),1)], [(V(0,2),1)], [(W(0,0),1)]))
        # M21 = A21*B12
        terms.append(([(U(2,1),1)], [(V(1,2),1)], [(W(1,1),1)]))
        # M22 = A02*B21
        terms.append(([(U(0,2),1)], [(V(2,1),1)], [(W(1,2),1)]))
        # M23 = A10*B22
        terms.append(([(U(1,0),1)], [(V(2,2),1)], [(W(2,0),1)]))
    
    # Block partition: n=3 as [3], m=5 as [3,2], p=6 as [3,3]
    n_block_ranges = [(0, 3)]
    m_block_ranges = [(0, 3), (3, 5)]
    p_block_ranges = [(0, 3), (3, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                a_rows = list(range(ns, ne))
                c_cols = list(range(ms, me))
                d_cols = list(range(ps, pe))
                
                if n_size == 3 and m_size == 3 and p_size == 3:
                    add_laderman(a_rows, c_cols, d_cols)
                elif n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                else:
                    add_schoolbook_block(a_rows, c_cols, d_cols)
    
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
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if Strassen is wrong
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)]))
        
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
