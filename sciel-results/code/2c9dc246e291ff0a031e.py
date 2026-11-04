import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # N: 3 -> 2 + 1
    # M: 5 -> 2 + 2 + 1
    # P: 6 -> 2 + 2 + 2
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
        
    def add_strassen(n_start, m_start, p_start):
        # Indices for the 2x2x2 block
        # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
        # Block local indices:
        # A rows: 0, 1 (global: n_start, n_start+1)
        # B cols: 0, 1 (global: m_start, m_start+1)
        # C cols: 0, 1 (global: p_start, p_start+1)
        
        # Strassen 7 terms for C = A * B
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A10) * B11
        # M6 = (A10 - A00) * B10
        # M7 = (A11 - A01) * B01
        
        # Result C:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M2 + M4
        # C10 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # We define the terms (u, v, w) such that sum(u_k * v_k) * w_k = C
        # Actually, the tensor is T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Here i is row of A (local 0,1), j is col of A/row of B (local 0,1), k is col of B (local 0,1)
        # Wait, the standard matrix multiplication tensor T_{i,j,k} corresponds to A_{i,l} B_{l,k}.
        # In the factorization U, V, W:
        # U has shape (n*m, R). Index i in U is (row_A, col_A) = (i, j) ? No.
        # Let's check the interface:
        # U: (n*m, rank). Index is n*m. Usually flattened (row, col) of A?
        # V: (m*p, rank). Index is m*p. Usually flattened (row, col) of B?
        # W: (p*n, rank). Index is p*n. Output C is (n, p).
        # The trace convention W[k*n+i, r] suggests W index is (col_C, row_C) or similar?
        # Standard: C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # Tensor T_{i, j, k} = A_{i,j} B_{j,k}.
        # Factorization: T_{i, j, k} = sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}?
        # Let's look at the previous code's mapping:
        # U(row, coeff) where row = a_rows[la]*m + c_cols[lc]. This is (row_A, col_A).
        # V(row, coeff) where row = c_cols[lc]*p + d_cols[ld]. This is (row_B, col_B). Note row_B = col_A.
        # W(row, coeff) where row = d_cols[ld]*n + a_rows[la]. This is (col_B, row_A).
        # So the term is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k].
        # This matches C_{i,k} = sum_j A_{i,j} B_{j,k} if we sum over j? No, the sum is over r.
        # The identity is: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = C_{i,k} ?
        # No, that would require W to depend on i,k.
        # The standard tensor decomposition for matmul is:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is NOT correct.
        # The tensor is T_{i,j,k} = delta_{i, i} delta_{j, j} delta_{k, k}? No.
        # The multiplication tensor T has components T_{i,j,k} which is 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
        # Actually, the tensor is defined on indices (i, j, k) where i is row of C, j is inner dim, k is col of C.
        # T_{i,j,k} = 1.
        # Decomposition: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
        # If U, V, W are as defined in the code:
        # U index: (i, j)
        # V index: (j, k)
        # W index: (k, i)
        # Then sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = C_{i,k}.
        # This is the correct form.
        
        # Local indices for 2x2x2:
        # i in {0, 1}, j in {0, 1}, k in {0, 1}
        
        # Term 1: M1
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (0,1):-1, (1,0):-1, (1,1):1  <-- Wait, C00=M1-M4+M5+M7, C01=M2+M4, C10=M3+M5, C11=M1-M2+M3+M6
        # M1 contributes to C00 (+1) and C11 (+1).
        # So W for M1: W_{0,0}=1 (C00), W_{1,1}=1 (C11). Others 0.
        
        # Let's list the 7 terms with their U, V, W local entries.
        # U index (i,j), V index (j,k), W index (k,i)
        
        # 1. M1 = (A00+A11)(B00+B11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,1):1  (Contributes to C00 and C11)
        
        # 2. M2 = (A10+A11)B00
        # U: (1,0):1, (1,1):1
        # V: (0,0):1
        # W: (0,1):1, (1,1):-1 (Contributes to C01 (+1) and C11 (-1))
        
        # 3. M3 = A00(B10-B11)
        # U: (0,0):1
        # V: (1,0):1, (1,1):-1
        # W: (0,1):1, (1,1):1 (Contributes to C01? No. C10 = M3+M5. C11 = ...+M3.
        #    M3 contributes to C10 (+1) and C11 (+1).
        #    W index (k,i). C10 is i=1, k=0. W_{0,1}. C11 is i=1, k=1. W_{1,1}.
        #    So W: (0,1):1, (1,1):1.
        
        # 4. M4 = A11(B01-B00)
        # U: (1,1):1
        # V: (0,1):1, (0,0):-1
        # W: (0,0):-1, (0,1):1 (Contributes to C00 (-1) and C01 (+1).
        #    C00 is i=0, k=0. W_{0,0}. C01 is i=0, k=1. W_{1,0}.
        #    Wait, W index is (k,i).
        #    C00: i=0, k=0 -> W_{0,0}.
        #    C01: i=0, k=1 -> W_{1,0}.
        #    So W: (0,0):-1, (1,0):1.
        
        # 5. M5 = (A00+A10)B11
        # U: (0,0):1, (1,0):1
        # V: (1,1):1
        # W: (0,0):1, (1,0):1 (Contributes to C00 (+1) and C10 (+1).
        #    C00: i=0, k=0 -> W_{0,0}.
        #    C10: i=1, k=0 -> W_{0,1}.
        #    So W: (0,0):1, (0,1):1.
        
        # 6. M6 = (A10-A00)B10
        # U: (1,0):1, (0,0):-1
        # V: (1,0):1
        # W: (1,1):1 (Contributes to C11 (+1).
        #    C11: i=1, k=1 -> W_{1,1}.
        
        # 7. M7 = (A11-A01)B01
        # U: (1,1):1, (0,1):-1
        # V: (0,1):1
        # W: (0,0):1 (Contributes to C00 (+1).
        #    C00: i=0, k=0 -> W_{0,0}.
        
        # Let's re-verify C formulas:
        # C00 = M1 - M4 + M5 + M7
        #   M1: W_{0,0}=1
        #   M4: W_{0,0}=-1
        #   M5: W_{0,0}=1
        #   M7: W_{0,0}=1
        #   Sum: 1 - (-1) + 1 + 1? No.
        #   M4 W_{0,0} is -1. So - M4 means -(-1) = +1?
        #   The term is sum U*V*W.
        #   M4 contributes -1 * (1) * (-1) = +1 to C00?
        #   M4 U: (1,1):1. V: (0,1):1, (0,0):-1.
        #   For C00 (i=0, k=0), we need U_{0,j}, V_{j,0}.
        #   M4 has U_{1,1}. This doesn't match i=0.
        #   Wait, M4 is A11 * (B01 - B00).
        #   A11 is i=1, j=1.
        #   So M4 only affects rows where i=1?
        #   But C00 is i=0.
        #   So M4 should NOT contribute to C00.
        #   My formula C00 = M1 - M4 + M5 + M7 is WRONG.
        
        # Correct Strassen:
        # C00 = M1 - M4 + M5 + M7
        # C01 = M2 + M4
        # C10 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check indices again.
        # M1 = (A00+A11)(B00+B11).
        #   A00: i=0,j=0. A11: i=1,j=1.
        #   B00: j=0,k=0. B11: j=1,k=1.
        #   M1 terms: A00B00 (i=0,k=0), A00B11 (i=0,k=1? No, B11 is j=1,k=1, A00 is j=0. Mismatch in j).
        #   Strassen works by summing over j implicitly in the product.
        #   M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        #   But A00B11 is not a valid matmul term because j must match.
        #   The "A" and "B" in Strassen are 2x2 blocks.
        #   A = [[A00, A01], [A10, A11]].
        #   B = [[B00, B01], [B10, B11]].
        #   C = A*B = [[A00B00+A01B10, A00B01+A01B11], [A10B00+A11B10, A10B01+A11B11]].
        
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        #   Note: A00B11 is A[0,0]B[1,1]. This is not in C.
        #   However, in the tensor decomposition, we are constructing the identity.
        #   The terms U, V, W must satisfy sum_r U_{i,j,r}V_{j,k,r}W_{k,i,r} = delta_{i,k} ? No.
        #   The target is T_{i,j,k} = 1 for all i,j,k? No.
        #   The target tensor for matmul is T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j}B_{j,k}.
        #   Actually, the tensor is defined as T_{i,j,k} = 1.
        #   And we want sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = C_{i,k} ?
        #   No, the factorization is for the tensor T where T_{i,j,k} = 1.
        #   And the property is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 ?
        #   No, that would mean the product is 1.
        #   The correct identity is:
        #   For any A, B: C = A B.
        #   C_{i,k} = sum_j A_{i,j} B_{j,k}.
        #   We want sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} A_{i,j} B_{j,k} ? No.
        #   The factorization is:
        #   sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 ?
        #   No.
        #   The tensor T has rank R.
        #   T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        #   And we require T_{i,j,k} = 1 for all i,j,k?
        #   No, that's not right.
        #   If T_{i,j,k} = 1, then sum_r ... = 1.
        #   Then for any A, B:
        #   sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} = sum_{i,j,k} A_{i,j} B_{j,k} sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        #   = sum_r (sum_{i,j} A_{i,j} U_{i,j,r}) (sum_{j,k} B_{j,k} V_{j,k,r}) W_{k,i,r} ? No.
        #   The indices don't match for a simple product.
        
        # Let's stick to the previous code's convention which was:
        # U index (i,j), V index (j,k), W index (k,i).
        # And the term is added to U, V, W.
        # The check is: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 ?
        # If this holds, then for any A, B:
        # sum_{i,j,k} A_{i,j} B_{j,k} (sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r})
        # = sum_r W_{k,i,r} (sum_i A_{i,j} U_{i,j,r}) ? No.
        
        # Actually, the standard form is:
        # C_{i,k} = sum_r U_{i,k,r} (sum_j A_{i,j} V_{j,r}) W_{k,j,r} ?
        
        # Let's look at the error message: "tensor identity failed at target coordinate (0, 1, 3)".
        # This implies the evaluator checks a specific identity.
        # Given the previous code structure, it likely checks:
        # sum_r U[row_U, r] * V[row_V, r] * W[row_W, r] == ExpectedValue
        # where row_U, row_V, row_W are derived from i, j, k.
        
        # I will replace the manual Strassen with a verified one.
        
        # Local 2x2x2 Strassen Terms:
        # U, V, W are lists of (index, coeff).
        # Index U: i*m + j (local m=2) -> i*2+j
        # Index V: j*p + k (local p=2) -> j*2+k
        # Index W: k*n + i (local n=2) -> k*2+i
        
        # 1. M1
        add_term([(0*2+0, 1), (1*2+1, 1)],  # U: A00, A11
                 [(0*2+0, 1), (1*2+1, 1)],  # V: B00, B11
                 [(0*2+0, 1), (1*2+1, 1)])  # W: C00, C11
        
        # 2. M2
        add_term([(1*2+0, 1), (1*2+1, 1)],  # U: A10, A11
                 [(0*2+0, 1)],               # V: B00
                 [(1*2+0, 1), (1*2+1, -1)])  # W: C01, C11 (-1)
        
        # 3. M3
        add_term([(0*2+0, 1)],               # U: A00
                 [(1*2+0, 1), (1*2+1, -1)],  # V: B10, B11(-1)
                 [(0*2+1, 1), (1*2+1, 1)])   # W: C10, C11
        
        # 4. M4
        add_term([(1*2+1, 1)],               # U: A11
                 [(0*2+1, 1), (0*2+0, -1)],  # V: B01, B00(-1)
                 [(0*2+0, -1), (1*2+0, 1)])  # W: C00(-1), C01
        
        # 5. M5
        add_term([(0*2+0, 1), (1*2+0, 1)],  # U: A00, A10
                 [(1*2+1, 1)],               # V: B11
                 [(0*2+0, 1), (0*2+1, 1)])   # W: C00, C10
        
        # 6. M6
        add_term([(1*2+0, 1), (0*2+0, -1)],  # U: A10, A00(-1)
                 [(1*2+0, 1)],               # V: B10
                 [(1*2+1, 1)])               # W: C11
        
        # 7. M7
        add_term([(1*2+1, 1), (0*2+1, -1)],  # U: A11, A01(-1)
                 [(0*2+1, 1)],               # V: B01
                 [(0*2+0, 1)])               # W: C00
        
        # Apply offsets
        for u, v, w in terms:
            pass # This is inside add_term, so we need to offset before appending.
            
        # I will rewrite the add_term calls to include offsets.
        
    def add_strassen_offset(n_start, m_start, p_start):
        # Offset for U: n_start*m + m_start
        # Offset for V: m_start*p + p_start
        # Offset for W: p_start*n + n_start
        
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        
        # 1. M1
        terms.append(([(u_off + 0, 1), (u_off + 3, 1)],
                      [(v_off + 0, 1), (v_off + 3, 1)],
                      [(w_off + 0, 1), (w_off + 3, 1)]))
        # 2. M2
        terms.append(([(u_off + 2, 1), (u_off + 3, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 2, 1), (w_off + 3, -1)]))
        # 3. M3
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 2, 1), (v_off + 3, -1)],
                      [(w_off + 1, 1), (w_off + 3, 1)]))
        # 4. M4
        terms.append(([(u_off + 3, 1)],
                      [(v_off + 1, 1), (v_off + 0, -1)],
                      [(w_off + 0, -1), (w_off + 2, 1)]))
        # 5. M5
        terms.append(([(u_off + 0, 1), (u_off + 2, 1)],
                      [(v_off + 3, 1)],
                      [(w_off + 0, 1), (w_off + 1, 1)]))
        # 6. M6
        terms.append(([(u_off + 2, 1), (u_off + 0, -1)],
                      [(v_off + 2, 1)],
                      [(w_off + 3, 1)]))
        # 7. M7
        terms.append(([(u_off + 3, 1), (u_off + 1, -1)],
                      [(v_off + 1, 1)],
                      [(w_off + 0, 1)]))

    def add_schoolbook_2x2x1(n_start, m_start, p_start):
        # 2x2 A, 2x1 B, 2x1 C
        # C0 = A00B0 + A01B1
        # C1 = A10B0 + A11B1
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        
        # Term 1: A00 B0
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 0, 1)]))
        # Term 2: A01 B1
        terms.append(([(u_off + 1, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 0, 1)]))
        # Term 3: A10 B0
        terms.append(([(u_off + 2, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 1, 1)]))
        # Term 4: A11 B1
        terms.append(([(u_off + 3, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 1, 1)]))

    def add_schoolbook_2x1x2(n_start, m_start, p_start):
        # 2x1 A, 1x2 B, 2x2 C
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        
        # Term 1: A0 B0
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 0, 1)]))
        # Term 2: A0 B1
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 2, 1)]))
        # Term 3: A1 B0
        terms.append(([(u_off + 1, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 1, 1)]))
        # Term 4: A1 B1
        terms.append(([(u_off + 1, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 3, 1)]))

    def add_schoolbook_1x2x2(n_start, m_start, p_start):
        # 1x2 A, 2x2 B, 1x2 C
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        
        # Term 1: A0 B00
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 0, 1)]))
        # Term 2: A1 B10
        terms.append(([(u_off + 1, 1)],
                      [(v_off + 2, 1)],
                      [(w_off + 0, 1)]))
        # Term 3: A0 B01
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 1, 1)]))
        # Term 4: A1 B11
        terms.append(([(u_off + 1, 1)],
                      [(v_off + 3, 1)],
                      [(w_off + 1, 1)]))

    def add_schoolbook_1x1x2(n_start, m_start, p_start):
        # 1x1 A, 1x2 B, 1x2 C
        # C0 = A0 B0
        # C1 = A0 B1
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        
        # Term 1: A0 B0
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 0, 1)]))
        # Term 2: A0 B1
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 1, 1)],
                      [(w_off + 1, 1)]))

    def add_schoolbook_1x1x1(n_start, m_start, p_start):
        u_off = n_start * m + m_start
        v_off = m_start * p + p_start
        w_off = p_start * n + n_start
        terms.append(([(u_off + 0, 1)],
                      [(v_off + 0, 1)],
                      [(w_off + 0, 1)]))

    n_block_ranges = [(0, 2), (2, 3)]
    m_block_ranges = [(0, 2), (2, 4), (4, 5)]
    p_block_ranges = [(0, 2), (2, 4), (4, 6)]
    
    for ni, (ns, ne) in enumerate(n_block_ranges):
        for mi, (ms, me) in enumerate(m_block_ranges):
            for pi, (ps, pe) in enumerate(p_block_ranges):
                n_size = ne - ns
                m_size = me - ms
                p_size = pe - ps
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen_offset(ns, ms, ps)
                elif n_size == 2 and m_size == 2 and p_size == 1:
                    add_schoolbook_2x2x1(ns, ms, ps)
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    add_schoolbook_2x1x2(ns, ms, ps)
                elif n_size == 1 and m_size == 2 and p_size == 2:
                    add_schoolbook_1x2x2(ns, ms, ps)
                elif n_size == 1 and m_size == 1 and p_size == 2:
                    add_schoolbook_1x1x2(ns, ms, ps)
                elif n_size == 1 and m_size == 1 and p_size == 1:
                    add_schoolbook_1x1x1(ns, ms, ps)
                else:
                    raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
    
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
