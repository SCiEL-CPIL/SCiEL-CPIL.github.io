import time
import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a rank-1 term
    # u_idx: (i, j) in A
    # v_idx: (j, k) in B
    # w_idx: (k, i) in C
    # coeff: value
    def add_term(u_idx, v_idx, w_idx, coeff, r):
        i, j = u_idx
        j_v, k = v_idx
        k_w, i_w = w_idx
        # Check consistency
        if i != i_w or j != j_v:
             # This shouldn't happen if we construct correctly
             pass
        U[i * m + j, r] += coeff * 1.0 # U holds A coeff
        V[j_v * p + k, r] += 1.0       # V holds B coeff? 
        # Wait. The tensor is sum U * V * W.
        # If we want A_ij * B_jk * C_ki, we set U=1, V=1, W=1.
        # If we want (A_00 + A_11) * (B_00 + B_11) * C_00,
        # we can split it.
        # Actually, the standard decomposition defines U, V, W such that
        # C_ik = sum_r U_ikr V_jkr W_jir? No.
        # The convention is:
        # C_ik = sum_j sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} A_{i,j} B_{j,k}
        # No, the variables A, B are inputs.
        # The tensor T is such that C = T(A,B).
        # T_{i,j,k} = 1 if it's a valid multiplication term.
        # Decomposition: T = sum_r U^{(r)} \otimes V^{(r)} \otimes W^{(r)}.
        # So C_{i,k} = sum_j sum_r U^{(r)}_{i,j} V^{(r)}_{j,k} W^{(r)}_{k,i} A_{i,j} B_{j,k} ??
        # No, A and B are contracted with U and V.
        # C_{i,k} = sum_r (sum_j U^{(r)}_{i,j} A_{i,j}) (sum_l V^{(r)}_{l,k} B_{l,k}) W^{(r)}_{k,i} ?
        # No, that's not right either.
        
        # Standard definition:
        # C_{i,k} = \sum_{j} A_{i,j} B_{j,k}
        # Decomposition:
        # C_{i,k} = \sum_{r} U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ?
        # The indices must match.
        # Usually: C_{i,k} = \sum_{r} \alpha_r ( \sum_j U_{i,j}^{(r)} A_{i,j} ) ( \sum_l V_{l,k}^{(r)} B_{l,k} ) W_{k,i}^{(r)} ?
        # No, the rank-1 tensor is $u \otimes v \otimes w$.
        # $(u \otimes v \otimes w)_{i,j,k} = u_i v_j w_k$.
        # The multiplication tensor is $T_{i,j,k} = \delta_{j, \text{mid}} \delta_{i, \text{out}} \delta_{k, \text{out}}$?
        # $T_{i,j,k}$ is non-zero when $A_{i,j}$ and $B_{j,k}$ multiply to give $C_{i,k}$.
        # So $T_{i,j,k} = 1$ for all $i,j,k$? No.
        # $C_{i,k}$ is sum over $j$.
        # So $T$ has components $T_{i,j,k}$.
        # If we fix $i,k$, the term is $\sum_j A_{i,j} B_{j,k}$.
        # So the tensor is $\sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}$.
        # The coefficient of $A_{i,j} B_{j,k} C_{i,k}$ is 1.
        # So $T_{i,j,k} = 1$ if the indices are valid?
        # Yes, for all $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$, the term $A_{i,j} B_{j,k} C_{i,k}$ appears with coeff 1.
        # Wait, $C_{i,k}$ is the output.
        # So we want $\sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} = 1$ for all $i,j,k$?
        # No, $W$ is indexed by output $i,k$.
        # The prompt says: `W[k*n+i, r]`. So $W$ is indexed by $k, i$.
        # And `U[i*m+j, r]`, `V[j*p+k, r]`.
        # The check is:
        # `s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))`
        # where `a = i*m+j`, `b = j*p+k`, `c = k*n+i`.
        # And it checks if `s == 1` if `j_a == j_b and k_b == k_c and i_a == i_c`.
        # `j_a` is the column of A. `j_b` is the row of B. They must match.
        # `k_b` is the column of B. `k_c` is the column of C. They must match.
        # `i_a` is the row of A. `i_c` is the row of C. They must match.
        # So yes, for every valid $(i,j,k)$, the sum must be 1.
        # For invalid, sum must be 0.
        
        # So we just need to populate U, V, W such that this holds.
        pass

    # We will manually construct the 39 terms.
    
    # Part 2: M_{2,1,7} (Rank 14)
    # A col 2, B row 2.
    for i in range(2):
        for k in range(7):
            r = 25 + i * 7 + k
            U[i * 3 + 2, r] = 1.0
            V[2 * 7 + k, r] = 1.0
            W[k * 2 + i, r] = 1.0
            
    # Part 1: M_{2,2,7} (Rank 25)
    # Blocks of k: [0,1], [2,3], [4,5], [6]
    
    # Strassen coefficients for M_{2,2,2}
    # U, V, W for each of the 7 terms.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # U: (0,0):1, (1,1):1
    # V: (0,0):1, (1,1):1
    # W: (0,0):1, (1,1):1
    
    # Term 2: (A10+A11)B00 -> C10
    # U: (1,0):1, (1,1):1
    # V: (0,0):1
    # W: (0,1):1  (k=0, i=1)
    
    # Term 3: A00(B01-B11) -> C01
    # U: (0,0):1
    # V: (0,1):1, (1,1):-1
    # W: (1,0):1  (k=1, i=0)
    
    # Term 4: A11(B10-B00) -> C10
    # U: (1,1):1
    # V: (1,0):1, (0,0):-1
    # W: (0,1):1  (k=0, i=1)
    
    # Term 5: (A00+A01)B11 -> C01, C00?
    # M5 = (A00+A01)B11
    # C01 += M5, C00 -= M5
    # U: (0,0):1, (0,1):1
    # V: (1,1):1
    # W: (1,0):1, (0,0):-1
    
    # Term 6: (A10-A00)(B00+B01) -> C11
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    # U: (1,0):1, (0,0):-1
    # V: (0,0):1, (0,1):1
    # W: (1,1):1
    
    # Term 7: (A01-A11)(B10+B11) -> C00, C11
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7, C11 -= M7?
    # Formula: C11 = M1 - M2 + M3 + M6.
    # Formula: C00 = M1 + M4 - M5 + M7.
    # So C00 gets +M7.
    # C11 gets -M7?
    # Let's re-verify Strassen.
    # C11 = M1 - M2 + M3 + M6.
    # Where is M7?
    # Ah, standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # Term 1: W(0,0)=1, W(1,1)=1
    # Term 2: W(0,1)=1 (k=0, i=1)
    # Term 3: W(1,0)=1 (k=1, i=0)
    # Term 4: W(0,1)=1 (k=0, i=1)
    # Term 5: W(1,0)=1, W(0,0)=-1
    # Term 6: W(1,1)=1 (k=1, i=1)
    # Term 7: W(0,0)=1, W(1,1)=-1

    def add_strassen_block(k_start, rank_start):
        # k_start is 0, 2, 4
        # rank_start is 0, 7, 14
        
        # Term 1
        r = rank_start
        U[0*3+0, r] = 1.0; U[1*3+1, r] = 1.0
        V[0*7+k_start, r] = 1.0; V[1*7+k_start+1, r] = 1.0
        W[k_start*2+0, r] = 1.0; W[(k_start+1)*2+1, r] = 1.0
        
        # Term 2
        r = rank_start + 1
        U[1*3+0, r] = 1.0; U[1*3+1, r] = 1.0
        V[0*7+k_start, r] = 1.0
        W[k_start*2+1, r] = 1.0
        
        # Term 3
        r = rank_start + 2
        U[0*3+0, r] = 1.0
        V[0*7+k_start+1, r] = 1.0; V[1*7+k_start+1, r] = -1.0
        W[(k_start+1)*2+0, r] = 1.0
        
        # Term 4
        r = rank_start + 3
        U[1*3+1, r] = 1.0
        V[1*7+k_start, r] = 1.0; V[0*7+k_start, r] = -1.0
        W[k_start*2+1, r] = 1.0
        
        # Term 5
        r = rank_start + 4
        U[0*3+0, r] = 1.0; U[0*3+1, r] = 1.0
        V[1*7+k_start+1, r] = 1.0
        W[(k_start+1)*2+0, r] = 1.0; W[k_start*2+0, r] = -1.0
        
        # Term 6
        r = rank_start + 5
        U[1*3+0, r] = 1.0; U[0*3+0, r] = -1.0
        V[0*7+k_start, r] = 1.0; V[0*7+k_start+1, r] = 1.0
        W[(k_start+1)*2+1, r] = 1.0
        
        # Term 7
        r = rank_start + 6
        U[0*3+1, r] = 1.0; U[1*3+1, r] = -1.0
        V[1*7+k_start, r] = 1.0; V[1*7+k_start+1, r] = 1.0
        W[k_start*2+0, r] = 1.0; W[(k_start+1)*2+1, r] = -1.0

    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)
    
    # Block 4: k=6, Rank 4 (21-24)
    # M_{2,2,1}
    # C_i,6 = A_i,0 B_0,6 + A_i,1 B_1,6
    
    # Term 1: A_0,0 B_0,6 -> C_0,6
    r = 21
    U[0*3+0, r] = 1.0
    V[0*7+6, r] = 1.0
    W[6*2+0, r] = 1.0
    
    # Term 2: A_1,0 B_0,6 -> C_1,6
    r = 22
    U[1*3+0, r] = 1.0
    V[0*7+6, r] = 1.0
    W[6*2+1, r] = 1.0
    
    # Term 3: A_0,1 B_1,6 -> C_0,6
    r = 23
    U[0*3+1, r] = 1.0
    V[1*7+6, r] = 1.0
    W[6*2+0, r] = 1.0
    
    # Term 4: A_1,1 B_1,6 -> C_1,6
    r = 24
    U[1*3+1, r] = 1.0
    V[1*7+6, r] = 1.0
    W[6*2+1, r] = 1.0
    
    return {"U": U, "V": V, "W": W}
