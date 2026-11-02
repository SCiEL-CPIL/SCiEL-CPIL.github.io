"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a decomposition of rank 40.
    # Strategy: Split n=3 into 2+1.
    # M_{3,4,5} = M_{2,4,5} (top 2 rows) + M_{1,4,5} (bottom 1 row)
    
    # Part 1: M_{1,4,5} (Rank 4)
    # C[2, k] = sum(j=0..3) A[2, j] B[j, k]
    # U[2, r] = 1 if r corresponds to this part
    # V[j, r] = delta(j, r)
    # W[k, r] = 1? No.
    # C[2, k] = sum_r U[2, r] W[k, r] M_r
    # M_r = A[2, r] B[r, k]
    # So U[2, r] = 1, W[k, r] = 1?
    # If W[k, r] = 1, then C[2, k] = sum_r 1 * 1 * A[2, r] B[r, k]. Correct.
    # But W is shared? No, W is a matrix of shape (p*n, rank).
    # Index for W is k*n + i.
    # For i=2 (0-indexed), k=0..4.
    # W[k*3 + 2, r] = 1.
    
    rank_part1 = 4
    # Indices for U: 0..11 (n*m). i=2 -> 2*4 + j = 8+j.
    # Indices for V: 0..19 (m*p). j*p + k.
    # Indices for W: 0..14 (p*n). k*n + i.
    
    # Part 2: M_{2,4,5} (Rank 36)
    # Split m=4 into 2+2.
    # M_{2,4,5} = M_{2,2,5} (cols 0,1 of A, rows 0,1 of B) + M_{2,2,5} (cols 2,3 of A, rows 2,3 of B)
    
    # Sub-part 2a: M_{2,2,5} for A[:, 0:2], B[0:2, :]
    # Split p=5 into 2+2+1.
    # M_{2,2,5} = M_{2,2,2} (B cols 0,1) + M_{2,2,2} (B cols 2,3) + M_{2,2,1} (B col 4)
    
    # M_{2,2,2} Rank 7 (Strassen)
    # M_{2,2,1} Rank 4
    
    # Total Rank = 4 (Part 1) + 2 * (7 + 7 + 4) = 4 + 36 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 block
    # Maps A_sub (2x2), B_sub (2x2) to C_sub (2x2)
    # Strassen constants:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen_222(offset, a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start):
        # a_i_start: row offset in A (0..1)
        # a_j_start: col offset in A (0..1)
        # b_j_start: row offset in B (0..1)
        # b_k_start: col offset in B (0..1)
        # c_i_start: row offset in C (0..1)
        # c_k_start: col offset in C (0..1)
        
        # Indices in U, V, W
        # U index: (a_i_start + i) * m + (a_j_start + j)
        # V index: (b_j_start + j) * p + (b_k_start + k)
        # W index: (c_k_start + k) * n + (c_i_start + i)
        
        # 7 terms
        terms = [
            # (A_idx, B_idx, C_idx, coeff)
            # M1
            [(0,0,0,1), (1,1,0,1), (0,0,0,1), (0,0,1,1), (1,1,0,1), (1,1,1,1), (0,1,0,1), (1,0,1,1)], # This is complex
        ]
        # Let's just hardcode the 7 rank-1 terms for 2x2x2
        # T1: (A11+A22)(B11+B22)
        # U: A11=1, A22=1. V: B11=1, B22=1. W: C11=1, C12=1, C21=1, C22=1? No.
        # C11 += M1, C12 += 0, C21 += 0, C22 -= M1? No.
        
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add these to U, V, W.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs, coeff):
            for (ai, aj, ui), (bj, bk, vj), (ck, ci, wk) in zip(u_coeffs, v_coeffs, w_coeffs):
                u_idx = (a_i_start + ai) * m + (a_j_start + aj)
                v_idx = (b_j_start + bj) * p + (b_k_start + bk)
                w_idx = (c_k_start + ck) * n + (c_i_start + ci)
                U[u_idx][offset] += coeff * ui
                V[v_idx][offset] += coeff * vj
                W[w_idx][offset] += coeff * wk

        # M1
        add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)], 1)
        # Wait, W coefficients for M1:
        # C11 += 1, C12 += 0, C21 += 0, C22 -= 1? No, C22 = M1 - M2 + M3 + M6. So C22 += 1.
        # C11 = M1 + ... So C11 += 1.
        # So W for M1: C11=1, C22=1.
        # My add_term above adds to all listed indices.
        
        # Let's rewrite add_term to take lists of (index, coeff)
        def add_rank1(a_coeffs, b_coeffs, c_coeffs):
            # a_coeffs: list of (ai, aj, val)
            # b_coeffs: list of (bj, bk, val)
            # c_coeffs: list of (ci, ck, val)
            for (ai, aj, va) in a_coeffs:
                for (bj, bk, vb) in b_coeffs:
                    for (ci, ck, vc) in c_coeffs:
                        u_idx = (a_i_start + ai) * m + (a_j_start + aj)
                        v_idx = (b_j_start + bj) * p + (b_k_start + bk)
                        w_idx = (c_k_start + ck) * n + (c_i_start + ci)
                        U[u_idx][offset] += va
                        V[v_idx][offset] += vb
                        W[w_idx][offset] += vc

        # This is O(N^3) per term, which is fine for small numbers.
        
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        add_rank1([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        
        # M2: (A21+A22)B11 -> C21 += 1, C22 -= 1
        add_rank1([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])
        
        # M3: A11(B21-B22) -> C12 += 1, C22 += 1
        add_rank1([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)])
        
        # M4: A22(B11-B21) -> C11 += 1, C21 += 1
        add_rank1([(1,1,1)], [(0,0,1), (1,0,-1)], [(0,0,1), (1,0,1)])
        
        # M5: (A11+A12)B22 -> C11 -= 1, C12 += 1
        add_rank1([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])
        
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        add_rank1([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        add_rank1([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])

    # Helper to add 2x2x1 block (Rank 4)
    def add_221(offset, a_i_start, a_j_start, b_j_start, b_k_start, c_i_start, c_k_start):
        # C10 = A10 B00 + A11 B10
        # C20 = A20 B00 + A21 B10
        # Rank 4:
        # M1 = A10 B00 -> C10 += 1
        # M2 = A11 B10 -> C10 += 1
        # M3 = A20 B00 -> C20 += 1
        # M4 = A21 B10 -> C20 += 1
        
        def add_rank1(a_coeffs, b_coeffs, c_coeffs):
            for (ai, aj, va) in a_coeffs:
                for (bj, bk, vb) in b_coeffs:
                    for (ci, ck, vc) in c_coeffs:
                        u_idx = (a_i_start + ai) * m + (a_j_start + aj)
                        v_idx = (b_j_start + bj) * p + (b_k_start + bk)
                        w_idx = (c_k_start + ck) * n + (c_i_start + ci)
                        U[u_idx][offset] += va
                        V[v_idx][offset] += vb
                        W[w_idx][offset] += vc

        add_rank1([(0,0,1)], [(0,0,1)], [(0,0,1)])
        add_rank1([(0,1,1)], [(1,0,1)], [(0,0,1)])
        add_rank1([(1,0,1)], [(0,0,1)], [(1,0,1)])
        add_rank1([(1,1,1)], [(1,0,1)], [(1,0,1)])

    # Assemble
    offset = 0
    
    # Part 1: M_{1,4,5} (i=2)
    # C[2, k] = sum_j A[2, j] B[j, k]
    # Rank 4
    for j in range(4):
        U[2 * 4 + j][offset] = 1
        V[j * 5 + 0][offset] = 1 # B[j, 0]? No, B is 4x5.
        # V index: j*p + k.
        # We need B[j, k].
        # So V[j*5 + k][offset] = 1 for all k?
        # If V[j*5 + k] = 1, then M = sum_j A[2,j] (sum_k B[j,k])? No.
        # M_r = sum_j V[j,r] A[2,j] B[j,k]?
        # If V[j,r] = delta(j,r), then M_r = A[2,r] B[r,k].
        # Then C[2,k] = sum_r U[2,r] W[k,r] M_r.
        # U[2,r] = 1. W[k,r] = 1.
        # C[2,k] = sum_r A[2,r] B[r,k]. Correct.
        
        V[j * 5 + 0][offset] = 1 # This is wrong. V depends on k?
        # No, V is a matrix. V[j, r] is the coefficient for A[2,j] in M_r.
        # But M_r also depends on B.
        # The term is U[i,r] V[j,r] W[k,r] A[i,j] B[j,k].
        # If we set V[j,r] = delta(j,r), then the term is U[i,r] W[k,r] A[i,r] B[r,k].
        # Summing over r: sum_r U[i,r] W[k,r] A[i,r] B[r,k].
        # If U[i,r]=1, W[k,r]=1, then sum_r A[i,r] B[r,k].
        # So we need V[j, r] = 1 if j==r, else 0.
        # And U[i, r] = 1.
        # And W[k, r] = 1.
        
        U[2 * 4 + j][offset] = 1
        V[j * 5 + 0][offset] = 1 # j index in V. k index in V?
        # V has shape (m*p, rank). Index is j*p + k.
        # We need to pick a specific k? No.
        # The term is V[j, r] B[j, k].
        # If V[j, r] is non-zero, it multiplies B[j, k].
        # But B[j, k] varies with k.
        # So V[j, r] must be such that the sum works out.
        # Actually, the rank 1 term $u \otimes v \otimes w$ in $T_{i,j,k}$
        # corresponds to $C_{i,k} += u_i w_k v_j A_{i,j} B_{j,k}$.
        # So for Part 1, we want $C_{2,k} += A_{2,j} B_{j,k}$.
        # So $u_2=1, w_k=1, v_j=1$.
        # This is one rank 1 term?
        # $T_{2,j,k} = 1$.
        # $u \otimes v \otimes w$ with $u_2=1, v_j=1, w_k=1$.
        # This is rank 1!
        # So we only need 1 term for M_{1,4,5}?
        # Let's check.
        # $C_{2,k} = \sum_r U_{2,r} W_{k,r} (\sum_j V_{j,r} A_{2,j} B_{j,k})$.
        # If $R=1$, $U_{2,1}=1, W_{k,1}=1, V_{j,1}=1$.
        # $C_{2,k} = 1 \cdot 1 \cdot (\sum_j 1 \cdot A_{2,j} B_{j,k}) = \sum_j A_{2,j} B_{j,k}$.
        # Yes! Rank is 1.
        
    # So Part 1 Rank is 1.
    # Update total rank: 1 + 36 = 37.
    
    # Re-initialize with rank 37
    rank = 37
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Offset 0
    U[2 * 4 + 0][0] = 1 # U[2, r] = 1. But U has shape (n*m, rank).
    # U index: i*m + j.
    # We need U[2, 0] = 1?
    # $C_{i,k} = \sum_r U_{i,r} W_{k,r} (\sum_j V_{j,r} A_{i,j} B_{j,k})$.
    # Here $i=2$.
    # $U_{2,0} = 1$.
    # $W_{k,0} = 1$ for all k.
    # $V_{j,0} = 1$ for all j.
    
    U[2 * 4][0] = 1 # i=2, j=0? No, U index is i*m+j.
    # The formula uses $U_{i,r}$. But U is stored as $U[i*m+j, r]$.
    # This implies $U_{i,r}$ is not well-defined unless $j$ is fixed?
    # No, the tensor is $T_{i,j,k}$.
    # $U$ has indices $(i,j)$. $V$ has $(j,k)$. $W$ has $(k,i)$.
    # $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
    # So for Part 1 ($i=2$):
    # We want $T_{2,j,k} = 1$.
    # Set $r=0$.
    # $U_{2,j,0} = 1$ for all j.
    # $V_{j,k,0} = 1$ for all j,k?
    # $W_{k,2,0} = 1$ for all k.
    # Then $T_{2,j,k} = \sum_r U_{2,j,r} V_{j,k,r} W_{k,2,r} = 1 \cdot 1 \cdot 1 = 1$.
    # Correct.
    
    for j in range(4):
        U[2 * 4 + j][0] = 1
    for j in range(4):
        for k in range(5):
            V[j * 5 + k][0] = 1
    for k in range(5):
        W[k * 3 + 2][0] = 1
        
    offset = 1
    
    # Part 2: M_{2,4,5} (i=0,1)
    # Split j=0..3 into 0,1 and 2,3.
    
    # Block 2a: A[0:2, 0:2], B[0:2, :]
    # Split k=0..4 into 0,1; 2,3; 4.
    
    # 2a1: M_{2,2,2} (k=0,1)
    add_strassen_222(offset, 0, 0, 0, 0, 0, 0)
    offset += 7
    
    # 2a2: M_{2,2,2} (k=2,3)
    add_strassen_222(offset, 0, 0, 0, 2, 0, 2)
    offset += 7
    
    # 2a3: M_{2,2,1} (k=4)
    add_221(offset, 0, 0, 0, 4, 0, 4)
    offset += 4
    
    # Block 2b: A[0:2, 2:4], B[2:4, :]
    # 2b1: M_{2,2,2} (k=0,1)
    add_strassen_222(offset, 0, 2, 2, 0, 0, 0)
    offset += 7
    
    # 2b2: M_{2,2,2} (k=2,3)
    add_strassen_222(offset, 0, 2, 2, 2, 0, 2)
    offset += 7
    
    # 2b3: M_{2,2,1} (k=4)
    add_221(offset, 0, 2, 2, 4, 0, 4)
    offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
