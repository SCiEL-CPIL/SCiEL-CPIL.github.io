"""Program to find a rank-minimal exact tensor decomposition for 3x3 by 3x8 matrix multiplication."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# Known Strassen formulas for 2x2x2 (indices 0..7)
# A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
# M1 = (A00 + A11)(B00 + B11)
# M2 = (A10 + A11)B00
# M3 = A00(B10 - B11)
# M4 = A11(B01 - B00)
# M5 = (A00 + A10)B11
# M6 = (A11 - A00)(B00 + B01)
# M7 = (A01 - A11)(B10 + B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 Strassen.
    U shape (4, 7): rows (i,j) -> i*2+j
    V shape (4, 7): rows (j,k) -> j*2+k
    W shape (4, 7): rows (k,i) -> k*2+i
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1: (A00 + A11)(B00 + B11)
    # A00 = (0,0) -> U[0] += 1
    # A11 = (1,1) -> U[3] += 1
    # B00 = (0,0) -> V[0] += 1
    # B11 = (1,1) -> V[3] += 1
    # Output:
    # C00: M1 -> W[0*2+0] = W[0] += 1
    # C11: M1 -> W[1*2+1] = W[3] += 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10 + A11)B00
    # A10 = (1,0) -> U[2] += 1
    # A11 = (1,1) -> U[3] += 1
    # B00 = (0,0) -> V[0] += 1
    # Output:
    # C10: M2 -> W[0*2+1] = W[1] += 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00(B10 - B11)
    # A00 = (0,0) -> U[0] += 1
    # B10 = (1,0) -> V[2] += 1
    # B11 = (1,1) -> V[3] -= 1
    # Output:
    # C01: M3 -> W[0*2+0 + 1] = W[1]? No, W[k*n+i]. n=2.
    # C01: k=0, i=0? No, C01 is row 0, col 1.
    # W[k*n+i]. For C01 (k=0, i=0? No. C is 2x2. C01 is k=0? No.
    # C[k, i] corresponds to W[k*n + i].
    # C00: k=0, i=0 -> W[0]
    # C01: k=0, i=1? No. C is [ [C00, C01], [C10, C11] ].
    # Row index of C is k? Col index of C is i?
    # Standard: C = AB. C_{ki} = sum_j A_{ij} B_{jk}.
    # So C_{ki} is row k, col i.
    # W index: k*n + i.
    # C00: k=0, i=0 -> W[0]
    # C01: k=0, i=1 -> W[1]
    # C10: k=1, i=0 -> W[2]
    # C11: k=1, i=1 -> W[3]
    
    # Correction on W indices:
    # C00 -> W[0]
    # C01 -> W[1]
    # C10 -> W[2]
    # C11 -> W[3]
    
    # Re-evaluating M3 output:
    # C01 = M3 + M5. So M3 contributes to W[1].
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4: A11(B01 - B00)
    # A11 = (1,1) -> U[3] += 1
    # B01 = (0,1) -> V[1] += 1
    # B00 = (0,0) -> V[0] -= 1
    # Output:
    # C00 = ... + M4 -> W[0] += 1
    # C10 = M2 + M4 -> W[2] += 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00 + A10)B11
    # A00 = (0,0) -> U[0] += 1
    # A10 = (1,0) -> U[2] += 1
    # B11 = (1,1) -> V[3] += 1
    # Output:
    # C00 = ... - M5 -> W[0] -= 1
    # C01 = M3 + M5 -> W[1] += 1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A11 - A00)(B00 + B01)
    # A11 = (1,1) -> U[3] += 1
    # A00 = (0,0) -> U[0] -= 1
    # B00 = (0,0) -> V[0] += 1
    # B01 = (0,1) -> V[1] += 1
    # Output:
    # C11 = ... + M6 -> W[3] += 1
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01 - A11)(B10 + B11)
    # A01 = (0,1) -> U[1] += 1
    # A11 = (1,1) -> U[3] -= 1
    # B10 = (1,0) -> V[2] += 1
    # B11 = (1,1) -> V[3] += 1
    # Output:
    # C00 = ... + M7 -> W[0] += 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify if U, V, W form a valid tensor decomposition for 3x3 * 3x8."""
    n, m, p = 3, 3, 8
    rank = len(U[0])
    # T[i, j, k] should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i, j, k] = 1 if (i, j, k) is a valid index, else 0?
    # Wait, the tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
    # is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has indices (i, j, k) and value T_{ijk} = 1 if the term A_{ij}B_{jk} contributes to C_{ik}.
    # So T_{ijk} = 1 for all valid i, j, k.
    # Let's check: sum_j A_{ij} B_{jk} = sum_j (sum_r U_{ij,r} V_{jk,r} W_{ik,r})? No.
    # The decomposition is: sum_r U_{ij,r} V_{jk,r} W_{ik,r} = delta_{...}?
    # Actually, the tensor T is the structure constant of the algebra.
    # T_{ijk} = 1 if the multiplication of basis elements e_i e_j = e_k? No.
    # For matrix multiplication, the tensor is T_{i, j, k} = 1.
    # Because C_{ik} = sum_j A_{ij} B_{jk}.
    # Let A_{ij} = sum_r u_{ij,r} lambda_{ij,r}? No.
    # The factorization is:
    # M_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    # where M_{ijk} is the entry of the multiplication tensor.
    # For matrix multiplication, M_{ijk} = 1 for all i, j, k.
    # Let's verify this.
    # If U, V, W are such that sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for all i,j,k,
    # then for any A, B:
    # sum_j A_{ij} B_{jk} = sum_j A_{ij} B_{jk} (sum_r U_{ij,r} V_{jk,r} W_{ik,r} / 1) ?
    # No, the identity is:
    # sum_j A_{ij} B_{jk} = sum_r (sum_j A_{ij} U_{ij,r}) (sum_k B_{jk} V_{jk,r}) W_{ik,r}?
    # This looks like:
    # C_{ik} = sum_r (sum_j A_{ij} U_{ij,r}) (sum_j B_{jk} V_{jk,r}) W_{ik,r}
    # This is not standard.
    
    # Standard tensor decomposition of matrix multiplication:
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where T_{ijk} = 1.
    # Then u_r(i,j) = U_{ij,r}, v_r(j,k) = V_{jk,r}, w_r(i,k) = W_{ik,r}.
    # So we need sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1 for all i, j, k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 3, 3, 8
    # Target: Reduce rank from 72.
    # Strategy: Use Strassen for the top-left 2x2 block of A (indices 0,1) and corresponding 2x8 block of B?
    # B is 3x8. The 2x2 block of A corresponds to j in {0,1}.
    # So B rows 0 and 1 are 2x8 blocks.
    # We can split the 8 columns of B into four 2-column blocks.
    # For each 2x2x2 sub-problem (A 2x2, B 2x2, C 2x2), we use Strassen (rank 7 instead of 8).
    # Total rank:
    # 4 blocks of 2x2x2 Strassen: 4 * 7 = 28.
    # Remaining A rows/cols:
    # A has indices (0,1) x (0,1) covered.
    # Remaining A: (0,1) x (2), (2) x (0,1), (2) x (2).
    # Let's handle the rest with schoolbook.
    # A_{0,2} B_{2,8} -> C_{0,8} (2x8 output for i=0,1? No, i=0,1 is 2 rows. j=2 is 1 row. k=0..7 is 8 cols.
    # So A (2x1) * B (1x8) = C (2x8). Rank = 2*1*8 = 16.
    # A_{2,0} B_{0,8} -> C_{2,8} (1x8 output for i=2? No, i=2 is 1 row. j=0,1 is 2 rows. k=0..7 is 8 cols.
    # So A (1x2) * B (2x8) = C (1x8). Rank = 1*2*8 = 16.
    # A_{2,2} B_{2,8} -> C_{2,8} (1x8 output for i=2. j=2 is 1 row. k=0..7 is 8 cols.
    # So A (1x1) * B (1x8) = C (1x8). Rank = 1*1*8 = 8.
    # Total rank = 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col_offset = 0
    
    # 1. Strassen blocks for i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for k_group in range(4):
        k_start = k_group * 2
        k0 = k_start
        k1 = k_start + 1
        
        # For this block, we map local indices to global.
        # Local i, j, k in {0,1}.
        # Global i, j, k:
        # i_global = i_local
        # j_global = j_local
        # k_global = k_start + k_local
        
        for i_l in range(2):
            for j_l in range(2):
                idx_U = i_l * 2 + j_l
                idx_U_global = i_l * 3 + j_l # i*m + j
                
                for r_l in range(7):
                    U[idx_U_global][col_offset + r_l] = U_str[idx_U][r_l]
        
        for j_l in range(2):
            for k_l in range(2):
                idx_V = j_l * 2 + k_l
                idx_V_global = j_l * 3 + (k_start + k_l) # j*p + k
                
                for r_l in range(7):
                    V[idx_V_global][col_offset + r_l] = V_str[idx_V][r_l]
        
        for k_l in range(2):
            for i_l in range(2):
                idx_W = k_l * 2 + i_l
                idx_W_global = (k_start + k_l) * 3 + i_l # k*n + i
                
                for r_l in range(7):
                    W[idx_W_global][col_offset + r_l] = W_str[idx_W][r_l]
        
        col_offset += 7
    
    # 2. A_{0,1} (i in 0,1) x A_{2} (j=2) * B_{2} (j=2) x B_{0,7} (k in 0..7)
    # This is i in {0,1}, j=2, k in {0..7}.
    # Rank = 2 * 1 * 8 = 16.
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col_offset] = 1
            V[2 * 3 + k][col_offset] = 1
            W[k * 3 + i][col_offset] = 1
            col_offset += 1
    
    # 3. A_{2} (i=2) x A_{0,1} (j in 0,1) * B_{0,1} (j in 0,1) x B_{0,7} (k in 0..7)
    # This is i=2, j in {0,1}, k in {0..7}.
    # Rank = 1 * 2 * 8 = 16.
    for j in range(2):
        for k in range(8):
            U[2 * 3 + j][col_offset] = 1
            V[j * 3 + k][col_offset] = 1
            W[k * 3 + 2][col_offset] = 1
            col_offset += 1
    
    # 4. A_{2} (i=2) x A_{2} (j=2) * B_{2} (j=2) x B_{0,7} (k in 0..7)
    # This is i=2, j=2, k in {0..7}.
    # Rank = 1 * 1 * 8 = 8.
    for k in range(8):
        U[2 * 3 + 2][col_offset] = 1
        V[2 * 3 + k][col_offset] = 1
        W[k * 3 + 2][col_offset] = 1
        col_offset += 1
    
    # Verify
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0]*rank_school for _ in range(n*m)]
        V_school = [[0]*rank_school for _ in range(m*p)]
        W_school = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school, "V": V_school, "W": W_school}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
