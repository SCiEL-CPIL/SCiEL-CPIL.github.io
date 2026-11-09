import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # A is n x m (2x4)
    # B is m x p (4x8)
    # C is n x p (2x8)
    
    # We use 8 blocks of Strassen (2x2x2)
    # Total rank = 8 * 7 = 56
    rank = 56
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen formulas for 2x2 matrices A, B:
    # P1 = (A00 + A11) * (B00 + B11)
    # P2 = (A10 + A11) * B00
    # P3 = A00 * (B10 - B11)
    # P4 = A11 * (B01 - B00)
    # P5 = (A00 + A01) * B11
    # P6 = (A10 - A00) * (B00 + B01)
    # P7 = (A01 - A11) * (B10 + B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Mapping local (a00, a01, a10, a11) to global A indices
    # Mapping local (b00, b01, b10, b11) to global B indices
    
    # Loop over the 8 independent 2x2x2 problems
    # k_block: 0 or 1 (corresponds to A columns 0-1 or 2-3, and B rows 0-1 or 2-3)
    # j_block: 0 to 3 (corresponds to B columns 0-1, 2-3, 4-5, 6-7)
    
    for k_block in range(2):
        for j_block in range(4):
            
            # Global indices for A
            # A is 2x4.
            # a00 -> A[0, k_block*2]
            # a01 -> A[0, k_block*2 + 1]
            # a10 -> A[1, k_block*2]
            # a11 -> A[1, k_block*2 + 1]
            
            u_a00 = 0 * m + (k_block * 2)
            u_a01 = 0 * m + (k_block * 2 + 1)
            u_a10 = 1 * m + (k_block * 2)
            u_a11 = 1 * m + (k_block * 2 + 1)
            
            # Global indices for B
            # B is 4x8.
            # b00 -> B[k_block*2, j_block*2]
            # b01 -> B[k_block*2, j_block*2 + 1]
            # b10 -> B[k_block*2 + 1, j_block*2]
            # b11 -> B[k_block*2 + 1, j_block*2 + 1]
            
            v_b00 = (k_block * 2) * p + (j_block * 2)
            v_b01 = (k_block * 2) * p + (j_block * 2 + 1)
            v_b10 = (k_block * 2 + 1) * p + (j_block * 2)
            v_b11 = (k_block * 2 + 1) * p + (j_block * 2 + 1)
            
            # Global indices for C
            # C is 2x8.
            # c00 -> C[0, j_block*2] -> W[(j_block*2)*n + 0]
            # c01 -> C[0, j_block*2 + 1] -> W[(j_block*2+1)*n + 0]
            # c10 -> C[1, j_block*2] -> W[(j_block*2)*n + 1]
            # c11 -> C[1, j_block*2 + 1] -> W[(j_block*2+1)*n + 1]
            
            w_c00 = (j_block * 2) * n + 0
            w_c01 = (j_block * 2 + 1) * n + 0
            w_c10 = (j_block * 2) * n + 1
            w_c11 = (j_block * 2 + 1) * n + 1
            
            # P1
            U[u_a00][col] += 1.0
            U[u_a11][col] += 1.0
            V[v_b00][col] += 1.0
            V[v_b11][col] += 1.0
            W[w_c00][col] += 1.0
            W[w_c11][col] += 1.0
            col += 1
            
            # P2
            U[u_a10][col] += 1.0
            U[u_a11][col] += 1.0
            V[v_b00][col] += 1.0
            W[w_c10][col] += 1.0
            W[w_c11][col] -= 1.0
            col += 1
            
            # P3
            U[u_a00][col] += 1.0
            V[v_b10][col] += 1.0
            V[v_b11][col] -= 1.0
            W[w_c01][col] += 1.0
            W[w_c11][col] += 1.0
            col += 1
            
            # P4
            U[u_a11][col] += 1.0
            V[v_b01][col] += 1.0
            V[v_b00][col] -= 1.0
            W[w_c00][col] += 1.0
            W[w_c10][col] += 1.0
            col += 1
            
            # P5
            U[u_a00][col] += 1.0
            U[u_a01][col] += 1.0
            V[v_b11][col] += 1.0
            W[w_c00][col] -= 1.0
            W[w_c01][col] += 1.0
            col += 1
            
            # P6
            U[u_a10][col] += 1.0
            U[u_a00][col] -= 1.0
            V[v_b00][col] += 1.0
            V[v_b01][col] += 1.0
            W[w_c11][col] += 1.0
            col += 1
            
            # P7
            U[u_a01][col] += 1.0
            U[u_a11][col] -= 1.0
            V[v_b10][col] += 1.0
            V[v_b11][col] += 1.0
            W[w_c00][col] += 1.0
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
