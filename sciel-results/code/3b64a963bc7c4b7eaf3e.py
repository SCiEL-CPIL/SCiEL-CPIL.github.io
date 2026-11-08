import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation
    # Block 1: 2x2x2 (Rank 7)
    # Block 2: 2x3x4 (Rank 18 = 7 + 7 + 4)
    total_rank = 7 + 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 block
    # i_off: offset for i (0 or 1)
    # j_off: offset for j (0, 2)
    # k_off: offset for k (0, 2, 4)
    # rank_offset: starting column index in U,V,W
    def add_strassen(i_off, j_off, k_off, rank_offset):
        # U indices: (i, j) -> i*m + j
        # V indices: (j, k) -> j*p + k
        # W indices: (k, i) -> k*n + i
        
        # Factor 1: (A00+A11)(B00+B11)
        U[i_off*m + j_off][rank_offset + 0] = 1
        U[(i_off+1)*m + (j_off+1)][rank_offset + 0] = 1
        V[j_off*p + k_off][rank_offset + 0] = 1
        V[(j_off+1)*p + (k_off+1)][rank_offset + 0] = 1
        W[k_off*n + i_off][rank_offset + 0] = 1
        W[(k_off+1)*n + (i_off+1)][rank_offset + 0] = 1
        
        # Factor 2: (A10+A11)B00
        U[(i_off+1)*m + j_off][rank_offset + 1] = 1
        U[(i_off+1)*m + (j_off+1)][rank_offset + 1] = 1
        V[j_off*p + k_off][rank_offset + 1] = 1
        W[k_off*n + (i_off+1)][rank_offset + 1] = 1
        
        # Factor 3: A00(B01-B11)
        U[i_off*m + j_off][rank_offset + 2] = 1
        V[j_off*p + (k_off+1)][rank_offset + 2] = 1
        V[(j_off+1)*p + (k_off+1)][rank_offset + 2] = -1
        W[(k_off+1)*n + i_off][rank_offset + 2] = 1
        W[(k_off+1)*n + (i_off+1)][rank_offset + 2] = 1
        
        # Factor 4: A11(B10-B00)
        U[(i_off+1)*m + (j_off+1)][rank_offset + 3] = 1
        V[(j_off+1)*p + k_off][rank_offset + 3] = 1
        V[j_off*p + k_off][rank_offset + 3] = -1
        W[k_off*n + i_off][rank_offset + 3] = 1
        W[k_off*n + (i_off+1)][rank_offset + 3] = 1
        
        # Factor 5: (A00+A01)B11
        U[i_off*m + j_off][rank_offset + 4] = 1
        U[i_off*m + (j_off+1)][rank_offset + 4] = 1
        V[(j_off+1)*p + (k_off+1)][rank_offset + 4] = 1
        W[(k_off+1)*n + i_off][rank_offset + 4] = 1
        W[k_off*n + i_off][rank_offset + 4] = -1
        
        # Factor 6: (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off][rank_offset + 5] = 1
        U[i_off*m + j_off][rank_offset + 5] = -1
        V[j_off*p + k_off][rank_offset + 5] = 1
        V[j_off*p + (k_off+1)][rank_offset + 5] = 1
        W[(k_off+1)*n + (i_off+1)][rank_offset + 5] = 1
        
        # Factor 7: (A01-A11)(B10+B11)
        U[i_off*m + (j_off+1)][rank_offset + 6] = 1
        U[(i_off+1)*m + (j_off+1)][rank_offset + 6] = -1
        V[(j_off+1)*p + k_off][rank_offset + 6] = 1
        V[(j_off+1)*p + (k_off+1)][rank_offset + 6] = 1
        W[k_off*n + i_off][rank_offset + 6] = 1

    # Helper to add a 2x1x4 block (Trivial)
    # i_off, j_off (single j), k_off (4 k's)
    def add_trivial_2x1x4(i_off, j_off, k_off, rank_offset):
        # A is 2x1, B is 1x4
        # C is 2x4
        # C[i,k] = A[i,j] B[j,k]
        for r in range(4):
            k_idx = k_off + r
            # U[i,j] = 1
            U[i_off*m + j_off][rank_offset + r] = 1
            U[(i_off+1)*m + j_off][rank_offset + r] = 1
            # V[j,k] = 1
            V[j_off*p + k_idx][rank_offset + r] = 1
            # W[k,i] = 1
            W[k_idx*n + i_off][rank_offset + r] = 1
            W[k_idx*n + (i_off+1)][rank_offset + r] = 1

    # 1. Add Block 1: 2x2x2 at i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0, 0)
    
    # 2. Add Block 2a: 2x2x2 at i=0,1; j=2,3; k=2,3
    add_strassen(0, 2, 2, 7)
    
    # 3. Add Block 2b: 2x2x2 at i=0,1; j=2,3; k=4,5
    add_strassen(0, 2, 4, 14)
    
    # 4. Add Block 2c: 2x1x4 at i=0,1; j=4; k=2,3,4,5
    add_trivial_2x1x4(0, 4, 2, 21)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
