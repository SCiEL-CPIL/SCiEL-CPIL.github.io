"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition using Strassen blocks"""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    def strassen_2x2x2(i0, j0, k0):
        """Strassen factors for 2x2x2 block at offset (i0, j0, k0)."""
        factors = []
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        factors.append((
            {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
            {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
            {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
        ))
        # M2 = A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
        factors.append((
            {(i0+1)*m+(j0+1): 1},
            {(j0+1)*p+k0: 1, j0*p+k0: -1},
            {k0*n+(i0+1): -1, (k0+1)*n+(i0+1): 1}
        ))
        # M3 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        factors.append((
            {i0*m+j0: 1, i0*m+(j0+1): 1},
            {(j0+1)*p+(k0+1): 1},
            {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
        ))
        # M4 = A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
        factors.append((
            {i0*m+j0: 1},
            {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
            {k0*n+i0: 1, (k0+1)*n+i0: -1}
        ))
        # M5 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        factors.append((
            {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
            {j0*p+k0: 1},
            {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
        ))
        # M6 = (A[i0,j0]-A[i0+1,j0]) * (B[j0+1,k0]+B[j0+1,k0+1])
        factors.append((
            {i0*m+j0: 1, (i0+1)*m+j0: -1},
            {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
            {k0*n+i0: 1, k0*n+(i0+1): 1}
        ))
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0,k0]+B[j0,k0+1])
        factors.append((
            {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
            {j0*p+k0: 1, j0*p+(k0+1): 1},
            {(k0+1)*n+i0: -1, (k0+1)*n+(i0+1): 1}
        ))
        return factors
    
    def schoolbook_block(i0, j0, k0, ni, nj, nk):
        """Schoolbook factors for block of size ni x nj x nk at offset (i0,j0,k0)."""
        factors = []
        for ii in range(ni):
            for jj in range(nj):
                for kk in range(nk):
                    i = i0 + ii
                    j = j0 + jj
                    k = k0 + kk
                    factors.append((
                        {i*m+j: 1},
                        {j*p+k: 1},
                        {k*n+i: 1}
                    ))
        return factors
    
    def decompose_2x2x3(i0, j0, k0):
        """Decompose 2x2x3 block: Strassen(2x2x2) + schoolbook(2x2x1). Rank = 7+2 = 9."""
        factors = []
        factors.extend(strassen_2x2x2(i0, j0, k0))
        factors.extend(schoolbook_block(i0, j0, k0+2, 2, 2, 1))
        return factors
    
    def decompose_2x3x3(i0, j0, k0):
        """Decompose 2x3x3 block: split j into 2+1. Rank = 9+6 = 15."""
        factors = []
        # 2x2x3 part at (i0, j0, k0)
        factors.extend(decompose_2x2x3(i0, j0, k0))
        # 2x1x3 part at (i0, j0+2, k0)
        factors.extend(schoolbook_block(i0, j0+2, k0, 2, 1, 3))
        return factors
    
    # Split: 4=2+2, 5=2+3, 6=3+3
    # 8 blocks: 4 of type 2x2x3, 4 of type 2x3x3
    all_factors = []
    
    # Top-left (i:0-1, j:0-1, k:0-2): 2x2x3
    all_factors.extend(decompose_2x2x3(0, 0, 0))
    # Top-left (i:0-1, j:0-1, k:3-5): 2x2x3
    all_factors.extend(decompose_2x2x3(0, 0, 3))
    # Top-right (i:0-1, j:2-4, k:0-2): 2x3x3
    all_factors.extend(decompose_2x3x3(0, 2, 0))
    # Top-right (i:0-1, j:2-4, k:3-5): 2x3x3
    all_factors.extend(decompose_2x3x3(0, 2, 3))
    # Bottom-left (i:2-3, j:0-1, k:0-2): 2x2x3
    all_factors.extend(decompose_2x2x3(2, 0, 0))
    # Bottom-left (i:2-3, j:0-1, k:3-5): 2x2x3
    all_factors.extend(decompose_2x2x3(2, 0, 3))
    # Bottom-right (i:2-3, j:2-4, k:0-2): 2x3x3
    all_factors.extend(decompose_2x3x3(2, 2, 0))
    # Bottom-right (i:2-3, j:2-4, k:3-5): 2x3x3
    all_factors.extend(decompose_2x3x3(2, 2, 3))
    
    rank = len(all_factors)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_factors):
        for row, coeff in u_dict.items():
            U[row][col] = coeff
        for row, coeff in v_dict.items():
            V[row][col] = coeff
        for row, coeff in w_dict.items():
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
