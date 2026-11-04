import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Strassen decomposition of 2x2x2 matrix multiplication, rank 7."""
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need U, V, W such that C_ij = sum_c U_ic V_cj W_kl ... wait.
    # The tensor is T_{ijk} where i is row of A, j is col of B, k is inner?
    # No, standard convention:
    # A is n x m, B is m x p, C is n x p.
    # T_{i, j, k} corresponds to C_{ij} = sum_k A_{ik} B_{kj}.
    # In the problem statement:
    # U has shape (n*m, rank) -> indexed by (i, k)
    # V has shape (m*p, rank) -> indexed by (k, j)
    # W has shape (p*n, rank) -> indexed by (j, i)
    
    # For 2x2x2:
    # n=2, m=2, p=2.
    # U indices: (0,0), (0,1), (1,0), (1,1)
    # V indices: (0,0), (0,1), (1,0), (1,1)
    # W indices: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen terms:
    # c=0: M1
    # A parts: A11, A22. B parts: B11, B22.
    # U: (1,0)=1, (1,1)=1 (A21? No, A22 is (1,1) in 0-indexed? 
    # Let's use 0-indexed:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)] # 2*2
    V = [[0]*7 for _ in range(4)] # 2*2
    W = [[0]*7 for _ in range(4)] # 2*2
    
    # c=0: M1
    # U: A00, A11
    U[0][0] = 1 # A00
    U[3][0] = 1 # A11
    # V: B00, B11
    V[0][0] = 1 # B00
    V[3][0] = 1 # B11
    # W: C00, C11
    W[0][0] = 1 # C00
    W[3][0] = 1 # C11
    
    # c=1: M2
    # U: A10, A11
    U[2][1] = 1 # A10
    U[3][1] = 1 # A11
    # V: B00
    V[0][1] = 1 # B00
    # W: C10
    W[2][1] = 1 # C10
    
    # c=2: M3
    # U: A00
    U[0][2] = 1 # A00
    # V: B10, B11
    V[2][2] = 1 # B10
    V[3][2] = -1 # B11
    # W: C01, C11
    W[1][2] = 1 # C01
    W[3][2] = 1 # C11
    
    # c=3: M4
    # U: A11
    U[3][3] = 1 # A11
    # V: B01, B00
    V[1][3] = 1 # B01
    V[0][3] = -1 # B00
    # W: C00, C10
    W[0][3] = 1 # C00
    W[2][3] = 1 # C10
    
    # c=4: M5
    # U: A00, A01
    U[0][4] = 1 # A00
    U[1][4] = 1 # A01
    # V: B11
    V[3][4] = 1 # B11
    # W: C00, C01
    W[0][4] = -1 # C00
    W[1][4] = 1 # C01
    
    # c=5: M6
    # U: A10, A00
    U[2][5] = 1 # A10
    U[0][5] = -1 # A00
    # V: B00, B01
    V[0][5] = 1 # B00
    V[1][5] = 1 # B01
    # W: C11
    W[3][5] = 1 # C11
    
    # c=6: M7
    # U: A01, A11
    U[1][6] = 1 # A01
    U[3][6] = -1 # A11
    # V: B10, B11
    V[2][6] = 1 # B10
    V[3][6] = 1 # B11
    # W: C00
    W[0][6] = 1 # C00
    
    return {"U": U, "V": V, "W": W}

def _combine(decomps):
    """Combine a list of decompositions by concatenating factors."""
    if not decomps:
        return {"U": [], "V": [], "W": []}
    
    total_rank = sum(len(d["U"][0]) for d in decomps)
    
    U = [[0] * total_rank for _ in range(len(decomps[0]["U"]))]
    V = [[0] * total_rank for _ in range(len(decomps[0]["V"]))]
    W = [[0] * total_rank for _ in range(len(decomps[0]["W"]))]
    
    col_offset = 0
    for d in decomps:
        r = len(d["U"][0])
        for i in range(len(d["U"])):
            for c in range(r):
                U[i][col_offset + c] = d["U"][i][c]
        for i in range(len(d["V"])):
            for c in range(r):
                V[i][col_offset + c] = d["V"][i][c]
        for i in range(len(d["W"])):
            for c in range(r):
                W[i][col_offset + c] = d["W"][i][c]
        col_offset += r
        
    return {"U": U, "V": V, "W": W}

def _sub_decomp(n, m, p, start_i, start_k, start_j):
    """
    Helper to create a sub-decomposition for a block of the tensor.
    This assumes the block is at a specific offset in the larger tensor.
    However, our construction strategy splits the dimensions, so the sub-tensors
    are contiguous blocks.
    """
    # This function is not directly used in the simple concatenation strategy
    # because we split the dimensions and treat them as separate problems,
    # then concatenate the results.
    pass

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy:
    # Split m=5 into m1=3, m2=2.
    # Problem 1: M_{3,3,6}
    # Problem 2: M_{3,2,6}
    
    # Decomposition for M_{3,3,6}:
    # Split p=6 into p1=3, p2=3.
    # Sub-problem 1a: M_{3,3,3}
    # Sub-problem 1b: M_{3,3,3}
    # Use Schoolbook for M_{3,3,3} (Rank 27)
    
    dec_333_a = _schoolbook(3, 3, 3)
    dec_333_b = _schoolbook(3, 3, 3)
    
    # We need to map these to the correct positions in the larger tensors.
    # For M_{3,3,6}, the inner dim is 3 (indices 0,1,2 of the original 5).
    # The outer dim is 6 (indices 0,1,2 for 'a', 3,4,5 for 'b').
    # The row dim is 3 (indices 0,1,2).
    
    # When we concatenate, we just sum the tensors.
    # The "position" is handled by the fact that we are building the full tensor
    # by summing contributions from sub-blocks.
    # However, the sub-blocks must correspond to disjoint sets of (i,k,j) indices?
    # No, the tensor is T_{ijk}.
    # If we split m into m1+m2, then T_{ijk} = sum_{k<m1} A_{ik}B_{kj} + sum_{k>=m1} A_{ik}B_{kj}.
    # These are two separate tensors T1 and T2.
    # T1 is non-zero only for k < m1.
    # T2 is non-zero only for k >= m1.
    # So we can decompose T1 and T2 independently and sum the decompositions.
    
    # So, for M_{3,3,6} (which is T1 of the main split):
    # We split p into 3+3.
    # T1 = T1a + T1b.
    # T1a is M_{3,3,3} with p indices 0,1,2.
    # T1b is M_{3,3,3} with p indices 3,4,5.
    
    # The decompositions _schoolbook(3,3,3) produce U, V, W for a 3x3x3 tensor.
    # We need to embed them into the 3x3x6 tensor.
    # U is (3*3, rank). Indices (i, k).
    # V is (3*6, rank). Indices (k, j).
    # W is (6*3, rank). Indices (j, i).
    
    # For T1a (k in 0..2, j in 0..2):
    # U is same as schoolbook U.
    # V: only k in 0..2, j in 0..2 are non-zero.
    # W: only j in 0..2 are non-zero.
    
    # Let's write a helper to embed a sub-decomposition into a larger one.
    
    def embed(dec, n, m, p, off_i, off_k, off_j):
        """
        Embed a decomposition of M_{n',m',p'} into M_{n,m,p} at offset (off_i, off_k, off_j).
        """
        n_prime = len(dec["U"]) // len(dec["V"]) # This is fragile. 
        # Better to pass dimensions explicitly.
        pass

    # Let's do it manually for the specific splits.
    
    # Main Split: m=5 -> 3+2
    # Part 1: M_{3,3,6} (k=0..2)
    # Part 2: M_{3,2,6} (k=3..4)
    
    # Part 1: M_{3,3,6}
    # Split p=6 -> 3+3
    # Part 1a: M_{3,3,3} (k=0..2, j=0..2)
    # Part 1b: M_{3,3,3} (k=0..2, j=3..5)
    
    # Part 2: M_{3,2,6}
    # Split n=3 -> 2+1
    # Part 2a: M_{2,2,6} (i=0..1, k=3..4)
    # Part 2b: M_{1,2,6} (i=2, k=3..4)
    
    # Part 2a: M_{2,2,6}
    # Split p=6 -> 2+2+2
    # Part 2a1: M_{2,2,2} (i=0..1, k=3..4, j=0..1)
    # Part 2a2: M_{2,2,2} (i=0..1, k=3..4, j=2..3)
    # Part 2a3: M_{2,2,2} (i=0..1, k=3..4, j=4..5)
    
    # Part 2b: M_{1,2,6}
    # Schoolbook (Rank 12)
    
    # Let's construct the embedded decompositions.
    
    def create_embedded_schoolbook(n, m, p, off_i, off_k, off_j):
        """Create schoolbook decomposition for a sub-block."""
        rank = n * m * p
        U = [[0] * rank for _ in range(3 * 5)] # Full U size
        V = [[0] * rank for _ in range(5 * 6)] # Full V size
        W = [[0] * rank for _ in range(6 * 3)] # Full W size
        
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[off_i + i, off_k + k][col] = 1
                    V[off_k + k, off_j + j][col] = 1
                    W[off_j + j, off_i + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def create_embedded_strassen(off_i, off_k, off_j):
        """Create embedded Strassen decomposition for a 2x2x2 sub-block."""
        dec = _strassen_222()
        rank = 7
        U = [[0] * rank for _ in range(3 * 5)]
        V = [[0] * rank for _ in range(5 * 6)]
        W = [[0] * rank for _ in range(6 * 3)]
        
        for c in range(rank):
            for i in range(2):
                for k in range(2):
                    U[off_i + i, off_k + k][c] = dec["U"][i * 2 + k][c]
            for k in range(2):
                for j in range(2):
                    V[off_k + k, off_j + j][c] = dec["V"][k * 2 + j][c]
            for j in range(2):
                for i in range(2):
                    W[off_j + j, off_i + i][c] = dec["W"][j * 2 + i][c]
                    
        return {"U": U, "V": V, "W": W}

    decomps = []
    
    # Part 1a: M_{3,3,3} at (0,0,0)
    decomps.append(create_embedded_schoolbook(3, 3, 3, 0, 0, 0))
    
    # Part 1b: M_{3,3,3} at (0,0,3)
    decomps.append(create_embedded_schoolbook(3, 3, 3, 0, 0, 3))
    
    # Part 2a1: M_{2,2,2} at (0,3,0)
    decomps.append(create_embedded_strassen(0, 3, 0))
    
    # Part 2a2: M_{2,2,2} at (0,3,2)
    decomps.append(create_embedded_strassen(0, 3, 2))
    
    # Part 2a3: M_{2,2,2} at (0,3,4)
    decomps.append(create_embedded_strassen(0, 3, 4))
    
    # Part 2b: M_{1,2,6} at (2,3,0)
    decomps.append(create_embedded_schoolbook(1, 2, 6, 2, 3, 0))
    
    # Combine
    final_dec = _combine(decomps)
    
    return final_dec

# EVOLVE-BLOCK-END
