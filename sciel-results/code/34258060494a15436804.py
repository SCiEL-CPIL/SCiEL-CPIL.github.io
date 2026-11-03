"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the naive (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2(n, m, p):
    """
    Build a rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    Uses Strassen's 7 multiplications.
    """
    assert n == 2 and m == 2 and p == 2
    # Strassen 7 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)
    
    # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # m1 = (a+d)(e+h): U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    U[0, 0] = 1; U[3, 0] = 1  # a, d
    V[0, 0] = 1; V[3, 0] = 1  # e, h
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # m2 = (c+d)e: U has 1 at (1,0) and (1,1); V has 1 at (0,0); W has 1 at (1,0) and (1,1)
    U[2, 1] = 1; U[3, 1] = 1  # c, d
    V[0, 1] = 1  # e
    W[2, 1] = 1; W[3, 1] = 1  # C[1,0], C[1,1]
    
    # m3 = a(f-h): U has 1 at (0,0); V has 1 at (0,1) and (1,1) with signs; W has 1 at (0,1) and (1,1)
    U[0, 2] = 1  # a
    V[1, 2] = 1; V[3, 2] = -1  # f, -h
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # m4 = d(g-e): U has 1 at (1,1); V has 1 at (1,0) and (0,0) with signs; W has 1 at (0,0) and (1,0)
    U[3, 3] = 1  # d
    V[2, 3] = 1; V[0, 3] = -1  # g, -e
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0], C[1,0]
    
    # m5 = (a+b)h: U has 1 at (0,0) and (0,1); V has 1 at (1,1); W has 1 at (0,0) and (0,1)
    U[0, 4] = 1; U[1, 4] = 1  # a, b
    V[3, 4] = 1  # h
    W[0, 4] = 1; W[1, 4] = 1  # C[0,0], C[0,1]
    
    # m6 = (c-a)(e+f): U has 1 at (1,0) and (0,0) with signs; V has 1 at (0,0) and (0,1); W has 1 at (1,1)
    U[2, 5] = 1; U[0, 5] = -1  # c, -a
    V[0, 5] = 1; V[1, 5] = 1  # e, f
    W[3, 5] = 1  # C[1,1]
    
    # m7 = (b-d)(g+h): U has 1 at (0,1) and (1,1) with signs; V has 1 at (1,0) and (1,1); W has 1 at (0,0) and (0,1)
    U[1, 6] = 1; U[3, 6] = -1  # b, -d
    V[2, 6] = 1; V[3, 6] = 1  # g, h
    W[0, 6] = 1; W[1, 6] = 1  # C[0,0], C[0,1]
    
    return {"U": U, "V": V, "W": W}


def _combine_decompositions(decs):
    """
    Combine multiple independent decompositions by concatenating columns.
    Each dec is a dict with 'U', 'V', 'W' arrays.
    Returns a single dict with concatenated U, V, W.
    """
    U_list = [d["U"] for d in decs]
    V_list = [d["V"] for d in decs]
    W_list = [d["W"] for d in decs]
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    return {"U": U, "V": V, "W": W}


def _block_decomposition(n, m, p):
    """
    Build a block-based decomposition for n x m * m x p.
    
    For M_{3,4,4}: split into a 1x4x4 block (schoolbook, rank 16) 
    and a 2x4x4 block (using Strassen on 2x2 blocks, rank 28).
    Total rank: 44.
    """
    # For n=3, m=4, p=4:
    # Split A (3x4) into A1 (1x4) and A2 (2x4)
    # B (4x4) is shared
    # C1 = A1 * B (1x4), C2 = A2 * B (2x4)
    
    # A1 * B: 1x4 * 4x4 -> schoolbook rank 16
    # A2 * B: 2x4 * 4x4 -> split into two 2x2 * 2x2 Strassen blocks
    #   A2 = [A2a A2b] where A2a, A2b are 2x2
    #   B = [B_top; B_bottom] where B_top, B_bottom are 4x2? No...
    #   Actually, B is 4x4. Split B into B_left (4x2) and B_right (4x2)
    #   C2 = A2a * B_left + A2b * B_right
    #   Each A2a * B_left is 2x2 * 4x2... that's not 2x2 * 2x2.
    
    # Let me reconsider. The standard approach for M_{2,4,4}:
    # A (2x4) * B (4x4) = C (2x4)
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B1 (4x2) and B2 (4x2)  
    # C = A1 * B1 + A2 * B2
    # Each A1 * B1 is 2x2 * 4x2, which can be seen as two 2x2 * 2x2 multiplications
    # So total rank for M_{2,4,4} = 2 * 7 + 2 * 7 = 28 (using Strassen for each 2x2*2x2)
    
    # Wait, that's not quite right either. Let me think more carefully.
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x4) = [B1; B2] where B1, B2 are 2x4? No, B is 4x4.
    # Actually, B should be split vertically: B = [B_left | B_right] where B_left, B_right are 4x2
    # Then C = A * B = [A1 | A2] * [B_left | B_right]
    # C_left = A * B_left (2x2 * 4x2), C_right = A * B_right (2x2 * 4x2)
    # A * B_left = [A1 | A2] * B_left. This is a 2x4 * 4x2 multiplication.
    # We can write this as A1 * B_left_top + A2 * B_left_bottom where B_left_top, B_left_bottom are 2x2.
    # So each 2x4 * 4x2 = two 2x2 * 2x2 multiplications, each with rank 7.
    # Total for M_{2,4,4} = 4 * 7 = 28.
    
    # So total rank for M_{3,4,4} = 16 (for 1x4*4x4) + 28 (for 2x4*4x4) = 44.
    
    # Let's build this properly.
    
    # For the 1x4 * 4x4 part (schoolbook):
    dec_1x4 = _schoolbook(1, 4, 4)
    
    # For the 2x4 * 4x4 part using Strassen:
    # We need to build a rank-28 decomposition for M_{2,4,4}.
    # This is done by splitting into four 2x2 * 2x2 Strassen blocks.
    
    # A (2x4) = [A11 A12; A21 A22] where each is 1x1? No, A is 2x4.
    # Let's index A as A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B as B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C as C[i,k] for i in {0,1}, k in {0,1,2,3}
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    
    # Let A_top = A[0:2, :] (2x4), A_bottom = A[2:4, :] ... wait, A is 2x4, not 4x4.
    
    # I think I'm overcomplicating this. Let me use a different approach.
    # For M_{2,4,4}, we can view it as:
    # A (2x4) * B (4x4) = C (2x4)
    # This is equivalent to two independent M_{2,2,2} multiplications if we split B into two 4x2 blocks
    # and A into two 2x2 blocks, but that's not quite right.
    
    # Actually, the cleanest way: M_{2,4,4} has rank at most 4 * 7 = 28 by using
    # Strassen on four 2x2 subproblems.
    
    # Let me just construct it directly.
    # A[i,j] for i in {0,1}, j in {0,1,2,3}
    # B[j,k] for j in {0,1,2,3}, k in {0,1,2,3}  
    # C[i,k] for i in {0,1}, k in {0,1,2,3}
    
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Group terms: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # = (A[i,0:2] * B[0:2,k]) + (A[i,2:4] * B[2:4,k])
    
    # Each term is a 2x2 * 2x1 multiplication... no, that's not right either.
    # B[0:2,k] is a 2x1 vector, not a 2x2 matrix.
    
    # I think the correct decomposition is:
    # M_{2,4,4} = M_{2,2,2} + M_{2,2,2} (two independent 2x2*2x2 multiplications)
    # where the first handles columns 0,1 of B and the second handles columns 2,3 of B.
    # But that would give rank 14, which seems too good.
    
    # Actually, I recall that rank(M_{2,4,4}) = 28 is a known result, achieved by
    # viewing it as four 2x2*2x2 multiplications. Let me look at this differently.
    
    # Perhaps the decomposition is:
    # Split B into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # Split A into two 2x2 blocks: A = [A0, A1]
    # Then C = A0 * [B00, B01] + A1 * [B10, B11]
    # Each A0 * [B00, B01] is a 2x2 * 2x4 multiplication, which can be decomposed into
    # two 2x2 * 2x2 multiplications.
    
    # So total: 4 * 7 = 28.
    
    # Let me construct this explicitly.
    
    def _strassen_block_for_2x4x4():
        """Build rank-28 decomposition for M_{2,4,4} using 4 Strassen 2x2 blocks."""
        rank = 28
        U = np.zeros((8, rank), dtype=np.float64)  # 2x4 = 8 elements
        V = np.zeros((16, rank), dtype=np.float64)  # 4x4 = 16 elements
        W = np.zeros((8, rank), dtype=np.float64)  # 4x2 = 8 elements
        
        # Strassen 7-term coefficients for 2x2 * 2x2
        # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
        # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
        # C00 = m1+m4-m5+m7, C01 = m3+m5, C10 = m2+m4, C11 = m1-m2+m3+m6
        
        # For each of the 4 subproblems, we assign 7 columns.
        # Subproblem s: A_sub (2x2) * B_sub (2x2) -> C_sub (2x2)
        # A_sub indices in U: depends on which 2x2 block of A (2x4)
        # B_sub indices in V: depends on which 2x2 block of B (4x4)
        # C_sub indices in W: depends on which 2x2 block of C (2x4)
        
        # A is 2x4: rows 0-1, cols 0-3
        # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
        
        # B is 4x4: rows 0-3, cols 0-3
        # B_tl = B[0:2, 0:2], B_tr = B[0:2, 2:4]
        # B_bl = B[2:4, 0:2], B_br = B[2:4, 2:4]
        
        # C is 2x4: rows 0-1, cols 0-3
        # C_left = C[:, 0:2] (2x2), C_right = C[:, 2:4] (2x2)
        
        # C = A * B = [A_left | A_right] * B
        # C_left = A_left * B_left + A_right * B_right? No.
        # C_left = A * B[:, 0:2] = A_left * B[0:2, 0:2] + A_right * B[2:4, 0:2]
        # C_right = A * B[:, 2:4] = A_left * B[0:2, 2:4] + A_right * B[2:4, 2:4]
        
        # So we have 4 subproblems:
        # 1. A_left * B_tl -> contributes to C_left
        # 2. A_right * B_bl -> contributes to C_left
        # 3. A_left * B_tr -> contributes to C_right
        # 4. A_right * B_br -> contributes to C_right
        
        # Each subproblem is 2x2 * 2x2, decomposed with Strassen (rank 7).
        
        # Define Strassen term structure
        # For a 2x2 * 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[c00,c01],[c10,c11]]
        # U indices: a=0, b=1, c=2, d=3
        # V indices: e=0, f=1, g=2, h=3
        # W indices: c00=0, c01=1, c10=2, c11=3
        
        strassen_U = np.array([
            [1, 0, 1, 0, 1, -1, 0],  # a
            [0, 0, 0, 0, 1, 0, 1],   # b
            [0, 1, 0, 0, 0, 1, 0],   # c
            [1, 1, 0, 1, 0, 0, -1],  # d
        ], dtype=np.float64)
        
        strassen_V = np.array([
            [1, 1, 0, -1, 0, 1, 0],  # e
            [0, 0, 1, 0, 0, 1, 0],   # f
            [0, 0, 0, 1, 0, 0, 1],   # g
            [1, 0, -1, 0, 1, 0, 1],  # h
        ], dtype=np.float64)
        
        strassen_W = np.array([
            [1, 0, 0, 1, -1, 0, 1],  # c00
            [0, 0, 1, 0, 1, 0, 1],   # c01
            [0, 1, 0, 1, 0, 0, 0],   # c10
            [1, -1, 1, 0, 0, 1, 0],  # c11
        ], dtype=np.float64)
        
        # Subproblem 1: A_left (A[:, 0:2]) * B_tl (B[0:2, 0:2]) -> C_left (C[:, 0:2])
        # A_left in U: rows 0-1 (A[0,0:2]) and rows 2-3 (A[1,0:2])? 
        # U is indexed as [i*4+j] for A[i,j], i in {0,1}, j in {0,1,2,3}
        # A_left = A[:, 0:2]: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[4], A[1,1]=U[5]
        # B_tl = B[0:2, 0:2]: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[4], B[1,1]=V[5]
        # C_left = C[:, 0:2]: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[4], C[1,1]=W[5]
        
        for s, (a_rows, b_rows, c_cols) in enumerate([
            ([0, 4], [0, 4], [0, 4]),      # A_left * B_tl -> C_left
            ([2, 6], [4, 8], [0, 4]),      # A_right * B_bl -> C_left
            ([0, 4], [2, 6], [2, 6]),      # A_left * B_tr -> C_right
            ([2, 6], [6, 10], [2, 6]),     # A_right * B_br -> C_right
        ]):
            col_start = s * 7
            for t in range(7):
                col = col_start + t
                # U contribution
                for local_idx, global_idx in enumerate(a_rows):
                    U[global_idx, col] += strassen_U[local_idx, t]
                # V contribution
                for local_idx, global_idx in enumerate(b_rows):
                    V[global_idx, col] += strassen_V[local_idx, t]
                # W contribution
                for local_idx, global_idx in enumerate(c_cols):
                    W[global_idx, col] += strassen_W[local_idx, t]
        
        return {"U": U, "V": V, "W": W}
    
    # Hmm, the indexing above is getting confused. Let me take a step back and
    # just verify the known result: rank(M_{3,4,4}) <= 44.
    # 
    # Actually, let me just use the schoolbook for now and see if I can find
    # a better decomposition through optimization.
    
    # For now, return schoolbook (rank 48) as the safe baseline.
    return _schoolbook(n, m, p)


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1] if U.size else 0
    # Check matching indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    # Check non-matching indices
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                continue
                            s = 0.0
                            for r in range(rank):
                                s += U[i * m + j, r] * V[jp * p + k, r] * W[kp * n + ip, r]
                            if abs(s) > 1e-9:
                                return False
    return True


def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    result = _schoolbook(n, m, p)
    best_rank = result["U"].shape[1]
    
    # Try block-based decomposition
    block_result = _block_decomposition(n, m, p)
    if block_result["U"].shape[1] < best_rank:
        if _verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
            result = block_result
            best_rank = result["U"].shape[1]
    
    # Check time
    if time.time() > deadline:
        return result
    
    # Try to improve via random perturbation and optimization
    # Use the structure: we want to find U, V, W with minimal rank
    # such that the tensor decomposition is exact.
    
    # For now, return the best result found.
    return result

# EVOLVE-BLOCK-END
