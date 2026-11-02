import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    
    Strategy:
    1. Construct a rank-30 solution using:
       - 4 Strassen 2x2x2 blocks for the 2x2x4 portion (rank 28)
       - 2 schoolbook 2x2x1 blocks for the 2x2x1 portion (rank 4)
       Total: 32... wait, let me reconsider.
       
    Actually, let me use the known decomposition:
    - Split the 4x5 matrix into 2x2 and 2x3 blocks along the m dimension
    - For 2x2x2: use Strassen (rank 7) for each 2x2x2 block
    - For 2x2x3: use a rank-10 decomposition
    - For 2x2x1: use schoolbook (rank 4)
    
    The 4x5 matrix can be viewed as:
    - First 4 columns: 2x2 blocks (2 blocks of 2x2)
    - Last 1 column: 2x1 block
    
    So: 2 Strassen blocks (2*7=14) + 1 schoolbook 2x2x1 block (4) = 18? No...
    
    Let me think more carefully. The tensor is T_{i,j,k} where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}.
    
    We can partition j into {0,1} and {2,3}, and k into {0,1,2,3} and {4}.
    
    For the j={0,1}, k={0,1} block: 2x2x2, rank 7 (Strassen)
    For the j={0,1}, k={2,3} block: 2x2x2, rank 7 (Strassen)
    For the j={2,3}, k={0,1} block: 2x2x2, rank 7 (Strassen)
    For the j={2,3}, k={2,3} block: 2x2x2, rank 7 (Strassen)
    For the j={0,1}, k={4} block: 2x2x1, rank 4 (schoolbook)
    For the j={2,3}, k={4} block: 2x2x1, rank 4 (schoolbook)
    
    Total: 4*7 + 2*4 = 28 + 8 = 36. That's what we already have.
    
    To get rank 30, we need a better decomposition. The key insight is to use a 2x2x3 decomposition of rank 10 instead of two 2x2x2 blocks.
    
    Let me use:
    - Partition k into {0,1,2} and {3,4}
    - For j={0,1}, k={0,1,2}: 2x2x3, rank 10
    - For j={2,3}, k={0,1,2}: 2x2x3, rank 10
    - For j={0,1}, k={3,4}: 2x2x2, rank 7
    - For j={2,3}, k={3,4}: 2x2x2, rank 7
    Total: 10+10+7+7 = 34
    
    Or better:
    - For j={0,1}, k={0,1,2,3}: 2x2x4, can be done with rank 14 (2 Strassen) or better
    - For j={2,3}, k={0,1,2,3}: 2x2x4, rank 14
    - For j={0,1}, k={4}: 2x2x1, rank 4
    - For j={2,3}, k={4}: 2x2x1, rank 4
    Total: 14+14+4+4 = 36
    
    Let me try the 2x2x3 rank-10 approach:
    - j={0,1}, k={0,1,2}: 2x2x3, rank 10
    - j={2,3}, k={0,1,2}: 2x2x3, rank 10
    - j={0,1}, k={3,4}: 2x2x2, rank 7
    - j={2,3}, k={3,4}: 2x2x2, rank 7
    Total: 34
    
    Actually, I recall that 2x2x3 has a known rank-10 decomposition. Let me construct it.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    time_limit = 130  # Leave margin
    
    def verify_solution(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                val = 0.0
                                for mu in range(R):
                                    val += U[a*m+b, mu] * V[c*p+d, mu] * W[k*n+i, mu]
                                if b == c and a == i and d == k:
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                                else:
                                    if abs(val - 0.0) > 1e-9:
                                        return False
        return True

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    # --- Build rank-34 solution using 2x2x3 rank-10 blocks ---
    def get_solution_rank34():
        rank = 34
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A10+A11)B00
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
            
            return col
        
        def add_2x2x3_rank10(i0, j0, k0):
            """
            Add a 2x2x3 decomposition of rank 10.
            A is 2x2 (rows i0, i0+1; cols j0, j0+1)
            B is 2x3 (rows j0, j0+1; cols k0, k0+1, k0+2)
            C is 2x3 (rows k0, k0+1, k0+2; cols i0, i0+1)
            """
            nonlocal col
            
            # Known rank-10 decomposition for 2x2x3
            # M1: (A00+A11)(B00+B11+B22)
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+2), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1; W[(k0+2)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A10+A11)(B00+B11)
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M3: A00(B10-B11)
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M4: A11(B01-B00)
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A00+A01)B11
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M8: A01*B20
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+2), col] += 1
            W[(k0+2)*n + i0, col] += 1
            col += 1
            
            # M9: A10*B21
            U[(i0+1)*m + j0, col] += 1
            V[j0*p + (k0+2), col] += 1
            W[(k0+2)*n + (i0+1), col] += 1
            col += 1
            
            # M10: A11*B22
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+2), col] += 1
            W[(k0+2)*n + (i0+1), col] += 1
            col += 1
            
            return col
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] += 1
                    V[(j0 + dj)*p + k0, col] += 1
                    W[k0*n + (i0 + di), col] += 1
                    col += 1
            return col
        
        # 2x2x3 blocks for k={0,1,2}
        add_2x2x3_rank10(0, 0, 0)  # i=0, j={0,1}, k={0,1,2}
        add_2x2x3_rank10(0, 2, 0)  # i=0, j={2,3}, k={0,1,2}
        
        # 2x2x2 blocks for k={3,4}
        add_strassen_222(0, 0, 3)  # i=0, j={0,1}, k={3,4}
        add_strassen_222(0, 2, 3)  # i=0, j={2,3}, k={3,4}
        
        return U, V, W

    # First, try the rank-34 solution
    U34, V34, W34 = get_solution_rank34()
    
    if verify_solution(U34, V34, W34):
        U_snap, V_snap, W_snap = snap_to_half_integers(U34, V34, W34)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            return {"U": U34.tolist(), "V": V34.tolist(), "W": W34.tolist()}
    
    # Fallback to rank-36 solution
    def get_solution_rank36():
        rank = 36
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            U[i0*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[(i0+1)*m + j0, col] += 1; U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1; W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1; V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1; W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1; W[k0*n + (i0+1), col] += 1
            col += 1
            
            U[i0*m + j0, col] += 1; U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1; W[(k0+1)*n + i0, col] += 1
            col += 1
            
            U[(i0+1)*m + j0, col] += 1; U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            U[i0*m + (j0+1), col] += 1; U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1; V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1
            
            return col
        
        def add_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0 + di)*m + (j0 + dj), col] += 1
                    V[(j0 + dj)*p + k0, col] += 1
                    W[k0*n + (i0 + di), col] += 1
                    col += 1
            return col
        
        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x2x1(0, 0, 4)
        add_schoolbook_2x2x1(0, 2, 4)
        
        return U, V, W

    U36, V36, W36 = get_solution_rank36()
    if verify_solution(U36, V36, W36):
        U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
        if verify_solution(U_snap, V_snap, W_snap):
            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        else:
            return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    # Fallback: schoolbook
    rank = nm * mp
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
