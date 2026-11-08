import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _strassen_2x2x2():
    """
    Return U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) - rows indexed by (i,j) where i in {0,1}, j in {0,1}
    V: (4, 7) - rows indexed by (j,k) where j in {0,1}, k in {0,1}
    W: (4, 7) - rows indexed by (k,i) where k in {0,1}, i in {0,1}
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M2 + M3 + M5
    # where
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indexing for U: row = i*2 + j (i, j in {0,1})
    # Indexing for V: row = j*2 + k (j, k in {0,1})
    # Indexing for W: row = k*2 + i (k, i in {0,1})
    
    # M1
    U[0][0] += 1; U[3][0] += 1  # A11, A22
    V[0][0] += 1; V[3][0] += 1  # B11, B22
    W[0][0] += 1; W[2][0] += 1  # C11 += M1, C22 += M1 (W index: k*2+i. C11 is k=0,i=0 -> 0. C22 is k=1,i=1 -> 3? Wait.)
    
    # Let's re-verify W indexing.
    # W[k*n+i, c]. For n=2.
    # C11 corresponds to i=0, k=0. Index = 0*2 + 0 = 0.
    # C12 corresponds to i=0, k=1. Index = 1*2 + 0 = 2.
    # C21 corresponds to i=1, k=0. Index = 0*2 + 1 = 1.
    # C22 corresponds to i=1, k=1. Index = 1*2 + 1 = 3.
    
    # So:
    # W[0] is C11
    # W[1] is C21
    # W[2] is C12
    # W[3] is C22
    
    # Let's redo M1:
    # C11 += M1, C22 += M1
    # W[0][0] += 1
    # W[3][0] += 1
    # My previous code had W[2][0] which is C12. That's wrong.
    
    # Let's restart the assignments carefully.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A: A11(0), A22(3). U[0]++, U[3]++
    # B: B11(0), B22(3). V[0]++, V[3]++
    # C: C11(0), C22(3). W[0]++, W[3]++
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A21 + A22)B11
    # A: A21(2), A22(3). U[2]++, U[3]++
    # B: B11(0). V[0]++
    # C: C12(2), C22(3). W[2]++, W[3]++
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += 1
    
    # M3 = A11(B12 - B22)
    # A: A11(0). U[0]++
    # B: B12(1), B22(3). V[1]++, V[3]-=1
    # C: C21(1), C22(3). W[1]++, W[3]++
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A22(B21 - B11)
    # A: A22(3). U[3]++
    # B: B21(2), B11(0). V[2]++, V[0]-=1
    # C: C11(0) -= M4, C22(3) += M4. Wait.
    # C11 = M1 - M4 + M6 + M7
    # C22 = M1 - M2 + M3 + M5
    # So C11 gets -M4. C22 gets +M4? No, C22 doesn't have M4.
    # Let's check formulas again.
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M2 + M3 + M5
    
    # M4 contribution:
    # C11 -= M4
    # C12 += M4
    # W[0][3] -= 1
    # W[2][3] += 1
    
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] -= 1; W[2][3] += 1
    
    # M5 = (A11 + A12)B22
    # A: A11(0), A12(1). U[0]++, U[1]++
    # B: B22(3). V[3]++
    # C: C21(1) += M5, C22(3) += M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[1][4] += 1; W[3][4] += 1
    
    # M6 = (A21 - A11)(B11 + B12)
    # A: A21(2), A11(0). U[2]++, U[0]-=1
    # B: B11(0), B12(1). V[0]++, V[1]++
    # C: C11(0) += M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += 1
    
    # M7 = (A12 - A22)(B21 + B22)
    # A: A12(1), A22(3). U[1]++, U[3]-=1
    # B: B21(2), B22(3). V[2]++, V[3]++
    # C: C11(0) += M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Return U, V, W for schoolbook decomposition of M_{n,m,p}.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _embed(U_sub, V_sub, W_sub, i_off, j_off, k_off, n, m, p):
    """
    Embed sub-matrices into global matrices.
    Sub-matrices correspond to A[i_off:i_off+ns, j_off:j_off+ms] * B[j_off:j_off+ms, k_off:k_off+ps]
    contributing to C[i_off:i_off+ns, k_off:k_off+ps].
    ns = len(U_sub)//ms? No, U_sub is (ns*ms, rank).
    We need ns, ms, ps to calculate shapes.
    U_sub rows: (i,j) with i in [0,ns), j in [0,ms). Global row: (i_off+i)*m + (j_off+j).
    V_sub rows: (j,k) with j in [0,ms), k in [0,ps). Global row: (j_off+j)*p + (k_off+k).
    W_sub rows: (k,i) with k in [0,ps), i in [0,ns). Global row: (k_off+k)*n + (i_off+i).
    
    We infer ns, ms, ps from U_sub, V_sub, W_sub shapes?
    U_sub shape (ns*ms, R). V_sub shape (ms*ps, R). W_sub shape (ps*ns, R).
    """
    R = len(U_sub[0])
    ns_ms = len(U_sub)
    ms_ps = len(V_sub)
    ps_ns = len(W_sub)
    
    # Find ms such that ns*ms = ns_ms, ms*ps = ms_ps, ps*ns = ps_ns.
    # Since ns, ms, ps are small integers, we can brute force or just pass them.
    # For safety, let's pass ns, ms, ps explicitly in the recursive calls.
    pass

def _build_recursive(n, m, p, deadline):
    """
    Returns U, V, W for M_{n,m,p} using recursive splitting and Strassen for 2x2x2.
    """
    if time.time() > deadline:
        return None
        
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
        
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
        
    # Split the largest dimension? Or fixed split?
    # Let's try splitting m if m >= 2.
    if m >= 2:
        m1 = m // 2
        m2 = m - m1
        # Split M_{n,m,p} into M_{n,m1,p} and M_{n,m2,p}
        U1, V1, W1 = _build_recursive(n, m1, p, deadline)
        if U1 is None: return None
        U2, V2, W2 = _build_recursive(n, m2, p, deadline)
        if U2 is None: return None
        
        R = len(U1[0])
        U = []
        # U1 rows: (i, j) for j in [0, m1). Global row: i*m + j.
        for i in range(n):
            for j in range(m1):
                U.append(U1[i*m1 + j])
            for j in range(m2):
                U.append(U2[i*m2 + j])
        
        V = []
        # V1 rows: (j, k) for j in [0, m1). Global row: j*p + k.
        for j in range(m1):
            V.append(V1[j*p + k]) # Wait, V1 has rows indexed by local j.
            # Local j in V1 is 0..m1-1. Global j is j_off + j. Here j_off=0.
        for j in range(m2):
            V.append(V2[j*p + k]) # Local j in V2 is 0..m2-1. Global j is m1 + j.
            # Wait, V2 rows are (local_j, local_k). local_j corresponds to global j = m1 + local_j.
            # So we just append them?
            # Yes, because the rows of V are indexed by (j, k).
            # V1 has rows for j in [0, m1). V2 has rows for j in [0, m2).
            # Global V needs rows for j in [0, m1) and [m1, m1+m2).
            # So V = V1_rows + V2_rows?
            # But V2's local j=0 corresponds to global j=m1.
            # The row content of V2[0] is for (j=0, k). We need it at global row m1*p + k.
            # Since we are just appending, the order matters.
            # V1 has m1*p rows. V2 has m2*p rows.
            # Total m*p rows.
            # Is the order correct?
            # V1 rows: j=0..m1-1.
            # V2 rows: j=0..m2-1.
            # If we append V2 after V1, the rows are j=0..m1-1, then j=0..m2-1 (from V2).
            # But global j for V2 should be m1..m1+m2-1.
            # The row index in the matrix V is just an index.
            # V[row, col] = V[row, col].
            # The row index in V corresponds to the flattened (j,k) pair.
            # If we simply concatenate, the row index for the first row of V2 is m1*p.
            # This corresponds to global j = m1*p // p = m1, k = 0.
            # This matches the requirement.
            # So V = V1 + V2 works?
            # Wait, V1 is a list of lists.
            # V1[0] is row (j=0, k=0).
            # V2[0] is row (local j=0, local k=0).
            # When we append V2[0] to V, it becomes V[m1*p].
            # This is correct.
            
        # So V = V1 + V2 is correct for m-split?
        # Let's check U.
        # U1 rows: (i, local_j). local_j in [0, m1).
        # U2 rows: (i, local_j). local_j in [0, m2).
        # Global U rows: (i, j).
        # For fixed i, global j goes 0..m-1.
        # Rows for i=0: j=0..m1-1 (from U1), j=m1..m-1 (from U2).
        # U1 has rows for i=0..n-1, j=0..m1-1.
        # Order in U1: i=0,j=0; i=0,j=1... i=0,j=m1-1; i=1,j=0...
        # We need to interleave them for each i.
        # So U is NOT U1 + U2.
        # We must construct U carefully.
        
        U = []
        for i in range(n):
            for j in range(m1):
                U.append(U1[i*m1 + j])
            for j in range(m2):
                U.append(U2[i*m2 + j])
                
        V = V1 + V2
        W = W1 + W2
        
        return U, V, W
        
    elif n >= 2:
        n1 = n // 2
        n2 = n - n1
        U1, V1, W1 = _build_recursive(n1, m, p, deadline)
        if U1 is None: return None
        U2, V2, W2 = _build_recursive(n2, m, p, deadline)
        if U2 is None: return None
        
        # U1 rows: (local_i, j). local_i in [0, n1).
        # U2 rows: (local_i, j). local_i in [0, n2).
        # Global U rows: (i, j).
        # For fixed j, i goes 0..n-1.
        # U1 has rows for i=0..n1-1. U2 for i=n1..n-1 (local 0..n2-1).
        # Order in U1: i=0,j=0... i=0,j=m-1; i=1,j=0...
        # We need to interleave for each j?
        # No, global row index is i*m + j.
        # For i=0, j=0..m-1.
        # For i=1, j=0..m-1.
        # U1 has rows for i=0..n1-1.
        # So for j=0, we need U1[0] (i=0,j=0), U1[m] (i=1,j=0)... U1[(n1-1)*m] (i=n1-1, j=0).
        # Then U2[0] (i=n1, j=0)...
        # So for each j, we take U1[j], U1[m+j], ..., U2[j], U2[m+j]...
        
        U = []
        for j in range(m):
            for i in range(n1):
                U.append(U1[i*m + j])
            for i in range(n2):
                U.append(U2[i*m + j])
                
        V = V1 + V2
        W = []
        # W1 rows: (k, local_i). local_i in [0, n1).
        # W2 rows: (k, local_i). local_i in [0, n2).
        # Global W rows: (k, i).
        # Index k*n + i.
        # For fixed k, i goes 0..n-1.
        # W1 has rows for i=0..n1-1.
        # So for each k, we take W1[k], W1[n1+k]...?
        # W1 index is k*n1 + local_i.
        # For fixed k, local_i=0..n1-1. Rows: k*n1, k*n1+1...
        # We need to group by k.
        for k in range(p):
            for i in range(n1):
                W.append(W1[k*n1 + i])
            for i in range(n2):
                W.append(W2[k*n2 + i])
                
        return U, V, W
        
    elif p >= 2:
        p1 = p // 2
        p2 = p - p1
        U1, V1, W1 = _build_recursive(n, m, p1, deadline)
        if U1 is None: return None
        U2, V2, W2 = _build_recursive(n, m, p2, deadline)
        if U2 is None: return None
        
        U = U1 + U2
        
        V = []
        # V1 rows: (j, local_k). local_k in [0, p1).
        # V2 rows: (j, local_k). local_k in [0, p2).
        # Global V rows: (j, k).
        # Index j*p + k.
        # For fixed j, k goes 0..p-1.
        # V1 has rows for k=0..p1-1.
        for j in range(m):
            for k in range(p1):
                V.append(V1[j*p1 + k])
            for k in range(p2):
                V.append(V2[j*p2 + k])
                
        W = []
        # W1 rows: (local_k, i). local_k in [0, p1).
        # W2 rows: (local_k, i). local_k in [0, p2).
        # Global W rows: (k, i).
        # Index k*n + i.
        # For fixed i, k goes 0..p-1.
        for i in range(n):
            for k in range(p1):
                W.append(W1[k*n + i])
            for k in range(p2):
                W.append(W2[k*n + i])
                
        return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 100s safety margin
    
    n, m, p = DIMENSIONS
    
    result = _build_recursive(n, m, p, deadline)
    if result is None:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
        
    U, V, W = result
    return {"U": U, "V": V, "W": W}
