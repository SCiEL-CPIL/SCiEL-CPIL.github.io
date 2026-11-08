import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7)  # indices i*2+j
    V shape: (4, 7)  # indices j*2+k
    W shape: (4, 7)  # indices k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    
    return U, V, W

def _strassen_2x2x4():
    """
    Construct 2x2x4 decomposition.
    A is 2x2, B is 2x4, C is 2x4.
    Rank = 14.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = [[0]*14 for _ in range(4)]
    V = [[0]*14 for _ in range(8)]
    W = [[0]*14 for _ in range(8)]
    
    # Left block (r=0..6)
    for r in range(7):
        for i in range(4):
            U[i][r] = U2[i][r]
        V[0][r] = V2[0][r]
        V[1][r] = V2[1][r]
        V[4][r] = V2[2][r]
        V[5][r] = V2[3][r]
        
        W[0][r] = W2[0][r]
        W[1][r] = W2[1][r]
        W[4][r] = W2[2][r]
        W[5][r] = W2[3][r]
        
    # Right block (r=7..13)
    for r in range(7):
        rr = r + 7
        for i in range(4):
            U[i][rr] = U2[i][r]
        V[2][rr] = V2[0][r]
        V[3][rr] = V2[1][r]
        V[6][rr] = V2[2][r]
        V[7][rr] = V2[3][r]
        
        W[2][rr] = W2[0][r]
        W[3][rr] = W2[1][r]
        W[6][rr] = W2[2][r]
        W[7][rr] = W2[3][r]
        
    return U, V, W

def _kronecker_product_4x4x8():
    """
    Construct 4x4x8 using Kronecker product of 2x2x2 (Strassen) and 2x2x4 (Strassen-based).
    Rank = 98.
    """
    U_block, V_block, W_block = _strassen_2x2()
    U_int, V_int, W_int = _strassen_2x2x4()
    
    rank_block = 7
    rank_int = 14
    total_rank = rank_block * rank_int # 98
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(32)]
    W = [[0]*total_rank for _ in range(32)]
    
    for rb in range(rank_block):
        for rs in range(rank_int):
            col = rb * rank_int + rs
            
            # U
            for ib in range(2):
                for i0 in range(2):
                    for jb in range(2):
                        for j0 in range(2):
                            a = ib * 8 + i0 * 4 + jb * 2 + j0
                            ab = ib * 2 + jb
                            as_idx = i0 * 2 + j0
                            U[a][col] = U_block[ab][rb] * U_int[as_idx][rs]
                            
            # V
            for jb in range(2):
                for j0 in range(2):
                    for kb in range(2):
                        for k0 in range(4):
                            b = jb * 16 + j0 * 8 + kb * 4 + k0
                            bb = jb * 2 + kb
                            bs = j0 * 4 + k0
                            V[b][col] = V_block[bb][rb] * V_int[bs][rs]
                            
            # W
            for kb in range(2):
                for k0 in range(4):
                    for ib in range(2):
                        for i0 in range(2):
                            c = kb * 16 + k0 * 4 + ib * 2 + i0
                            cb = kb * 2 + ib
                            cs = k0 * 2 + i0
                            W[c][col] = W_block[cb][rb] * W_int[cs][rs]
                            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(actual - 1.0) > 1e-9:
                    return False
                    
    # Check off-diagonal with random sampling
    np.random.seed(42)
    num_checks = 5000
    for _ in range(num_checks):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        
        if abs(actual - expected) > 1e-9:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x8 using Kronecker product
    try:
        U, V, W = _kronecker_product_4x4x8()
        rank = len(U[0])
        
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
