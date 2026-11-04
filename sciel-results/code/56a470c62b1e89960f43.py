import time
import numpy as np

START = time.time()
DEADLINE = START + 135.0

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U: (4, 7) - A[i,j] indexed by i*2+j
    V: (4, 7) - B[i,j] indexed by i*2+j  
    W: (4, 7) - C[i,j] indexed by i*2+j
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_2x2x4_rank14():
    """
    Build rank-14 decomposition for 2x2 * 2x4 matrix multiplication.
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0..3}
    
    Uses shared A-side Strassen combinations.
    U: (4, 14) - A[i,j] indexed by i*2+j
    V: (8, 14) - B[j,k] indexed by j*4+k
    W: (8, 14) - C[i,k] indexed by i*4+k
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank = 14
    U = np.zeros((4, rank), dtype=np.int16)
    V = np.zeros((8, rank), dtype=np.int16)
    W = np.zeros((8, rank), dtype=np.int16)
    
    for s in range(7):
        for block in range(2):
            r = s * 2 + block
            # U is the same for both blocks (shared A-side)
            U[:, r] = U2[:, s]
            
            # V: B-side for this Strassen step and block
            # B is 2x4, indexed as B[j, k] where k = block*2 + k_local
            for j in range(2):
                for kl in range(2):
                    kg = block * 2 + kl
                    V[j * 4 + kg, r] = V2[j * 2 + kl, s]
            
            # W: C-side for this Strassen step and block
            # C is 2x4, indexed as C[i, k] where k = block*2 + k_local
            for i in range(2):
                for kl in range(2):
                    kg = block * 2 + kl
                    W[i * 4 + kg, r] = W2[i * 2 + kl, s]
    
    return U, V, W

def _build_4x4x8_rank56():
    """
    Build rank-56 decomposition for 4x4 * 4x8 matrix multiplication.
    Decompose into 4 blocks of 2x2 * 2x4, each with rank 14.
    
    A is 4x4, B is 4x8, C is 4x8.
    Block structure:
    - A blocks: A[i0*2:i0*2+2, j0*2:j0*2+2] for i0,j0 in {0,1}
    - B blocks: B[j0*2:j0*2+2, k0*4:k0*4+4] for j0 in {0,1}, k0 in {0,1}
    - C blocks: C[i0*2:i0*2+2, k0*4:k0*4+4] for i0 in {0,1}, k0 in {0,1}
    
    For each (i0, j0, k0) triple, we compute:
    C[i0*2:i0*2+2, k0*4:k0*4+4] += A[i0*2:i0*2+2, j0*2:j0*2+2] * B[j0*2:j0*2+2, k0*4:k0*4+4]
    
    This is a 2x2 * 2x4 multiplication, rank 14.
    Total: 4 (i0,j0 pairs) * 2 (k0 blocks) * 14 = 112? No...
    
    Wait, let me reconsider. The 4x4 * 4x8 multiplication:
    C[i, k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Block into 2x2 blocks for A and B:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each is 2x2
    
    C = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each is 2x2
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C03 = A00*B03 + A01*B13
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    C13 = A10*B03 + A11*B13
    
    Each Ci_j is a 2x2 matrix = sum of two 2x2*2x2 products.
    
    For each row block i0 (0 or 1) and column block k0 (0..3):
    C[i0*2:i0*2+2, k0*2:k0*2+2] = A[i0*2:i0*2+2, 0:2] * B[0:2, k0*2:k0*2+2] + 
                                     A[i0*2:i0*2+2, 2:4] * B[2:4, k0*2:k0*2+2]
    
    This is a 2x2 * 2x4 multiplication (A-block is 2x2, B-block is 2x4), rank 14.
    Number of such blocks: 2 (i0) * 4 (k0) = 8 blocks.
    Total rank: 8 * 14 = 112. That's worse than 98!
    
    Let me try a different blocking.
    """
    pass

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Two independent 4x4x4 Strassen blocks.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        cs = block * 49
        ce = cs + 49
        U[:, cs:ce] = U4
        
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j*8 + kg, cs:ce] = V4[j*4 + kl, :]
        
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg*4 + i, cs:ce] = W4[kl*4 + i, :]
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _verify(U, V, W):
    """
    Fast verification using einsum on the full tensors.
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    """
    U_np = np.array(U, dtype=np.float64).reshape(4, 4, -1)
    V_np = np.array(V, dtype=np.float64).reshape(4, 8, -1)
    W_np = np.array(W, dtype=np.float64).reshape(8, 4, -1)
    
    res = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_np, optimize=True)
    return np.allclose(res, 1.0, atol=1e-9)

def _schoolbook():
    rank = 128
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                U[i*4+j][col] = 1
                V[j*8+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if _verify(res["U"], res["V"], res["W"]):
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
