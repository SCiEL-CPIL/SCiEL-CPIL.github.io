"""Tensor decomposition for 6x6x6 matrix multiplication."""
import time
import numpy as np

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a00+a11)*(b00+b11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)*b00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = a00*(b01-b11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = a11*(b10-b00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5 = (a00+a01)*b11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (a10-a00)*(b00+b01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (a01-a11)*(b10+b11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _schoolbook_3x3():
    """Return rank-27 schoolbook decomposition for 3x3x3.
    U: (9,27), V: (9,27), W: (9,27)
    U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    U = np.zeros((9, 27), dtype=float)
    V = np.zeros((9, 27), dtype=float)
    W = np.zeros((9, 27), dtype=float)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, c] = 1
                V[j*3+k, c] = 1
                W[k*3+i, c] = 1
                c += 1
    return U, V, W

def _block_decomposition_6x6x6():
    """
    Decompose 6x6x6 using 3x3 blocks of 2x2 matrices.
    Block-level: 3x3 schoolbook (rank 27)
    Intra-block: 2x2 Strassen (rank 7)
    Total rank = 27 * 7 = 189
    
    Index mapping:
    6 = 3 * 2
    Row index in 6x6: i = bi * 2 + li, where bi in {0,1,2}, li in {0,1}
    U row: i*6 + j = (bi*2+li)*6 + (bj*2+lj)
    V row: j*6 + k = (bj*2+lj)*6 + (bk*2+lk)
    W row: k*6 + i = (bk*2+lk)*6 + (bi*2+li)
    """
    n = 6
    sU, sV, sW = _strassen_2x2()  # (4,7) each
    bU, bV, bW = _schoolbook_3x3()  # (9,27) each
    
    rank = 189
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    
    col = 0
    for c_block in range(27):
        for c_strassen in range(7):
            # U part: block-level U * Strassen U
            for bi in range(3):
                for bj in range(3):
                    block_u = bU[bi*3 + bj, c_block]
                    if block_u == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            s_val = sU[li*2 + lj, c_strassen]
                            if s_val == 0:
                                continue
                            u_row = (bi*2 + li) * 6 + (bj*2 + lj)
                            U[u_row, col] += block_u * s_val
            
            # V part: block-level V * Strassen V
            for bj in range(3):
                for bk in range(3):
                    block_v = bV[bj*3 + bk, c_block]
                    if block_v == 0:
                        continue
                    for lj in range(2):
                        for lk in range(2):
                            s_val = sV[lj*2 + lk, c_strassen]
                            if s_val == 0:
                                continue
                            v_row = (bj*2 + lj) * 6 + (bk*2 + lk)
                            V[v_row, col] += block_v * s_val
            
            # W part: block-level W * Strassen W
            for bk in range(3):
                for bi in range(3):
                    block_w = bW[bk*3 + bi, c_block]
                    if block_w == 0:
                        continue
                    for lk in range(2):
                        for li in range(2):
                            s_val = sW[lk*2 + li, c_strassen]
                            if s_val == 0:
                                continue
                            w_row = (bk*2 + lk) * 6 + (bi*2 + li)
                            W[w_row, col] += block_w * s_val
            
            col += 1
    
    return U, V, W

def _verify_fast(U, V, W, n=6):
    """Fast verification using NumPy einsum."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape to 3D
    U3 = U.reshape(n, n, -1)  # (i, j, rank)
    V3 = V.reshape(n, n, -1)  # (j, k, rank)
    W3 = W.reshape(n, n, -1)  # (k, i, rank)
    
    # Compute sum over rank: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: result[i,k] = sum_{j,c} U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # But we need to check result[i,k] == delta[i,k] for the MM tensor
    # Actually the MM tensor T[i,j,k] = delta[i,k] means:
    # For each (i,j,k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta[i,k] if... 
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i)
    # The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if i==k else 0?
    # No - for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # So T[i,j,k] = delta[i,k]... no that's not right either.
    
    # Let me re-read: "U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]"
    # The MM tensor has entries T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Actually no. The MM tensor T satisfies:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that's the structure tensor. Let me think again.
    
    # The structure tensor of matrix multiplication MM_{n,m,p} has:
    # T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}
    # So T_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL i,j,k
    
    # Let's verify:
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    # result should be all 1s
    if not np.allclose(result, np.ones((n, n, n)), atol=1e-9):
        # Find where it fails
        max_err = np.max(np.abs(result - 1.0))
        return False, max_err
    return True, 0.0

def _schoolbook_6x6x6():
    """Schoolbook rank-216 decomposition."""
    n = 6
    rank = 216
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*6+j, c] = 1
                V[j*6+k, c] = 1
                W[k*6+i, c] = 1
                c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = 6, 6, 6
    
    # Primary: block decomposition rank 189
    try:
        if time.time() < deadline - 5.0:
            U, V, W = _block_decomposition_6x6x6()
            ok, err = _verify_fast(U, V, W, n)
            if ok:
                return {'U': U, 'V': V, 'W': W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank 216
    U, V, W = _schoolbook_6x6x6()
    return {'U': U, 'V': V, 'W': W}

# EVOLVE-BLOCK-END
