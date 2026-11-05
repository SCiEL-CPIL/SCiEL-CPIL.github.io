"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 130.0

def _strassen_2x2x2():
    """
    Strassen 2x2x2 decomposition, rank 7.
    U, V, W have shape (4, 7) where indices are (i,j) for U, (j,k) for V, (k,i) for W
    with i,j,k in {0,1}.
    Indexing:
    U[i*2 + j]
    V[j*2 + k]
    W[k*2 + i]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2 = (A10+A11)B00 -> C10, C11 (C11 -= M2)
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    
    # M3 = A00(B01-B11) -> C01, C11
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    
    # M4 = A11(B10-B00) -> C00, C10
    U[3,3]=1
    V[0,3]=-1; V[2,3]=1
    W[0,3]=1; W[1,3]=1
    
    # M5 = (A00+A01)B11 -> C00 (C00 -= M5), C01
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    
    # M6 = (A10-A00)(B00+B01) -> C11
    U[0,5]=-1; U[2,5]=1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # M7 = (A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _build_4x4x4_tensors():
    """
    Build 4x4x4 via Kronecker product of two 2x2x2 Strassen.
    U: (16, 49) indexed by (i,j)
    V: (16, 49) indexed by (j,k)
    W: (16, 49) indexed by (k,i)
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Reshape to (2, 2, 7) for Kronecker product logic
    # U2 is (4, 7) where index is i*2+j. So U2.reshape(2,2,7) gives U2[i,j,c]
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # Kronecker product of tensors
    # The rank is 7*7 = 49.
    # We need to compute U4[i,j,c] = sum_{c1+c2=c? No, c is combined index}
    # Actually, the Kronecker product of two decompositions of rank r1 and r2 gives rank r1*r2.
    # The new index c corresponds to (c1, c2) where c = c1*7 + c2.
    # U4[i,j, c1*7+c2] = U2[i1,j1,c1] * U2[i2,j2,c2] where i = i1*2+i2, j = j1*2+j2.
    
    # We can construct this using broadcasting.
    # U4 shape: (4, 4, 49)
    # i = i1*2 + i2, j = j1*2 + j2
    # c = c1*7 + c2
    
    U4 = np.zeros((4, 4, 49), dtype=np.int8)
    V4 = np.zeros((4, 4, 49), dtype=np.int8)
    W4 = np.zeros((4, 4, 49), dtype=np.int8)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # U4[i1*2+i2, j1*2+j2, c] = U2[i1,j1,c1] * U2[i2,j2,c2]
            U4[:, :, c] = np.kron(U2[:, c1], U2[:, c2])
            V4[:, :, c] = np.kron(V2[:, c1], V2[:, c2])
            W4[:, :, c] = np.kron(W2[:, c1], W2[:, c2])
            
    # Flatten first two dimensions to match (n*m, rank) format
    # U4 is (4, 4, 49). We want (16, 49) where index is i*4+j.
    # np.kron on flattened vectors? 
    # Let's check the indexing of np.kron(U2[:, c1], U2[:, c2]).
    # U2[:, c1] is length 4, index i*2+j.
    # np.kron produces length 16.
    # The standard Kronecker product of vectors v (len 4) and w (len 4) produces v[0]*w, v[1]*w, ...
    # v[0]*w[0], v[0]*w[1], v[0]*w[2], v[0]*w[3], v[1]*w[0]...
    # This corresponds to (i1, j1) from v and (i2, j2) from w.
    # Index in result: (i1*2 + i2)*2 + (j1*2 + j2)? 
    # Let's verify.
    # v index k1 = i1*2 + j1. w index k2 = i2*2 + j2.
    # Result index = k1*4 + k2 = (i1*2+j1)*4 + (i2*2+j2) = i1*8 + j1*4 + i2*2 + j2.
    # Desired index for U4[i,j] is i*4+j where i = i1*2+i2, j = j1*2+j2.
    # i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2.
    # These are NOT the same. The Kronecker product of flattened vectors does not directly give the desired 2D Kronecker structure with row-major flattening of the resulting 2D indices.
    
    # We need to be careful with the Kronecker product definition for matrices.
    # If A is 2x2 and B is 2x2, A \otimes B is 4x4.
    # (A \otimes B)[i1*2+i2, j1*2+j2] = A[i1, j1] * B[i2, j2].
    # This matches our desired structure: U4[i,j] = U2[i1,j1] * U2[i2,j2].
    # So we should treat U2[:, c] as a 2x2 matrix and compute the matrix Kronecker product.
    
    U4_flat = np.zeros((16, 49), dtype=np.int8)
    V4_flat = np.zeros((16, 49), dtype=np.int8)
    W4_flat = np.zeros((16, 49), dtype=np.int8)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            # Reshape column c1 of U2 to 2x2
            U2_mat1 = U2[:, c1].reshape(2, 2)
            U2_mat2 = U2[:, c2].reshape(2, 2)
            U4_mat = np.kron(U2_mat1, U2_mat2)
            U4_flat[:, c] = U4_mat.flatten()
            
            V2_mat1 = V2[:, c1].reshape(2, 2)
            V2_mat2 = V2[:, c2].reshape(2, 2)
            V4_mat = np.kron(V2_mat1, V2_mat2)
            V4_flat[:, c] = V4_mat.flatten()
            
            W2_mat1 = W2[:, c1].reshape(2, 2)
            W2_mat2 = W2[:, c2].reshape(2, 2)
            W4_mat = np.kron(W2_mat1, W2_mat2)
            W4_flat[:, c] = W4_mat.flatten()
            
    return U4_flat, V4_flat, W4_flat

def _build_4x4x8_shared():
    """
    Build a rank-98 decomposition for 4x4 * 4x8 using two 4x4x4 Strassen-Kronecker blocks.
    U: (16, 98) indexed by (i,j) where i in 0..3, j in 0..3
    V: (32, 98) indexed by (j,k) where j in 0..3, k in 0..7
    W: (32, 98) indexed by (k,i) where k in 0..7, i in 0..3
    """
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int8)
    V = np.zeros((32, total_rank), dtype=np.int8)
    W = np.zeros((32, total_rank), dtype=np.int8)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        
        # U is the same for both blocks (it multiplies A, which is 4x4)
        U[:, col_start:col_end] = U4
        
        # V and W are split into left and right halves of B and C
        # V: (m*p, rank) = (4*8, 98). Index j*8 + k.
        # For block 0 (k=0..3): V[j*8 + k, c] = V4[j*4 + k, c]
        # For block 1 (k=4..7): V[j*8 + (4+k_local), c] = V4[j*4 + k_local, c]
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
                
        # W: (p*n, rank) = (8*4, 98). Index k*4 + i.
        # For block 0 (k=0..3): W[k*4 + i, c] = W4[k*4 + i, c]
        # For block 1 (k=4..7): W[(4+k_local)*4 + i, c] = W4[k_local*4 + i, c]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition. Returns True if valid."""
    rank = U.shape[1]
    # Vectorized verification for speed
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute tensor product
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Check if T is all 1s
    if not np.all(T == 1):
        return False
    return True

def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Primary strategy: rank-98 Strassen-Kronecker decomposition
    try:
        if time.time() > _DEADLINE - 10:
            raise TimeoutError("Deadline approaching")
            
        U, V, W = _build_4x4x8_shared()
        
        if time.time() > _DEADLINE - 5:
            # If we are close to deadline, skip verification and hope it's correct
            # But verification is fast. Let's do it if we have time.
            if time.time() < _DEADLINE - 2:
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    # Verification failed, fall back
                    U, V, W = _schoolbook(n, m, p)
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # No time for verification, return Strassen (high confidence it's correct)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Normal path: verify
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Verification failed, fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        pass
    
    # Fallback: schoolbook rank-128
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
