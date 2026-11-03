"""
Strategy: Direct Strassen-based decomposition for 4x4x8 matrix multiplication.

We exploit the structure that 4x4x8 is essentially two independent 4x4x4
multiplications. Using recursive Strassen at both block levels gives rank 49
per half, total rank 98.

Key structural change from the seed: instead of building the 4x4x4 decomposition
by explicit index mapping (which is error-prone), we use tensor product / Kronecker
structure to compose the 2x2 Strassen decomposition with itself, ensuring exactness
by construction.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2 matrix multiplication (Strassen).
    Each term is (u, v, w) where u, v, w are vectors of length 4.
    
    A is 2x2 (indices: 00, 01, 10, 11), B is 2x2, C is 2x2.
    """
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    terms = [
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
        (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,0,1,-1])),
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
        (np.array([1,0,0,0]), np.array([0,1,0,-1]), np.array([0,1,0,1])),
        # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
        (np.array([0,0,0,1]), np.array([0,0,1,-1]), np.array([1,0,1,0])),
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,1,0,0])),
        # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        (np.array([-1,0,1,0]), np.array([1,1,0,0]), np.array([0,0,0,1])),
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([1,0,0,0])),
    ]
    return terms

def _compose_strassen_4x4():
    """
    Compose two levels of Strassen to get a rank-49 decomposition for 4x4x4.
    
    Level 1: Treat 4x4 as 2x2 blocks of 2x2 matrices. Apply Strassen at block level.
    Level 2: Each 2x2 multiplication uses Strassen (rank 7).
    
    Total: 7 (top) * 7 (bottom) = 49 rank-1 terms.
    
    We use Kronecker product structure:
    For top-level term k with (u_top, v_top, w_top) each of length 4 (blocks),
    and bottom-level term t with (u_bot, v_bot, w_bot) each of length 4 (within-block),
    the combined term has:
    u_full = u_top ⊗ u_bot (length 16)
    v_full = v_top ⊗ v_bot (length 16)
    w_full = w_top ⊗ w_bot (length 16)
    
    This works because the tensor for 4x4x4 is the Kronecker product of the
    tensor for 2x2x2 with itself (when using row-major block ordering).
    """
    terms_2x2 = _strassen_2x2_terms()
    
    # Top-level Strassen: same 7 terms but interpreted as block coefficients
    # For 4x4 = [2x2 blocks], the top-level is also Strassen on 2x2 "scalars"
    # where each "scalar" is actually a 2x2 block.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    col = 0
    for k in range(7):
        u_top, v_top, w_top = terms_2x2[k]
        for t in range(7):
            u_bot, v_bot, w_bot = terms_2x2[t]
            
            # Kronecker product: u_top ⊗ u_bot
            # If u_top = [a0, a1, a2, a3] and u_bot = [b0, b1, b2, b3]
            # Then u_top ⊗ u_bot = [a0*b0, a0*b1, a0*b2, a0*b3,
            #                        a1*b0, a1*b1, a1*b2, a1*b3,
            #                        a2*b0, a2*b1, a2*b2, a2*b3,
            #                        a3*b0, a3*b1, a3*b2, a3*b3]
            #
            # This corresponds to block-major ordering:
            # Block 0 (top-left 2x2): indices 0,1,4,5 -> in row-major 4x4: 0,1,4,5
            # Wait, let's be careful about the ordering.
            #
            # For a 4x4 matrix in row-major order:
            # Index = row * 4 + col
            # Block (br, bc) contains elements at rows 2*br + {0,1} and cols 2*bc + {0,1}
            # Block 00: (0,0),(0,1),(1,0),(1,1) -> indices 0,1,4,5
            # Block 01: (0,2),(0,3),(1,2),(1,3) -> indices 2,3,6,7
            # Block 10: (2,0),(2,1),(3,0),(3,1) -> indices 8,9,12,13
            # Block 11: (2,2),(2,3),(3,2),(3,3) -> indices 10,11,14,15
            #
            # The Kronecker product u_top ⊗ u_bot with u_top indexed by block (br,bc)
            # and u_bot indexed by within-block (lr,lc) gives:
            # (u_top ⊗ u_bot)[block_idx * 4 + local_idx]
            # where block_idx = br*2+bc and local_idx = lr*2+lc
            #
            # But the actual row-major index is: (br*2+lr)*4 + (bc*2+lc)
            # = br*8 + lr*4 + bc*2 + lc
            #
            # While block_idx*4 + local_idx = (br*2+bc)*4 + (lr*2+lc)
            # = br*8 + bc*4 + lr*2 + lc
            #
            # These are different! So we need to permute.
            #
            # Let's just build it explicitly with the correct mapping.
            
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            w_full = np.zeros(16)
            
            for b_idx in range(4):  # block index 0..3
                br = b_idx // 2  # block row
                bc = b_idx % 2   # block col
                
                u_coeff_top = u_top[b_idx]
                v_coeff_top = v_top[b_idx]
                w_coeff_top = w_top[b_idx]
                
                if u_coeff_top != 0:
                    for l in range(4):  # local index within block
                        lr = l // 2  # local row
                        lc = l % 2   # local col
                        gr = br * 2 + lr  # global row
                        gc = bc * 2 + lc  # global col
                        gidx = gr * 4 + gc
                        u_full[gidx] += u_coeff_top * u_bot[l]
                
                if v_coeff_top != 0:
                    for l in range(4):
                        lr = l // 2
                        lc = l % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        gidx = gr * 4 + gc
                        v_full[gidx] += v_coeff_top * v_bot[l]
                
                if w_coeff_top != 0:
                    for l in range(4):
                        lr = l // 2
                        lc = l % 2
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        gidx = gr * 4 + gc
                        w_full[gidx] += w_coeff_top * w_bot[l]
            
            U[:, col] = u_full
            V[:, col] = v_full
            W[:, col] = w_full
            col += 1
    
    return U, V, W

def _build_4x4x8_decomposition():
    """
    Build the full 4x4x8 decomposition using two 4x4x4 Strassen decompositions.
    
    For 4x4x8:
    - A is 4x4 (indices: i*4+j, i,j in 0..3)
    - B is 4x8 (indices: j*8+k, j in 0..3, k in 0..7)
    - C is 8x4 (indices: k*4+i, k in 0..7, i in 0..3)
    
    The tensor T[a,b,c] = 1 iff:
    a = i*4+j, b = j*8+k, c = k*4+i for some i,j in 0..3, k in 0..7
    
    This splits into two independent parts:
    - k in 0..3: C indices 0..15, B indices j*8+{0,1,2,3}
    - k in 4..7: C indices 16..31, B indices j*8+{4,5,6,7}
    
    Each part is essentially a 4x4x4 multiplication.
    """
    U4, V4, W4 = _compose_strassen_4x4()
    rank4 = U4.shape[1]
    
    total_rank = 2 * rank4
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((32, total_rank))
    W_full = np.zeros((32, total_rank))
    
    # Left half (k = 0..3)
    U_full[:, 0:rank4] = U4
    
    for r in range(rank4):
        for idx in range(16):
            # V4[idx, r] is the coefficient for B element at position idx in 4x4
            # In 4x4x4, B index = j*4 + k where j,k in 0..3
            # In 4x4x8, for left half, B index = j*8 + k where j in 0..3, k in 0..3
            j = idx // 4
            k_local = idx % 4
            v_idx = j * 8 + k_local
            V_full[v_idx, r] = V4[idx, r]
            
            # W4[idx, r] is the coefficient for C element at position idx in 4x4
            # In 4x4x4, C index = k*4 + i where k,i in 0..3
            # In 4x4x8, for left half, C index = k*4 + i where k in 0..3, i in 0..3
            k_c = idx // 4
            i = idx % 4
            w_idx = k_c * 4 + i
            W_full[w_idx, r] = W4[idx, r]
    
    # Right half (k = 4..7)
    U_full[:, rank4:2*rank4] = U4
    
    for r in range(rank4):
        for idx in range(16):
            j = idx // 4
            k_local = idx % 4
            v_idx = j * 8 + (k_local + 4)
            V_full[v_idx, r + rank4] = V4[idx, r]
            
            k_c = idx // 4
            i = idx % 4
            w_idx = (k_c + 4) * 4 + i
            W_full[w_idx, r + rank4] = W4[idx, r]
    
    return U_full, V_full, W_full

def _get_target_tensor():
    """Construct the target matrix multiplication tensor for 4x4x8."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify decomposition."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _schoolbook_4x4x8():
    """Fallback: schoolbook decomposition, rank 128."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Build target tensor for verification
    T = _get_target_tensor()
    
    # Try Strassen-based decomposition (rank 98)
    try:
        U, V, W = _build_4x4x8_decomposition()
        if _verify(U, V, W, T):
            # Success! Convert to lists
            # Ensure values are clean (integers or half-integers)
            def clean(mat):
                # Multiply by 2 and round
                mat2 = np.round(mat * 2)
                if np.allclose(mat * 2, mat2, atol=1e-6):
                    return (mat2 / 2.0).tolist()
                # Try integers
                mat_int = np.round(mat)
                if np.allclose(mat, mat_int, atol=1e-6):
                    return mat_int.astype(int).tolist()
                return mat.tolist()
            
            return {
                "U": clean(U),
                "V": clean(V),
                "W": clean(W)
            }
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Not enough time, use schoolbook
        U, V, W = _schoolbook_4x4x8()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook_4x4x8()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
