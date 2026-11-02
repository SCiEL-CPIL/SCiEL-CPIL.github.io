"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And compare with the matrix multiplication tensor
    
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where W is indexed as W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # True tensor value: 1 (since C[i,k] includes A[i,j]*B[j,k])
                true_val = 1.0
                
                # Decomposed value
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                
                if abs(dec_val - true_val) > 1e-9:
                    return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Use 2 blocks of 2x3x3, each with rank 11
    # Total rank = 2 * 11 = 22
    # Block 1: C[:,0:3] = A * B[:,0:3]
    # Block 2: C[:,3:6] = A * B[:,3:6]
    
    def build_2x3x3_decomposition():
        """Build a rank-11 decomposition for 2x3x3 multiplication.
        
        A is 2x3 (flat index i*3+j), B is 3x3 (flat index j*3+k), C is 2x3.
        W is indexed as W[k*2+i] (n=2).
        Returns list of 11 terms, each with 'u', 'v', 'w' dicts.
        """
        terms = []
        
        # We need: for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
        #   sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
        
        # W[k*n + i] where n=2:
        # C[0,0]: k=0,i=0 -> W[0]
        # C[0,1]: k=1,i=0 -> W[2]
        # C[0,2]: k=2,i=0 -> W[4]
        # C[1,0]: k=0,i=1 -> W[1]
        # C[1,1]: k=1,i=1 -> W[3]
        # C[1,2]: k=2,i=1 -> W[5]
        
        # M1 = (A00+A11)(B00+B11) -> should give C[0,0] and C[1,1]
        # U: A[0,0]=idx0, A[1,1]=idx4
        # V: B[0,0]=idx0, B[1,1]=idx4
        # W: C[0,0]=W[0], C[1,1]=W[3]
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 3: 1}})
        
        # M2 = (A10+A11)*B00 -> should give C[1,0] and C[1,1]
        # U: A[1,0]=idx3, A[1,1]=idx4
        # V: B[0,0]=idx0
        # W: C[1,0]=W[1], C[1,1]=W[3]
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {1: 1, 3: -1}})
        
        # M3 = A00*(B01-B11) -> should give C[0,1] and C[1,1]
        # U: A[0,0]=idx0
        # V: B[0,1]=idx1, B[1,1]=idx4
        # W: C[0,1]=W[2], C[1,1]=W[3]
        terms.append({'u': {0: 1}, 'v': {1: 1, 4: -1}, 'w': {2: 1, 3: 1}})
        
        # M4 = A11*(B10-B00) -> should give C[0,0] and C[1,0]
        # U: A[1,1]=idx4
        # V: B[1,0]=idx3, B[0,0]=idx0
        # W: C[0,0]=W[0], C[1,0]=W[1]
        terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {0: 1, 1: 1}})
        
        # M5 = (A00+A01)*B11 -> should give C[0,0] and C[0,1]
        # U: A[0,0]=idx0, A[0,1]=idx1
        # V: B[1,1]=idx4
        # W: C[0,0]=W[0], C[0,1]=W[2]
        terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {0: -1, 2: 1}})
        
        # M6 = (A10-A00)*(B00+B01) -> should give C[1,1]
        # U: A[1,0]=idx3, A[0,0]=idx0
        # V: B[0,0]=idx0, B[0,1]=idx1
        # W: C[1,1]=W[3]
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        
        # M7 = (A01-A11)*(B10+B11) -> should give C[0,0]
        # U: A[0,1]=idx1, A[1,1]=idx4
        # V: B[1,0]=idx3, B[1,1]=idx4
        # W: C[0,0]=W[0]
        terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1}})
        
        # T8: A[0,2]*B[2,0] -> C[0,0]
        terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}})
        # T9: A[0,2]*B[2,1] -> C[0,1]
        terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {2: 1}})
        # T10: A[1,2]*B[2,0] -> C[1,0]
        terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {1: 1}})
        # T11: A[1,2]*B[2,1] -> C[1,1]
        terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {3: 1}})
        
        return terms
    
    def verify_2x3x3(terms):
        """Verify a rank-11 decomposition for 2x3x3."""
        rank = len(terms)
        U = np.zeros((6, rank))
        V = np.zeros((9, rank))
        W = np.zeros((6, rank))
        
        for r, term in enumerate(terms):
            for idx, val in term['u'].items():
                U[idx, r] = val
            for idx, val in term['v'].items():
                V[idx, r] = val
            for idx, val in term['w'].items():
                W[idx, r] = val
        
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    dec_val = 0.0
                    for r in range(rank):
                        dec_val += U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
                    if abs(dec_val - 1.0) > 1e-9:
                        return False, (i, j, k, dec_val)
        return True, None
    
    # Build and verify the 2x3x3 decomposition
    terms_233 = build_2x3x3_decomposition()
    valid, err = verify_2x3x3(terms_233)
    
    if not valid:
        # Fall back to rank-33 if rank-22 doesn't work
        return _construct_rank33()
    
    # Build the full 2x3x6 decomposition using two 2x3x3 blocks
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    for block in range(2):
        for t_idx, term in enumerate(terms_233):
            col = block * 11 + t_idx
            
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            for v_idx, val in term['v'].items():
                j_local = v_idx // 3
                k_local = v_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            
            for w_idx, val in term['w'].items():
                k_local = w_idx // 2
                i_local = w_idx % 2
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        if _verify(U, V, W):
            return result
        else:
            # Fallback to rank-33
            return _construct_rank33()
    
    return result

def _construct_rank33():
    """Fallback rank-33 construction using three 2x3x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
