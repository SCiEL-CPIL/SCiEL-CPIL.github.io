"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_start_time = time.time()
_DEADLINE = 135  # seconds, safety margin before 150s kill

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition exactly reconstructs the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The tensor T is defined such that T[i, j, k] = 1 if i==j and k==i (wait, no).
    Matrix multiplication C = A * B.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    The tensor M is the structure tensor of matrix multiplication.
    M[i, j, k] = 1 if the product A[i, j] * B[j, k] contributes to C[i, k].
    In the decomposition U, V, W:
    sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = M[i, j, k]
    
    Here indices are:
    i in 0..n-1 (row of A, row of C)
    j in 0..m-1 (col of A, row of B)
    k in 0..p-1 (col of B, col of C)
    
    U shape: (n*m, rank) -> U[i*m + j, r]
    V shape: (m*p, rank) -> V[j*p + k, r]
    W shape: (p*n, rank) -> W[k*n + i, r]
    
    We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i, j, k) is a valid index for multiplication?
    Actually, the tensor for matrix multiplication M_{n,m,p} has entries:
    T[i, j, k] = 1 if the basis element e_i of A, e_j of B, e_k of C interact.
    Specifically, the structure constants are 1 for the triple (i, j, k) corresponding to C[i,k] += A[i,j]B[j,k].
    So T[i, j, k] = 1 for all valid i, j, k.
    Wait, the "tensor" is usually defined on the basis vectors.
    The multiplication map is linear in A and B, bilinear.
    The tensor T is in V* \otimes V* \otimes V.
    T(e_i^A, e_j^B) = e_k^C where k is determined by the multiplication rule?
    No, the output is a matrix C. C[i, k] = sum_j A[i, j] B[j, k].
    So for each fixed (i, k), the term is sum_j A[i, j] B[j, k].
    The tensor decomposition represents the bilinear map.
    The "target" tensor has entries 1 at positions corresponding to the standard multiplication.
    Let's define the target tensor T_target of shape (n, m, p).
    T_target[i, j, k] = 1 for all i, j, k.
    Because A[i, j] * B[j, k] contributes to C[i, k] with coefficient 1.
    And the reconstruction is:
    Reconstructed[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We want Reconstructed[i, j, k] == 1 for all i, j, k.
    """
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    T = np.einsum('ijr,jkr,ki r->ijk'.replace(' ', ''), U3, V3, W3)
    
    # Target is all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _count_errors(U, V, W, n, m, p):
    """Count how many entries are not exactly 1 (within tolerance)."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Error is 1 if T[i,j,k] != 1
    # Since we are searching for exact integer/half-integer solutions, 
    # we can check exact equality or close equality.
    errors = np.sum(np.abs(T - 1.0) > 1e-9)
    return int(errors)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook rank is n*m*p = 128.
    # We want to find a lower rank.
    # Known lower bound is around n^3 for square, but for rectangular it's complex.
    # Laderman for 4x4 is 20. For 4x4x8, we can view as two 4x4 multiplications?
    # If we split B into two 4x4 blocks, we get 2 * 20 = 40 rank.
    # Can we do better?
    # Let's try to search for a rank R decomposition.
    
    # We will try ranks from a low estimate upwards.
    # The current seed program found 40 (or failed and returned 128?).
    # The prompt says score 1.0000, which usually means it returned the seed (128) or the best known.
    # Wait, the seed program in the prompt returns 128 if it fails to verify the 40-rank one.
    # Let's check if the 40-rank Laderman construction in the seed is valid.
    # The seed's `_build_laderman_4x4x8` just concatenates two 4x4 Laderman decompositions.
    # This should be valid for 4x4x8 if the 4x4x4 Laderman is valid.
    # However, the seed's `_build_laderman_4x4x4` might be flawed.
    
    # Strategy:
    # 1. Try to construct a valid rank-40 solution using the block method (two 4x4x4 Laderman).
    # 2. If that fails, try to find a valid rank-40 (or lower) solution via discrete search.
    
    # First, let's implement a correct Laderman 4x4x4 decomposition.
    # Laderman's algorithm for 4x4 matrix multiplication uses 20 multiplications.
    # Reference: Laderman, "A noncommutative algorithm for multiplying 2x2 matrices" (1976) is for 2x2.
    # For 4x4, it's often derived from 2x2 blocks.
    # Actually, the rank of 4x4 matrix multiplication is known to be 20 (Laderman).
    # Let's try to build it correctly.
    
    # A = [[A00, A01], [A10, A11]] where Aij are 2x2.
    # B = [[B00, B01], [B10, B11]] where Bij are 2x2.
    # C = A*B.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # This is just 8 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done in 7 multiplications (Strassen).
    # 8 * 7 = 56? No, that's not optimal.
    # Laderman's 20 products are for the whole 4x4.
    
    # Let's use the known 20-product formula for 4x4.
    # From "A 20-product algorithm for 4x4 matrix multiplication"
    
    # Since deriving this correctly from scratch is error-prone, 
    # I will use a Discrete Search approach for a smaller rank, e.g., 30-40.
    # But search space is huge.
    
    # Alternative: Use the fact that 4x4x8 can be decomposed into 4x4x4 + 4x4x4.
    # If I can find a valid 4x4x4 rank-20 decomposition, I can just duplicate it.
    # Let's try to find a valid 4x4x4 rank-20 decomposition using discrete search as well, 
    # or use a known correct one.
    
    # Known correct 2x2 Strassen (rank 7):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We can build a 4x4x4 decomposition using 2x2 blocks and Strassen.
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B = [[B00, B01], [B10, B11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # This requires 8 multiplications of 2x2 matrices.
    # If we use Strassen for each, we need 8 * 7 = 56 scalar multiplications.
    # This is worse than Laderman's 20.
    # But 56 is a valid upper bound.
    # For 4x4x8, we can split B into two 4x4 matrices B1, B2.
    # C1 = A*B1, C2 = A*B2.
    # Total rank = 56 + 56 = 112.
    # This is better than 128 (schoolbook).
    
    # Can we do better?
    # Laderman's 20 for 4x4x4 is the best known.
    # If I can implement Laderman's 20 correctly, I get rank 40 for 4x4x8.
    
    # Let's try to implement Laderman's 20 correctly.
    # Source: "Laderman's algorithm"
    # The 20 products are:
    # P1 = A00*B00
    # P2 = (A00+A01)*(B10-B11)
    # P3 = (A00+A10)*(B01-B00)
    # P4 = A01*B11
    # P5 = A10*B00
    # P6 = A11*B11
    # P7 = (A00+A11)*(B00+B11)
    # P8 = (A01+A11)*(B10-B00)
    # P9 = (A10+A11)*(B01-B00)
    # P10 = (A00+A01+A10+A11)*(B00+B01+B10+B11)
    # P11 = (A00-A01+A10-A11)*(B00-B01+B10-B11)
    # P12 = (A00+A01-A10-A11)*(B00-B01-B10+B11)
    # P13 = (A00-A01-A10+A11)*(B00+B01-B10-B11)
    # P14 = (A01+A10)*(B00+B11)
    # P15 = (A00+A11)*(B01+B10)
    # P16 = (A01+A11)*(B00+B10)
    # P17 = (A10+A11)*(B01+B00)
    # P18 = (A00+A01)*(B11+B10)
    # P19 = (A00+A10)*(B01+B11)
    # P20 = (A01+A10)*(B00+B11)
    
    # This matches the seed program's list.
    # The issue might be in the reconstruction formulas (C coefficients).
    # Let's verify the reconstruction for C00.
    # C00 = A00*B00 + A01*B10
    # In terms of P's:
    # P1 = A00*B00
    # P5 = A10*B00 (no)
    # P2 = (A00+A01)*(B10-B11) = A00*B10 - A00*B11 + A01*B10 - A01*B11
    # P4 = A01*B11
    # P3 = (A00+A10)*(B01-B00) = A00*B01 - A00*B00 + A10*B01 - A10*B00
    # P7 = (A00+A11)*(B00+B11)
    
    # This is getting complicated to verify manually.
    # I will use the discrete search to find a valid decomposition for a small rank, 
    # say starting from 30, and see if it can find one.
    # But 30-40 rank with 16xR matrices is a huge search space.
    
    # Better approach:
    # 1. Construct the 112-rank solution (2x 56-rank Strassen-based 4x4x4).
    # 2. Try to reduce the rank by merging/optimizing?
    # 3. Or just return the 112-rank solution, which is better than 128.
    # 4. Or try to find a valid 40-rank solution by fixing the Laderman coefficients via search.
    
    # Let's build the 112-rank solution first as a safe fallback.
    # Strassen 2x2x2 (rank 7)
    def _strassen_2x2():
        # Returns U, V, W for 2x2x2 multiplication.
        # n=2, m=2, p=2.
        # U shape (4, 7), V shape (4, 7), W shape (4, 7).
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        # A00 is (0,0), A11 is (1,1)
        # B00 is (0,0), B11 is (1,1)
        # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1 (Wait, Strassen formulas)
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # Indices:
        # A: 00->0, 01->1, 10->2, 11->3
        # B: 00->0, 01->1, 10->2, 11->3
        # C: 00->0, 01->1, 10->2, 11->3
        
        # M1: A00+A11, B00+B11
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
        
        # M2: A10+A11, B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A00, B10-B11
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A11, B01-B00
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5: A00+A01, B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = -1
        
        # M6: A10-A00, B00+B01
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        # M7: A01-A11, B10+B11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[2, 6] = 1
        
        return U, V, W
    
    # Build 4x4x4 using 2x2 blocks and Strassen.
    def _build_4x4x4_strassen():
        # n=4, m=4, p=4
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # We need to compute 8 products of 2x2 matrices.
        # Each product uses 7 scalar multiplications.
        # Total rank = 56.
        
        U4 = np.zeros((16, 56))
        V4 = np.zeros((16, 56))
        W4 = np.zeros((16, 56))
        
        U2, V2, W2 = _strassen_2x2()
        
        # Helper to place a 2x2 Strassen decomposition into the 4x4 structure.
        # Block (br, bc) of A corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1.
        # Flattened index for A: i*4 + j.
        # Block (br, bc) covers i in 2*br..2*br+1, j in 2*bc..2*bc+1.
        
        # Product 1: A00 * B00 -> contributes to C00
        # A00 is block (0,0), B00 is block (0,0), C00 is block (0,0)
        # We need to map the 2x2 Strassen indices to the 4x4 indices.
        
        def place_block(U_src, V_src, W_src, a_block, b_block, c_block, offset):
            # a_block: (br, bc) for A
            # b_block: (br, bc) for B
            # c_block: (br, bc) for C
            # offset: rank offset
            
            # A indices
            for r in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        if U_src[i2*2+j2, r] != 0:
                            i = 2*a_block[0] + i2
                            j = 2*a_block[1] + j2
                            U4[i*4+j, offset+r] += U_src[i2*2+j2, r]
                            
            # B indices
            for r in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        if V_src[i2*2+j2, r] != 0:
                            i = 2*b_block[0] + i2
                            j = 2*b_block[1] + j2
                            V4[i*4+j, offset+r] += V_src[i2*2+j2, r]
                            
            # C indices
            for r in range(7):
                for i2 in range(2):
                    for j2 in range(2):
                        if W_src[i2*2+j2, r] != 0:
                            i = 2*c_block[0] + i2
                            j = 2*c_block[1] + j2
                            W4[i*4+j, offset+r] += W_src[i2*2+j2, r]
                            
        # 8 products
        # 1. A00 * B00 -> C00
        place_block(U2, V2, W2, (0,0), (0,0), (0,0), 0)
        # 2. A01 * B10 -> C00
        place_block(U2, V2, W2, (0,1), (1,0), (0,0), 7)
        # 3. A00 * B01 -> C01
        place_block(U2, V2, W2, (0,0), (0,1), (0,1), 14)
        # 4. A01 * B11 -> C01
        place_block(U2, V2, W2, (0,1), (1,1), (0,1), 21)
        # 5. A10 * B00 -> C10
        place_block(U2, V2, W2, (1,0), (0,0), (1,0), 28)
        # 6. A11 * B10 -> C10
        place_block(U2, V2, W2, (1,1), (1,0), (1,0), 35)
        # 7. A10 * B01 -> C11
        place_block(U2, V2, W2, (1,0), (0,1), (1,1), 42)
        # 8. A11 * B11 -> C11
        place_block(U2, V2, W2, (1,1), (1,1), (1,1), 49)
        
        return U4, V4, W4
        
    # Build 4x4x8 using two 4x4x4 Strassen decompositions.
    def _build_4x4x8_strassen():
        U4, V4, W4 = _build_4x4x4_strassen()
        
        U8 = np.zeros((16, 112))
        V8 = np.zeros((32, 112))
        W8 = np.zeros((32, 112))
        
        # First 4x4x4: B columns 0-3, C columns 0-3
        U8[:, 0:56] = U4
        V8[0:16, 0:56] = V4
        W8[0:16, 0:56] = W4
        
        # Second 4x4x4: B columns 4-7, C columns 4-7
        U8[:, 56:112] = U4
        V8[16:32, 56:112] = V4
        W8[16:32, 56:112] = W4
        
        return U8, V8, W8
        
    # Get the 112-rank solution
    U112, V112, W112 = _build_4x4x8_strassen()
    
    # Verify it
    if _verify(U112, V112, W112, n, m, p):
        best_U, best_V, best_W = U112, V112, W112
        best_rank = 112
    else:
        # Fallback to schoolbook 128
        U128 = np.zeros((16, 128))
        V128 = np.zeros((32, 128))
        W128 = np.zeros((32, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U128[i*m+j, col] = 1.0
                    V128[j*p+k, col] = 1.0
                    W128[k*n+i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U128, V128, W128
        best_rank = 128
        
    # Now, try to improve the rank using discrete search.
    # We will try to find a valid decomposition with rank < best_rank.
    # Start from best_rank - 2 and go down to 30.
    # For each rank, initialize with random integers and perform coordinate descent.
    
    for target_rank in range(best_rank - 2, 29, -2):
        if time.time() > _DEADLINE - 10:
            break
            
        # Try a few seeds
        for seed in range(5):
            if time.time() > _DEADLINE - 5:
                break
                
            np.random.seed(seed)
            random.seed(seed)
            
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
            
            # Coordinate descent
            errors = _count_errors(U, V, W, n, m, p)
            prev_errors = errors + 1
            no_improve_count = 0
            
            while errors > 0 and no_improve_count < 50 and time.time() < _DEADLINE - 5:
                # Pick a random entry from U, V, or W
                factor = random.choice(['U', 'V', 'W'])
                if factor == 'U':
                    idx = random.randint(0, n*m - 1)
                    r = random.randint(0, target_rank - 1)
                    # Try nearby integer values
                    for val in range(-5, 6):
                        if val == int(U[idx, r]):
                            continue
                        old_val = U[idx, r]
                        U[idx, r] = float(val)
                        new_errors = _count_errors(U, V, W, n, m, p)
                        if new_errors < errors:
                            errors = new_errors
                            no_improve_count = 0
                        else:
                            U[idx, r] = old_val
                            
                elif factor == 'V':
                    idx = random.randint(0, m*p - 1)
                    r = random.randint(0, target_rank - 1)
                    for val in range(-5, 6):
                        if val == int(V[idx, r]):
                            continue
                        old_val = V[idx, r]
                        V[idx, r] = float(val)
                        new_errors = _count_errors(U, V, W, n, m, p)
                        if new_errors < errors:
                            errors = new_errors
                            no_improve_count = 0
                        else:
                            V[idx, r] = old_val
                            
                else: # W
                    idx = random.randint(0, p*n - 1)
                    r = random.randint(0, target_rank - 1)
                    for val in range(-5, 6):
                        if val == int(W[idx, r]):
                            continue
                        old_val = W[idx, r]
                        W[idx, r] = float(val)
                        new_errors = _count_errors(U, V, W, n, m, p)
                        if new_errors < errors:
                            errors = new_errors
                            no_improve_count = 0
                        else:
                            W[idx, r] = old_val
                            
                if errors == prev_errors:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    prev_errors = errors
                    
                if errors == 0:
                    break
                    
            if errors == 0:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
