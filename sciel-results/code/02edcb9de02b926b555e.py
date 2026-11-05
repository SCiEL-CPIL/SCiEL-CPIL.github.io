import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    def get_4x4x4_factors():
        # Returns list of 28 tuples (u, v, w) where u, v, w are length 16
        # u[i*4+j], v[j*4+k], w[k*4+i]
        strassen_factors = get_strassen_factors()
        factors_4x4 = []
        
        for uf, vf, wf in strassen_factors:
            # uf, vf are length 4: [00, 01, 10, 11]
            # We need to expand the product of two 2x2 matrices defined by uf, vf
            # Let P be the 2x2 matrix from uf, Q from vf
            # P = [[uf[0], uf[1]], [uf[2], uf[3]]]
            # Q = [[vf[0], vf[1]], [vf[2], vf[3]]]
            # Product R = P Q
            # R[0,0] = P[0,0]Q[0,0] + P[0,1]Q[1,0]
            # R[0,1] = P[0,0]Q[0,1] + P[0,1]Q[1,1]
            # R[1,0] = P[1,0]Q[0,0] + P[1,1]Q[1,0]
            # R[1,1] = P[1,0]Q[0,1] + P[1,1]Q[1,1]
            
            # We expand each term of R into scalar multiplications of A and B blocks
            # A blocks: A00, A01, A10, A11 (indices 0,1,2,3 in local 2x2 grid)
            # B blocks: B00, B01, B10, B11
            
            # Term 1: R[0,0]
            # P[0,0]Q[0,0]: uf[0] * vf[0]
            # This contributes to C[0,0] (local)
            # But wait, the Strassen output wf tells us how R contributes to C.
            # wf is length 4: [C00, C01, C10, C11]
            
            # So for each scalar multiplication in the 2x2 product, we need to
            # add a factor to the global list.
            
            # Let's iterate over the 4 scalar products in the 2x2 matrix multiplication
            # P[0,0] * Q[0,0]
            # P[0,1] * Q[1,0]
            # P[0,0] * Q[0,1]
            # P[0,1] * Q[1,1]
            # P[1,0] * Q[0,0]
            # P[1,1] * Q[1,0]
            # P[1,0] * Q[0,1]
            # P[1,1] * Q[1,1]
            
            # For each product P[r,c] * Q[c,d], it contributes to R[r,d]
            # R[r,d] contributes to C via wf[r*2+d]
            
            # The scalar product P[r,c] * Q[c,d] is:
            # (sum_{i,j} uf[i*2+j] A[i,j]) * (sum_{k,l} vf[k*2+l] B[k,l])
            # This is a sum of 4 scalar multiplications A[i,j] * B[k,l]
            
            # So for each Strassen factor, we generate 4 (from 2x2 product structure) * 4 (from scalar expansion) = 16 factors?
            # No.
            # The "factor" in the tensor decomposition is a rank-1 term: u * v * w.
            # Here u is a vector of coefficients for A, v for B, w for C.
            
            # Let's trace one scalar multiplication: A[i,j] * B[k,l].
            # This term appears in the product P[r,c] * Q[c,d] with coefficient uf[i*2+j] * vf[k*2+l].
            # This product contributes to R[r,d] with that coefficient.
            # R[r,d] contributes to C[local_i, local_k] with coefficient wf[local_i*2+local_k].
            
            # So for a fixed Strassen factor (uf, vf, wf), and fixed i,j,k,l,
            # the coefficient for A[i,j] is uf[i*2+j].
            # The coefficient for B[k,l] is vf[k*2+l].
            # The coefficient for C[local_i, local_k] is sum over r,c,d such that r*2+d corresponds to local_i, local_k?
            # No, R is a 2x2 matrix. R[r,d] is the element at row r, col d.
            # C is the 2x2 result. C[a,b] = sum_{r,d} wf[a*2+b] * R[r,d] ?
            # No, Strassen says C = f(wf, M1..M7).
            # Specifically, C[0,0] = M1 - M4 + M5 - M7 (example).
            # So C[a,b] = sum_{k=1..7} wf_k[a,b] * M_k.
            # And M_k is the matrix product P_k Q_k.
            
            # So for a specific scalar term A[i,j] B[k,l] inside M_k:
            # It contributes to M_k[r,d] with coeff uf[i*2+j] * vf[k*2+l] * (1 if i*2+j maps to P[r,c] and k*2+l maps to Q[c,d]? No.)
            
            # Let's be precise.
            # P is 2x2. P[r,c] = sum_{i,j} uf[i*2+j] A[i,j] ?
            # No, uf is length 4. uf[0]=A00 coeff, uf[1]=A01 coeff, etc.
            # So P[r,c] is NOT a single A element.
            # Strassen defines P_k as a 2x2 matrix where each entry is a linear combination of A blocks.
            # Wait, in standard Strassen for 2x2, A00 is a scalar.
            # Here A00 is a 2x2 block.
            # So P_k is a 2x2 matrix of 2x2 blocks.
            # P_k = [[P_k00, P_k01], [P_k10, P_k11]]
            # P_k00 = uf[0]*A00 + uf[1]*A01 + ... ?
            # No, the Strassen factors uf, vf are for the 2x2 scalar case.
            # When we block it, A00 becomes a 2x2 matrix.
            # So P_k00 is a 2x2 matrix: uf[0] * A00.
            # P_k01 is a 2x2 matrix: uf[1] * A01.
            
            # So P_k is a 2x2 block matrix.
            # Q_k is a 2x2 block matrix.
            # M_k = P_k Q_k is a 2x2 block matrix.
            # M_k[r,d] = sum_{c=0..1} P_k[r,c] Q_k[c,d].
            # P_k[r,c] is a 2x2 matrix: uf[r*2+c] * A_{r,c} (if we map indices correctly).
            # Actually, uf is [1,0,0,1] for M1.
            # P_1 = [[A00, 0], [0, A11]].
            # So P_1[0,0] = A00. P_1[0,1] = 0.
            
            # So P_k[r,c] = uf[r*2+c] * A_{r,c}. (Assuming A_{r,c} is the block at row r, col c).
            # Similarly Q_k[c,d] = vf[c*2+d] * B_{c,d}.
            
            # Then M_k[r,d] = sum_{c} uf[r*2+c] vf[c*2+d] (A_{r,c} B_{c,d}).
            
            # A_{r,c} B_{c,d} is a 2x2 matrix product.
            # Let's expand A_{r,c} B_{c,d} into scalar terms.
            # A_{r,c} = [[A_{r,c,00}, A_{r,c,01}], [A_{r,c,10}, A_{r,c,11}]]
            # B_{c,d} = [[B_{c,d,00}, B_{c,d,01}], [B_{c,d,10}, B_{c,d,11}]]
            
            # Product R_{r,d} = A_{r,c} B_{c,d}
            # R_{r,d}[i,j] = sum_{t=0..1} A_{r,c}[i,t] B_{c,d}[t,j]
            
            # So for each c, we have 4 scalar products.
            # Total scalar products for M_k is 2 (c=0,1) * 4 = 8.
            # For 7 Strassen factors, total 56 scalar products?
            # But we said rank 28.
            # Ah, the rank of the 2x2 matrix multiplication is 4.
            # So each block product A_{r,c} B_{c,d} has rank 4.
            # M_k is sum of 2 such block products.
            # Rank(M_k) <= 8.
            # Total rank <= 7 * 8 = 56.
            
            # Why did I think 28?
            # If we used Schoolbook for the 4x4x4 directly, rank is 64.
            # If we use Strassen for the 4x4x4 structure (7 block products), and Schoolbook for each 2x2 block product (rank 4), the total rank is 7 * 2 * 4 = 56?
            # No, M_k = P_k Q_k.
            # P_k, Q_k are 2x2 block matrices.
            # M_k[0,0] = P_k[0,0]Q_k[0,0] + P_k[0,1]Q_k[1,0].
            # Each term is a 2x2 matrix product (rank 4).
            # So M_k[0,0] is sum of 2 rank-4 terms.
            # The tensor decomposition of M_k[0,0] has rank at most 8.
            # So total rank is 7 * 4 (entries in M_k) * ?
            
            # Actually, the rank of the 4x4x4 tensor is the sum of ranks of the components.
            # If we just use Schoolbook for 4x4x4, rank is 64.
            # If we use Strassen, we replace the 16 block products of the 2x2 structure?
            # No, Strassen replaces 8 block products (for 2x2) with 7.
            # For 4x4, we have 4x4=16 scalar products in schoolbook.
            # With Strassen on 2x2 blocks:
            # We have 7 "big" products.
            # Each "big" product is a 2x2 matrix multiplication of 2x2 blocks.
            # Rank of 2x2 matrix multiplication is 4.
            # So each "big" product has rank 4 * 4 (entries) = 16?
            # No. The rank of the tensor for 2x2 matrix multiplication is 4.
            # So each 2x2 block product has rank 4.
            # M_k has 4 entries. Each entry is a sum of 2 block products.
            # This is getting complicated.
            
            # Let's stick to the simplest improvement:
            # Partition 5 = 4 + 1.
            # Use Schoolbook for the 4x4x4 block (Rank 64).
            # Total Rank = 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125?
            # Wait.
            # 4x4x4: 64.
            # 4x4x1: 16.
            # 4x1x4: 16.
            # 4x1x1: 4.
            # 1x4x4: 16.
            # 1x4x1: 4.
            # 1x1x4: 4.
            # 1x1x1: 1.
            # Sum = 64 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 125.
            # This is not better than 117.
            
            # So we MUST use a better decomposition for 4x4x4.
            # Rank 49 (Strassen-Strassen) or Rank 28 (Strassen-Schoolbook)?
            # If Rank 28 is possible, Total = 28 + 47 = 75.
            # Score = 125/75 = 1.66.
            
            # Let's assume Rank 28 is achievable.
            # How to generate it?
            # We need to output 28 factors for the 4x4x4 block.
            
            return factors_4x4

    # ... rest of the code
