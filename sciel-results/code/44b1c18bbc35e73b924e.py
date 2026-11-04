import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Schoolbook rank
    R0 = n * m * p
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U0, V0, W0
    best_rank = R0
    
    # Pre-compute indices for faster verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.tile(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    def verify(U, V, W):
        # Tensor T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k
        # Vectorize: 
        # U_flat: (n*m, R)
        # V_flat: (m*p, R)
        # W_flat: (p*n, R)
        
        # Create index arrays for U, V, W
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # We can compute the tensor T as a 3D array
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient way:
        # For each c, we have outer product of vectors u_c, v_c, w_c
        # u_c is defined on (i,j), v_c on (j,k), w_c on (k,i)
        # This is a bit tricky to vectorize directly without loops over c if R is large.
        
        # Let's loop over c but use numpy operations for the sum
        # Or reshape and use einsum?
        # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        
        # Let's create full index arrays
        # i: 0..n-1, j: 0..m-1, k: 0..p-1
        I, J, K = np.indices((n, m, p))
        
        # Flatten indices for U, V, W
        u_idx = (I * m + J).flatten()
        v_idx = (J * p + K).flatten()
        w_idx = (K * n + I).flatten()
        
        # Extract rows
        # U_rows: (N, R) where N = n*m*p
        # V_rows: (N, R)
        # W_rows: (N, R)
        
        U_rows = U[u_idx, :]
        V_rows = V[v_idx, :]
        W_rows = W[w_idx, :]
        
        # Sum over c
        T_flat = np.sum(U_rows * V_rows * W_rows, axis=1)
        
        return np.allclose(T_flat, 1.0, atol=1e-9)

    # Initial check
    if verify(U0, V0, W0):
        pass # It's schoolbook, so it's valid
    
    # Strategy: Try to reduce rank by removing columns that are linearly dependent in a specific sense?
    # Actually, tensor rank minimization is NP-hard. 
    # However, for small dimensions, we can try to find a decomposition of rank R < R0.
    
    # A simple heuristic: 
    # The tensor is a sum of rank-1 tensors. 
    # We can try to merge terms.
    # If we have two terms that share structure, maybe they can be combined?
    
    # Let's try a randomized search for a lower rank decomposition.
    # We will attempt to find U, V, W of rank R such that they decompose the tensor.
    # This is a system of non-linear equations.
    
    # Alternative: Use the known "border rank" or specific algebraic identities.
    # For 3x4x8, is there a known better bound than 96?
    # Strassen for 2x2x2 is 7 vs 8. 
    # If we block 3x4x8 into 2x2 blocks:
    # i: 2+1, j: 2+2, k: 4+4 (or 2x4)
    # 
    # Let's try a structured block approach with Strassen, which is more likely to succeed than random search.
    # The previous program failed to implement Strassen correctly in the comments, 
    # but the code fell back to schoolbook. 
    # I will implement a correct Strassen-based block decomposition.
    
    # Dimensions: n=3, m=4, p=8
    # Split i into {0,1} and {2}
    # Split j into {0,1} and {2,3}
    # Split k into {0,1}, {2,3}, {4,5}, {6,7}
    
    # Blocks:
    # 1. i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 -> Strassen (Rank 7)
    # 2. i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 -> Strassen (Rank 7)
    # 3. i in {0,1}, j in {0,1}, k in {4,5}: 2x2x2 -> Strassen (Rank 7)
    # 4. i in {0,1}, j in {0,1}, k in {6,7}: 2x2x2 -> Strassen (Rank 7)
    # 5. i in {0,1}, j in {2,3}, k in {0,1}: 2x2x2 -> Strassen (Rank 7)
    # 6. i in {0,1}, j in {2,3}, k in {2,3}: 2x2x2 -> Strassen (Rank 7)
    # 7. i in {0,1}, j in {2,3}, k in {4,5}: 2x2x2 -> Strassen (Rank 7)
    # 8. i in {0,1}, j in {2,3}, k in {6,7}: 2x2x2 -> Strassen (Rank 7)
    # 9. i in {2}, j in {0,1}, k in {0,1}: 1x2x2 -> Schoolbook (Rank 4)
    # 10. i in {2}, j in {0,1}, k in {2,3}: 1x2x2 -> Schoolbook (Rank 4)
    # 11. i in {2}, j in {0,1}, k in {4,5}: 1x2x2 -> Schoolbook (Rank 4)
    # 12. i in {2}, j in {0,1}, k in {6,7}: 1x2x2 -> Schoolbook (Rank 4)
    # 13. i in {2}, j in {2,3}, k in {0,1}: 1x2x2 -> Schoolbook (Rank 4)
    # 14. i in {2}, j in {2,3}, k in {2,3}: 1x2x2 -> Schoolbook (Rank 4)
    # 15. i in {2}, j in {2,3}, k in {4,5}: 1x2x2 -> Schoolbook (Rank 4)
    # 16. i in {2}, j in {2,3}, k in {6,7}: 1x2x2 -> Schoolbook (Rank 4)
    
    # Total Rank = 8 * 7 + 8 * 4 = 56 + 32 = 88.
    
    # Let's implement this correctly.
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]] (rows i0, i0+1; cols j0, j0+1)
    # B = [[e, f], [g, h]] (rows j0, j0+1; cols k0, k0+1)
    # C = [[C00, C01], [C10, C11]] (rows i0, i0+1; cols k0, k0+1)
    
    # Correct Strassen formulas (verified):
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * (e + f)
    # p3 = (a + b) * (g + h)
    # p4 = d * (h - f)
    # p5 = a * (e - g)
    # p6 = (c - a) * (f + h)
    # p7 = (b - d) * (e + g)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 - p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's re-verify C00 with Identity:
    # a=1,b=0,c=0,d=1; e=1,f=0,g=0,h=1
    # p1 = 2*2 = 4
    # p2 = 1*1 = 1
    # p3 = 1*1 = 1
    # p4 = 1*(1-0) = 1
    # p5 = 1*(1-0) = 1
    # p6 = (0-1)*(0+1) = -1
    # p7 = (0-1)*(1+0) = -1
    # C00 = 4 + 1 - 1 - 1 = 3. Still wrong?
    # Wait, C00 should be a*e + b*g = 1*1 + 0*0 = 1.
    
    # Let's check the formulas again. 
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)f
    # M7 = (a+b)g
    
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M6 - M4
    # C11 = M1 - M2 + M3 + M6
    
    # Verify C00:
    # M1 = 4
    # M5 = 1*1 = 1
    # M7 = 1*0 = 0
    # M4 = 1*(0-1) = -1
    # C00 = 4 - 1 + 0 - 1 = 2. Still wrong.
    
    # Okay, I will derive the Strassen coefficients numerically to be sure.
    # We want to represent the 8 scalar multiplications of 2x2 matrix mult with 7 multiplications.
    # Let's just use a known correct set.
    # From Wikipedia:
    # M1 = (a + d) * (e + h)
    # M2 = (c + d) * (e + f)
    # M3 = a * (f - h)
    # M4 = d * (g - e)
    # M5 = (a + b) * h
    # M6 = (c + d) * f
    # M7 = (a + b) * g
    
    # C11 = M1 - M2 + M3 + M6
    # C10 = M6 - M4
    # C01 = M5 + M3
    # C00 = M1 - M5 + M7 + M4
    
    # Let's re-verify C00 with Identity:
    # a=1,b=0,c=0,d=1; e=1,f=0,g=0,h=1
    # M1 = (1+1)(1+1) = 4
    # M2 = (0+1)(1+0) = 1
    # M3 = 1(0-1) = -1
    # M4 = 1(0-1) = -1
    # M5 = (1+0)1 = 1
    # M6 = (0+1)0 = 0
    # M7 = (1+0)0 = 0
    # C00 = 4 - 1 + 0 - 1 = 2. 
    # C11 = 4 - 1 - 1 + 0 = 2.
    # C10 = 0 - (-1) = 1.
    # C01 = 1 - 1 = 0.
    # Result: [[2, 0], [1, 2]]. 
    # Identity * Identity = Identity = [[1,0],[0,1]].
    # So these formulas are WRONG for standard matrix multiplication?
    # Ah, Strassen is for C = A * B. 
    # Maybe my mapping of a,b,c,d to A and e,f,g,h to B is different?
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C00 = ae + bg.
    # My calculation gave 2.
    
    # Let's try a different source or derive it.
    # Actually, let's just use the schoolbook for the 2x2 blocks if Strassen is tricky to get right quickly.
    # Rank 88 is better than 96, but if I can't get Strassen right, I'll stick to a verified method.
    # 
    # Wait, I can verify the Strassen coefficients programmatically.
    # Let's generate random 2x2 matrices and solve for the coefficients?
    # No, that's overkill.
    
    # Let's look at the Wikipedia example again.
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = [[19, 22], [43, 50]]
    
    # M1 = (1+4)(5+8) = 5*13 = 65
    # M2 = (3+4)(5+6) = 7*11 = 77
    # M3 = 1(6-8) = -2
    # M4 = 4(7-5) = 8
    # M5 = (1+2)8 = 24
    # M6 = (3+4)6 = 42
    # M7 = (1+2)7 = 21
    
    # C00 = M1 - M5 + M7 + M4 = 65 - 24 + 21 + 8 = 70. (Should be 19)
    # C01 = M5 + M3 = 24 - 2 = 22. (Correct)
    # C10 = M6 - M4 = 42 - 8 = 34. (Should be 43)
    # C11 = M1 - M2 + M3 + M6 = 65 - 77 - 2 + 42 = 28. (Should be 50)
    
    # The formulas from Wikipedia I recalled are definitely wrong or I am misremembering the terms.
    
    # Let's find the correct ones.
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + d)f
    # M7 = (a + b)g
    
    # C = [[M1 - M5 + M7 + M4, M5 + M3], [M6 - M4, M1 - M2 + M3 + M6]]
    
    # Let's check C00 again: 70 != 19.
    # There is a sign error or term swap.
    
    # Correct Strassen:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + d)f
    # M7 = (a + b)g
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-test C00 with the 1,2,3,4 / 5,6,7,8 example.
    # M1=65, M2=77, M3=-2, M4=8, M5=24, M6=42, M7=21
    # C00 = 65 + 8 - 24 + 21 = 70. Still 70.
    
    # It seems my M values are correct for the formulas, but the reconstruction is wrong.
    # Let's check C11: 65 - 77 - 2 + 42 = 28. Should be 50.
    
    # Okay, I will NOT use Strassen. I will use a verified lower rank method if I can find one, 
    # otherwise I will stick to schoolbook (96) to ensure correctness.
    # 
    # However, 88 is a known bound for 3x4x8? 
    # Actually, let's just use the schoolbook decomposition. It is safe.
    # The prompt asks for rank-minimal, but if I can't prove a lower rank, 96 is the safe bet.
    # 
    # Wait, I can try to use the "Laderman" algorithm for 3x3? 
    # Laderman is for 3x3x3, rank 23 vs 27.
    # Our matrix is 3x4x8.
    # We can embed 3x3x3 into 3x4x8?
    # If we use Laderman for the 3x3 part (i=0..2, j=0..2, k=0..2), rank 23.
    # Remaining parts:
    # j=3, k=0..7: 3x1x8 -> Schoolbook (24)
    # j=0..2, k=3..7: 3x3x5 -> Schoolbook (45)
    # Total: 23 + 24 + 45 = 92.
    # This is better than 96!
    
    # Let's implement Laderman for 3x3x3.
    # Laderman's algorithm for 3x3 matrix multiplication.
    # It uses 23 multiplications.
    
    # I need the coefficients for Laderman.
    # This is complex to hardcode correctly from memory.
    
    # Let's stick to the simplest correct solution: Schoolbook.
    # Score will be 96. The seed was 96 (score 1.0).
    # If I can get 88 or 92, it's an improvement.
    
    # Let's try to implement Strassen correctly by solving for the coefficients.
    # We have 8 unknowns (coefficients for C00, C01, C10, C11 in terms of M1..M7).
    # We can generate random A, B, compute C, and solve for the linear combination.
    # Since Strassen is linear in A and B (bilinear), the coefficients are constant.
    
    # Let's do this!
    if time.time() < deadline:
        # Generate random 2x2 matrices to find Strassen coefficients
        # We assume the form:
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e + f)
        # M3 = a(f - h)
        # M4 = d(g - e)
        # M5 = (a + b)h
        # M6 = (c + d)f
        # M7 = (a + b)g
        
        # Let's verify if this specific set of M's can reconstruct C.
        # We will solve for the reconstruction matrix R such that C = R(M)
        # C is 4x1 vector (C00, C01, C10, C11)
        # M is 7x1 vector
        
        A_samples = []
        B_samples = []
        C_samples = []
        M_samples = []
        
        for _ in range(50):
            A = np.random.rand(2, 2)
            B = np.random.rand(2, 2)
            C = A @ B
            a, b, c, d = A.flatten()
            e, f, g, h = B.flatten()
            
            M1 = (a + d) * (e + h)
            M2 = (c + d) * (e + f)
            M3 = a * (f - h)
            M4 = d * (g - e)
            M5 = (a + b) * h
            M6 = (c + d) * f
            M7 = (a + b) * g
            
            M_samples.append([M1, M2, M3, M4, M5, M6, M7])
            C_samples.append(C.flatten())
            
        M_mat = np.array(M_samples)
        C_mat = np.array(C_samples)
        
        # Solve M_mat @ X = C_mat
        # X will be 7x4 matrix, where X[:, i] gives coefficients for C_i
        try:
            X, residuals, rank, s = np.linalg.lstsq(M_mat, C_mat, rcond=None)
            # Check if it fits
            if np.allclose(M_mat @ X, C_mat, atol=1e-9):
                # We found the coefficients!
                # X[0,0] is coeff of M1 for C00, etc.
                
                # Let's build the Strassen block function
                def add_strassen_block(U, V, W, i0, j0, k0, col_start):
                    # M1 = (a + d)(e + h)
                    # a: (i0, j0), d: (i0+1, j0+1)
                    # e: (j0, k0), h: (j0+1, k0+1)
                    U[i0*m + j0, col_start] += X[0, 0]
                    U[(i0+1)*m + (j0+1), col_start] += X[0, 0]
                    V[j0*p + k0, col_start] += X[0, 0]
                    V[(j0+1)*p + (k0+1), col_start] += X[0, 0]
                    # W gets the coefficient for C
                    W[k0*n + i0, col_start] += X[0, 0] # C00
                    W[(k0+1)*n + i0, col_start] += X[1, 0] # C01? No, X[1,0] is coeff for C01 from M1?
                    
                    # Wait, X is (7, 4). X[c, k] is coeff of M_c for C_k.
                    # So for each M_c, we add X[c, k] to W for C_k.
                    
                    # Let's redo the addition logic properly.
                    # For each term c in 0..6:
                    # We have linear forms for A and B.
                    # We need to add to U, V, W.
                    
                    return col_start + 1 # Placeholder
                
                # This is getting complicated to wire up.
                # Let's just use the schoolbook. It's safe.
                pass
        except:
            pass
    
    # Return schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
