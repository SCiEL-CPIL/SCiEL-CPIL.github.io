import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n, m, p = 2, 2, 11
    N, M, P = n * m, m * p, p * n  # 4, 22, 22

    # Target Tensor T[i, j, k] = 1 if i,j,k correspond to matmul C[i,k] = sum_j A[i,j]B[j,k]
    # Indices:
    # A is n x m. Index in U: i * m + j
    # B is m x p. Index in V: j * p + k
    # C is n x p. Index in W: k * n + i (Trace convention W[k*n+i, r])
    
    T_target = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def check_validity(U, V, W, rank):
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_decomp, T_target, atol=1e-9)

    # --- Step 1: Construct Strassen-Winograd Decomposition for 2x2x11 ---
    # The 2x2x2 matmul can be done in rank 7 (Strassen).
    # For 2x2x11, we can tile the 2x2 block multiplication.
    # However, a direct Strassen-Winograd for 2x2x11 is known to be rank 39? 
    # Let's verify the standard decomposition.
    # Actually, the rank of 2x2x11 is known to be 39? Or is it lower?
    # Schoolbook is 4*22 = 88? No, schoolbook for 2x2x11 is 2*2*11 = 44 terms.
    # Wait, U is (2*2, R), V is (2*11, R), W is (11*2, R).
    # Schoolbook: R = 44.
    # Strassen-Winograd for 2x2x2 is rank 7.
    # If we tile 2x2x2 into 2x2x11, we have 11/2 blocks? No, 11 is odd.
    # We can pad 11 to 12? 2x2x12 would be 6 blocks of 2x2x2. Rank 6*7 = 42.
    # But we can do better.
    
    # Let's use the known minimal rank for 2x2x11.
    # According to literature, the rank of 2x2xn is 3n-3 for n>=3?
    # For n=2, rank 7 (3*2-3=3? No, 7 is correct for 2x2x2).
    # Formula for 2x2xn: rank is 3n-3? 
    # n=2: 3(2)-3 = 3? No, 2x2x2 is 7.
    # n=3: 3(3)-3 = 6? No, 2x2x3 is likely higher.
    # Actually, the rank of 2x2xn is known to be 3n-3 for n >= 2? 
    # Let's check n=2: 3(2)-3 = 3. But Strassen is 7. So formula is wrong.
    # Correct formula for 2x2xn is likely related to Strassen.
    # For 2x2x11, let's try to construct a rank 39 decomposition.
    # Or even better, let's use a general algorithm to find the rank.
    
    # Since I don't have the explicit 39-term construction memorized perfectly for 11,
    # I will use a different strategy:
    # 1. Start with Schoolbook (Rank 44).
    # 2. Use a "Rank Reduction" heuristic that is more powerful than simple drop.
    #    Instead of dropping one, we try to find a linear combination of 2 terms that equals 1 term?
    #    No, that's not how it works.
    
    # Let's try a different approach: **Randomized Linear Algebra Search**.
    # We know the rank is likely around 39-40.
    # We can generate a large dictionary of random rank-1 tensors and use 
    # **Greedy Forward Selection with Exact Arithmetic Check**.
    
    # But the previous program did this and got 1.0 (likely failed to find exact solution or returned schoolbook).
    # The issue with random search is that the "dictionary" might not contain the right basis vectors.
    
    # **New Structural Idea: Use the Strassen-Winograd construction for 2x2x2 and extend it.**
    # For 2x2x2, the 7 terms are well-known.
    # We can treat the 11 columns of B as 5 blocks of 2 columns + 1 column of 1.
    # Block 1: cols 0,1 -> 2x2x2
    # Block 2: cols 2,3 -> 2x2x2
    # ...
    # Block 5: cols 8,9 -> 2x2x2
    # Block 6: col 10 -> 2x2x1 (Rank 2? Schoolbook is 2)
    
    # Total Rank: 5 * 7 + 2 = 35 + 2 = 37.
    # This is a valid decomposition of rank 37!
    # Let's verify this construction.
    
    # Strassen 2x2x2 terms (U, V, W):
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen 7 terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-h)
    # M7 = (a+b)(g-e)
    #
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # We need to map this to the tensor indices.
    # U vector for term r: entries for A[i,j]
    # V vector for term r: entries for B[j,k]
    # W vector for term r: entries for C[k,i] (Trace: W[k*n+i])
    
    # Let's build the 7 terms for a 2x2x2 block.
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Trace: W[k*2+i]
    # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    
    # Term 1: M1 = (a+d)(e+h)
    # U1: a=1, d=1 -> [1, 0, 0, 1]
    # V1: e=1, h=1 -> [1, 0, 0, 1]
    # W1: C11 += 1, C22 += 1 -> [1, 0, 0, 1]
    
    # Term 2: M2 = (c+d)e
    # U2: c=1, d=1 -> [0, 0, 1, 1]
    # V2: e=1 -> [1, 0, 0, 0]
    # W2: C21 += 1 -> [0, 0, 1, 0] (Wait, C21 is index 2? C[1,0] -> k=0, i=1 -> W[0*2+1] = W[1]. 
    # Let's re-check trace convention.
    # C is n x p. C[i, k].
    # W index is k*n + i.
    # C[0,0] -> k=0, i=0 -> W[0]
    # C[0,1] -> k=1, i=0 -> W[2]
    # C[1,0] -> k=0, i=1 -> W[1]
    # C[1,1] -> k=1, i=1 -> W[3]
    
    # So for 2x2x2:
    # C11 (i=0,k=0) -> W[0]
    # C12 (i=0,k=1) -> W[2]
    # C21 (i=1,k=0) -> W[1]
    # C22 (i=1,k=1) -> W[3]
    
    # Let's re-define W vectors for Strassen terms:
    # M1: C11+=1, C22+=1 -> W[0]=1, W[3]=1 -> [1, 0, 0, 1]
    # M2: C21+=1 -> W[1]=1 -> [0, 1, 0, 0]
    # M3: C12+=1 -> W[2]=1 -> [0, 0, 1, 0]
    # M4: C11-=1, C21+=1 -> W[0]=-1, W[1]=1 -> [-1, 1, 0, 0]
    # M5: C11-=1, C12+=1 -> W[0]=-1, W[2]=1 -> [-1, 0, 1, 0]
    # M6: C22-=1 -> W[3]=-1 -> [0, 0, 0, -1]
    # M7: C11+=1 -> W[0]=1 -> [1, 0, 0, 0]
    
    # Let's verify C11 = M1 - M4 - M5 + M7
    # W contributions to W[0] (C11):
    # M1: 1
    # M4: -1
    # M5: -1
    # M7: 1
    # Sum: 1 - 1 - 1 + 1 = 0? 
    # Wait. C11 = ae + bg.
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # M4 = d(g-e) = dg - de
    # M5 = (a+b)h = ah + bh
    # M7 = (a+b)(g-e) = ag - ae + bg - be
    # M1 - M4 - M5 + M7 = (ae + ah + de + dh) - (dg - de) - (ah + bh) + (ag - ae + bg - be)
    # = ae + ah + de + dh - dg + de - ah - bh + ag - ae + bg - be
    # = ae - ae + ah - ah + de + de + dh - dg - bh + ag + bg - be
    # = 2de + dh - dg - bh + ag + bg
    # This does not look like ae + bg.
    
    # I must have the Strassen formulas slightly wrong or the mapping wrong.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-h)
    # M7 = (a+b)(g-e)
    
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    # Let's re-evaluate C11:
    # M1 - M4 - M5 + M7
    # = (ae+ah+de+dh) - (dg-de) - (ah+bh) + (ag-ae+bg-be)
    # = ae + ah + de + dh - dg + de - ah - bh + ag - ae + bg - be
    # = (ae - ae) + (ah - ah) + (de + de) + dh - dg - bh + ag + bg - be
    # = 2de + dh - dg - bh + ag + bg - be
    # Still not ae+bg.
    
    # Ah, the standard Strassen is for 2x2.
    # Let's check a reliable source or derive it.
    # Actually, let's just use the **Rank 37 construction** idea but verify it numerically.
    # If I construct 5 blocks of 2x2x2 (Rank 7 each) and 1 block of 2x2x1 (Rank 2),
    # and the total rank is 37, I need to ensure the blocks are independent.
    # They are independent because they operate on disjoint sets of columns in B (and thus disjoint entries in C).
    # So the rank is simply the sum of the ranks.
    
    # Let's build the 2x2x2 decomposition correctly.
    # I will use a known correct set of vectors.
    # If I can't derive it quickly, I will use a **different known low-rank decomposition**.
    
    # Alternative: Use **Winograd's algorithm** for 2x2x2 which is also rank 7.
    
    # Let's try to find the 7 vectors programmatically by solving a system?
    # No, that's what the previous program failed at.
    
    # Let's use the **Rank 37** construction explicitly.
    # I will define the 7 terms for a 2x2x2 block.
    # I will use the vectors from a known implementation.
    
    # Vectors for 2x2x2 (Strassen):
    # U (A indices: 0,1,2,3), V (B indices: 0,1,2,3), W (C indices: 0,1,2,3)
    # Note: W index mapping: C[i,k] -> W[k*2+i]
    # C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3
    
    # Let's use the following 7 terms (verified in my head or via standard reference):
    # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  (M1)
    # Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0]  (M2)
    # Term 3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,0,1,0] (M3)
    # Term 4: U=[0,0,0,1], V=[0,0,1,-1], W=[-1,1,0,0] (M4)
    # Term 5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,0,1,0] (M5)
    # Term 6: U=[0,0,1,1], V=[0,1,0,-1], W=[0,0,0,-1] (M6)
    # Term 7: U=[1,1,0,0], V=[0,0,1,-1], W=[1,0,0,0]  (M7)
    
    # Let's check C11 (W[0]):
    # T1: 1
    # T4: -1
    # T5: -1
    # T7: 1
    # Sum: 0. Still 0.
    
    # There is a sign error in my manual derivation or the vectors.
    # Let's assume the **Rank 37** construction is valid and I just need the correct vectors.
    # I will use a **numerical search** to find the 7 vectors for 2x2x2?
    # No, that's too slow.
    
    # **Pivot Strategy:**
    # I will use the **Schoolbook Rank 44** and apply **Gaussian Elimination** to the tensor space.
    # The space of 2x2x11 tensors has dimension 4*22*22 = 1936.
    # The target tensor is one specific vector in this space.
    # The "rank" is the number of rank-1 tensors needed to sum to it.
    # This is NOT a linear dimension problem. The set of rank-1 tensors is a variety, not a subspace.
    # So Gaussian Elimination on the flattened tensor does NOT directly give the rank.
    # The previous program's "Greedy Forward Selection" is essentially trying to find a sparse representation in a dictionary of rank-1 tensors.
    
    # **Why did the previous program fail?**
    # 1. The dictionary of 300 random terms was likely too small or not diverse enough to span the target with few terms.
    # 2. Greedy selection can get stuck in local minima.
    
    # **New Structural Change:**
    # Use a **larger, more structured dictionary**.
    # Include:
    # 1. All Schoolbook terms (44).
    # 2. All "Shifted" Schoolbook terms (e.g., swapping indices).
    # 3. Random terms with larger support.
    # 4. **Crucially:** Use **Orthogonal Matching Pursuit (OMP)** with a **much larger dictionary** (e.g., 1000-2000 terms) to find a sparse solution.
    # 5. **Post-processing:** Once a sparse solution is found, try to **merge** terms or find linear dependencies among the selected terms to reduce rank further.
    
    # Let's implement OMP with a large dictionary.
    
    def generate_dictionary():
        U_dict = []
        V_dict = []
        W_dict = []
        
        # 1. Schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(N)
                    v = np.zeros(M)
                    w = np.zeros(P)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_dict.append(u)
                    V_dict.append(v)
                    W_dict.append(w)
                    
        # 2. Random terms
        rng = np.random.RandomState(42)
        num_random = 500
        for _ in range(num_random):
            u = rng.choice([-1, 0, 1], size=N)
            v = rng.choice([-1, 0, 1], size=M)
            w = rng.choice([-1, 0, 1], size=P)
            # Ensure non-zero
            if np.all(u == 0) or np.all(v == 0) or np.all(w == 0):
                continue
            U_dict.append(u)
            V_dict.append(v)
            W_dict.append(w)
            
        return np.array(U_dict).T, np.array(V_dict).T, np.array(W_dict).T

    U_dict, V_dict, W_dict = generate_dictionary()
    D = U_dict.shape[1]
    
    # Flatten dictionary terms
    A = np.zeros((N * M * P, D))
    for r in range(D):
        t = np.einsum('i,j,k->ijk', U_dict[:, r], V_dict[:, r], W_dict[:, r])
        A[:, r] = t.flatten()
        
    b = T_target.flatten()
    
    # OMP
    S = []
    residual = b.copy()
    max_iter = 60
    tol = 1e-9
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        if np.linalg.norm(residual) < tol:
            break
            
        # Correlation
        corr = np.abs(A.T @ residual)
        for j in S:
            corr[j] = 0
            
        best_j = np.argmax(corr)
        
        if corr[best_j] < 1e-12:
            break
            
        S.append(best_j)
        
        # Solve least squares
        A_S = A[:, S]
        try:
            x_S, _, _, _ = np.linalg.lstsq(A_S, b, rcond=None)
        except:
            break
            
        residual = b - A_S @ x_S
        
        if np.allclose(residual, 0, atol=tol):
            # Found solution
            new_rank = len(S)
            U_new = np.zeros((N, new_rank))
            V_new = np.zeros((M, new_rank))
            W_new = np.zeros((P, new_rank))
            
            for idx, j in enumerate(S):
                U_new[:, idx] = U_dict[:, j] * x_S[idx]
                V_new[:, idx] = V_dict[:, j]
                W_new[:, idx] = W_dict[:, j]
                
            if check_validity(U_new, V_new, W_new, new_rank):
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
                
    # Fallback to Schoolbook
    U = np.zeros((N, 44))
    V = np.zeros((M, 44))
    W = np.zeros((P, 44))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
