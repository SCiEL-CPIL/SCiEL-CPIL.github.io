import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Dimensions
    n, m, p = 2, 3, 6
    
    # We aim to construct a low-rank decomposition directly.
    # Strategy: Use Kronecker product structure.
    # The tensor for M_{n, m, p} can be built from smaller tensors.
    # Specifically, M_{2, 3, 6} can be viewed as M_{2, 3, 3} \otimes M_{1, 1, 2} ? No.
    # But we can use the fact that 6 = 3 * 2.
    # Let's try to find a decomposition of M_{2, 3, 3} with rank R1.
    # And M_{2, 3, 2} with rank R2?
    # Actually, there is a known rank 11 decomposition for M_{2, 3, 3} (Strassen-like or similar).
    # Wait, standard Strassen for 2x2 is rank 7.
    # For 2x3x2, rank is known to be 8? Or 9?
    # Let's look at the inspiration: "Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6".
    # This suggests R(2,3,3) <= 11.
    # If we can get R(2,3,3) = 11, then R(2,3,6) <= 11 * 2 = 22?
    # How does the tensor product work here?
    # T_{2,3,6} corresponds to multiplying A (2x3) and B (3x6).
    # We can split B into two blocks B1, B2 (3x3 each).
    # C = A B1 + A B2.
    # This is not a tensor product in the multiplicative sense of rank (rank(A*B1 + A*B2) is not rank(A*B1)+rank(A*B2) directly in a simple Kronecker way for the decomposition itself, unless we treat the outputs separately).
    # However, if we have a decomposition for M_{2,3,3} of rank 11:
    # (A B1)_{ik} = sum_{l=1}^{11} U_{a,b,l} V_{c,d,l} W_{k,i,l}
    # And another for M_{2,3,3} for B2:
    # (A B2)_{ik} = sum_{l=1}^{11} U'_{a,b,l} V'_{c,d,l} W'_{k,i,l}
    # Then C = A B1 + A B2.
    # The decomposition for the sum is just the concatenation of the two decompositions?
    # No, the output index k for B1 is 0..2, for B2 is 3..5.
    # So W for B1 is non-zero only for k in 0..2. W for B2 is non-zero only for k in 3..5.
    # Since the W vectors have disjoint supports, the rank of the sum is the sum of the ranks.
    # So if we have a rank 11 decomposition for M_{2,3,3}, we can get a rank 22 decomposition for M_{2,3,6}.
    # Is rank 11 known for M_{2,3,3}?
    # Schoolbook for 2x3x3 is 18.
    # Strassen for 2x2x2 is 7.
    # 2x3x3 can be embedded in 3x3x3?
    # Known bounds: R(2,3,3) is between 14 and 17?
    # Wait, the inspiration says "Rank 11 decomposition for 2x3x3".
    # Let me verify if such a thing exists or if I can construct one.
    # Actually, R(2,3,3) = 14 is a common lower bound discussion, but 11 seems very low.
    # Maybe the inspiration meant M_{2,3,2}?
    # R(2,3,2) is 8? (Strassen for 2x2 is 7, add one for the extra column? 8?)
    # If R(2,3,2) = 8, then R(2,3,6) <= 8 * 3 = 24.
    # If R(2,3,3) = 11, then R(2,3,6) <= 22.
    # Let's try to construct a rank 11 for M_{2,3,3}.
    # If I can't find a verified rank 11, I should try to construct a rank 14 or 15 and see if I can beat 34.
    # Even rank 20 would be a huge improvement (36/20 = 1.8).
    
    # Let's try to hardcode a known low-rank solution for M_{2,3,3}.
    # A well-known solution for M_{2,3,3} with rank 14?
    # Or maybe I can use the "Kronecker" trick with M_{2,3,2} and M_{2,3,4}?
    
    # Let's try to generate a random solution with rank 22 and verify? No, random won't work.
    
    # Let's look for a specific structure.
    # For M_{2,3,3}, the schoolbook is 18.
    # We can save 2 multiplications?
    # Let's try to construct a rank 16 solution for M_{2,3,3} using a simple heuristic, then double it to 32? No, 32 < 34.
    # But we want to beat 34.
    
    # Let's try to find a rank 11 solution for M_{2,3,3} online in my "memory".
    # Actually, there is a known rank 11 algorithm for 2x3x3?
    # Coppersmith-Winograd? No, that's asymptotic.
    # Bini's algorithms?
    
    # Let's try a different approach: **Randomized Search for Rank 20**.
    # We initialize U, V, W with random integers in {-1, 0, 1}.
    # We use a local search to minimize the error || T - Sum U_l V_l W_l ||.
    # Since the problem is small (rank 20, dims 6x18x12), we can afford to check many candidates.
    # But "exact" integer solution is hard to hit randomly.
    
    # Better: **Constructive Approach based on M_{2,2,2} (Strassen)**.
    # Strassen rank 7 for 2x2x2.
    # Can we build M_{2,3,6} from Strassen blocks?
    # 2x3 is [2x2, 2x1].
    # 3x6 is [2x6, 1x6].
    # This is getting complex.
    
    # Let's stick to the "Double the M_{2,3,3} solution" idea.
    # I will implement a search for a rank 14 solution for M_{2,3,3} using a greedy method,
    # and if found, double it to rank 28.
    # If I find rank 11, I get 22.
    # If I find rank 10, I get 20.
    
    # However, implementing a robust rank-14 finder for M_{2,3,3} from scratch in 150s is risky.
    
    # Alternative: **Hardcode a known Rank 22 solution?**
    # I don't have one memorized.
    
    # Let's try **Rank 24** via M_{2,3,2} x 3.
    # M_{2,3,2} schoolbook is 12.
    # Can we do better?
    # 2x3 by 3x2.
    # Result 2x2.
    # We can use Strassen for the 2x2 part?
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x2, B2 is 1x2.
    # C = A1 B1 + A2 B2.
    # This doesn't map directly to Strassen.
    
    # Let's try a **Greedy Integer Search for Rank 22** on the full M_{2,3,6} problem.
    # We start with random integer matrices U, V, W of shape (6, 22), (18, 22), (12, 22).
    # We define the error tensor E = T - sum(U_l V_l W_l).
    # We iteratively pick the rank-1 term (u, v, w) that best reduces the Frobenius norm of E,
    # and update E.
    # This is "Greedy Rank-1 Approximation".
    # It often finds exact integer solutions for small tensors if the rank is low enough and we snap to integers.
    
    def greedy_search(rank, n, m, p, max_iters=5000, seed=42):
        np.random.seed(seed)
        
        # Build target tensor T
        # T[a, b, c, d, i, k] = 1 if a==i, c==k, b==d else 0
        # We will work with the flattened vector of T.
        # Size: 2*3*3*6*2*6 = 1296.
        
        def tensor_to_vec(T_6d):
            return T_6d.flatten()
            
        # Construct T
        T = np.zeros((n, m, m, p, n, p))
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        if b == d:
                            T[a, b, c, d, a, c] = 1.0 # Wait, indices:
                            # U index: a*m+b
                            # V index: c*p+d
                            # W index: k*n+i
                            # Condition: a==i, c==k, b==d.
                            # So T[a, b, c, d, i, k] is 1 if i=a, k=c, d=b.
                            # My loop:
                            # a, b, c, d.
                            # i must be a. k must be c. d must be b.
                            # So T[a, b, c, b, a, c] = 1.
                            
        # Let's rebuild T correctly.
        T = np.zeros((n, m, m, p, n, p))
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    # d must be b
                    d = b
                    # i must be a
                    i = a
                    # k must be c
                    k = c
                    T[a, b, c, d, i, k] = 1.0
        
        T_vec = T.flatten()
        E = T_vec.copy()
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for l in range(rank):
            # Find best rank-1 update to E
            # We want to find u, v, w such that <E, u v w> is maximized.
            # This is equivalent to finding the best rank-1 approximation of E.
            # We can use SVD on the reshaped E.
            # E is a 6D tensor. Reshape to (n*m, m*p, p*n)?
            # E[a, b, c, d, i, k]
            # Let U_idx = a*m+b
            # Let V_idx = c*p+d
            # Let W_idx = k*n+i
            # So E can be reshaped to (n*m, m*p, p*n).
            
            E_3d = E.reshape(n*m, m*p, p*n)
            
            # Compute SVD
            try:
                U_svd, S, Vt_svd = np.linalg.svd(E_3d, full_matrices=False)
            except:
                break
                
            # The best rank-1 approximation is U_svd[:, 0] * S[0] * Vt_svd[0, :].
            # But Vt_svd[0, :] is a vector of size m*p * p*n?
            # No, SVD of a 3D tensor A_{i,j,k} -> U_i S_j V_{j,k}?
            # np.linalg.svd on a 2D matrix.
            # We need to reshape E_3d to 2D to use standard SVD?
            # No, SVD is for matrices.
            # We can flatten two modes.
            # Let's reshape E_3d to (n*m, m*p * p*n).
            # This finds the best u and a combined v-w vector.
            # This is not the best rank-1 tensor decomposition, but a good heuristic.
            
            E_2d = E_3d.reshape(n*m, -1)
            U_svd, S, Vt_svd = np.linalg.svd(E_2d, full_matrices=False)
            
            u_best = U_svd[:, 0]
            vw_flat = Vt_svd[0, :]
            
            # Now we need to decompose vw_flat (size m*p * p*n) into v (m*p) and w (p*n).
            # This is again a rank-1 approximation problem.
            # vw_flat is a vector. We can reshape it to (m*p, p*n).
            # And find its best rank-1 approximation?
            # No, vw_flat IS the product v * w?
            # If we assume the best v, w are the top singular vectors of the matrix formed by vw_flat?
            # Wait, if E_{i,j,k} = u_i v_j w_k, then E_{i, (j,k)} = u_i (v_j w_k).
            # So the second factor in the 2D SVD is the vector z_{j,k} = v_j w_k.
            # We need to decompose z into v and w.
            # We can just pick v and w randomly? No.
            # We can use SVD on the matrix Z = z.reshape(m*p, p*n).
            # Z_{j,k} = v_j w_k.
            # This is a rank-1 matrix.
            # So we can just take the top singular vectors of Z.
            
            Z = vw_flat.reshape(m*p, p*n)
            U_z, S_z, Vt_z = np.linalg.svd(Z, full_matrices=False)
            
            v_best = U_z[:, 0] * np.sqrt(S_z[0])
            w_best = Vt_z[0, :] * np.sqrt(S_z[0])
            
            # Snap to integers?
            # The SVD gives real numbers.
            # We want integer/half-integer solutions.
            # Let's try to round them.
            u_int = np.round(u_best * 2) / 2.0
            v_int = np.round(v_best * 2) / 2.0
            w_int = np.round(w_best * 2) / 2.0
            
            # Check if this update reduces error significantly.
            # Calculate new E.
            update = np.outer(np.outer(u_int, v_int), w_int).flatten()
            E_new = E - update
            
            # If error doesn't decrease, try the original float values?
            # But we need integer/half-integer for the final answer.
            # The task says "integer/half-integer entries".
            # So we MUST use rounded values.
            
            # If the rounded update is bad, maybe try rounding to nearest integer?
            # Or try a few nearby integer vectors?
            
            # For now, let's just use the rounded one and see if it works.
            # If the final verification fails, we fail.
            
            U[:, l] = u_int
            V[:, l] = v_int
            W[:, l] = w_int
            
            E = E_new
            
            # Early stopping if E is close to 0
            if np.linalg.norm(E) < 1e-6:
                break
                
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                                if not np.isclose(s, target, atol=1e-9):
                                    return False
        return True

    # Try ranks 20, 21, 22, 23, 24
    for rank in [20, 21, 22, 23, 24]:
        if time.time() - start_time > 130:
            break
            
        U, V, W = greedy_search(rank, n, m, p)
        
        if verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to Rank 34 if greedy fails
    # Reuse the rank 34 generator from the previous program to ensure we always return a valid solution.
    def get_rank34_solution():
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U0, V0, W0 = get_rank34_solution()
    return {
        "U": U0.tolist(),
        "V": V0.tolist(),
        "W": W0.tolist()
    }

if __name__ == "__main__":
    construct()
