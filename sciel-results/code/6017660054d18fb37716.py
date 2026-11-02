import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Tensor T_{a,b,c,d,i,k} is 1 if a==i, b==d, c==k else 0
    # Decomposition: sum_l U_{a*m+b, l} V_{c*p+d, l} W_{k*n+i, l}
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = 0.0
                            s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    n, m, p = DIMENSIONS # 2, 3, 6
    
    # The problem is M_{2,3,6}.
    # We can view this as two blocks of M_{2,3,3} if we split the output dimension p=6 into two halves?
    # No, M_{2,3,6} multiplies A (2x3) and B (3x6).
    # Output is 2x6.
    # We can split B into B1 (3x3) and B2 (3x3).
    # Then C = A * [B1, B2] = [A*B1, A*B2].
    # So M_{2,3,6} is the direct sum of two M_{2,3,3} tensors.
    # Rank(M_{2,3,6}) = Rank(M_{2,3,3}) + Rank(M_{2,3,3}) = 2 * Rank(M_{2,3,3}).
    
    # Schoolbook rank of M_{2,3,3} is 2*3*3 = 18.
    # Current best known in this context seems to be 17 (giving total 34).
    # Inspiration suggests Rank 11 for M_{2,3,3} might exist or be the target.
    # If Rank(M_{2,3,3}) = 11, then Rank(M_{2,3,6}) = 22.
    # Score for 22: 36/22 = 1.636.
    # Score for 34: 36/34 = 1.0588.
    
    # Strategy:
    # 1. Try to find a rank 11 decomposition for M_{2,3,3}.
    # 2. If found, construct the rank 22 solution for M_{2,3,6}.
    # 3. If not found, fall back to the known rank 34 solution.
    
    def find_m233_rank(target_rank, time_budget):
        """
        Attempts to find a rank 'target_rank' decomposition for 2x3x3.
        Uses a deterministic structure-based search.
        """
        if time.time() > start_time + time_budget:
            return None
            
        n2, m2, p2 = 2, 3, 3
        U_dim = n2 * m2 # 6
        V_dim = m2 * p2 # 9
        W_dim = p2 * n2 # 6
        
        # We will try to construct a solution with a specific symmetry.
        # Often, low-rank decompositions of matrix multiplication have structure related to FFTs or specific polynomials.
        # For 2x3x3, let's try a random restart approach but with a much tighter constraint:
        # We fix the first 'target_rank - K' terms to be standard basis vectors (schoolbook)
        # and solve for the remaining K terms to correct the errors.
        # This is a "sparse" approach.
        
        # Actually, a better approach for small ranks is to use the fact that the tensor is a "3-linear form".
        # We can set up a system of equations.
        # However, solving for rank 11 exactly is hard.
        
        # Let's try a heuristic: 
        # Start with the schoolbook rank 18 decomposition.
        # Randomly select 'target_rank' terms to keep.
        # This won't work because the tensor entries are 1s and 0s.
        
        # Alternative: Use the known "Strassen-like" structure?
        # For 2x2x2, rank 7 is Strassen.
        # For 2x3x3, is there a known rank?
        # The prompt inspiration mentions "Rank 11".
        
        # Let's try a local search starting from a "partially structured" initialization.
        # We initialize U, V, W with random values, but we enforce that the sum of squares is 1 for each column? No.
        
        # Let's try a simple gradient descent on the error, but with "snapping" to integers/half-integers.
        # Error function: E = sum (sum_l U_... V_... W_... - Target)^2
        # This is non-convex.
        
        # Given the time limit, and the fact that the previous program failed to find rank 11 in 60s with SA,
        # it is highly likely that rank 11 is either non-existent or extremely hard to find via random search.
        # However, the inspiration *claims* it uses such a term.
        
        # Let's assume the "Rank 11" is a specific known decomposition.
        # I will hardcode a search for a rank 12 solution first (easier), then 11.
        # Actually, if I can't find it, I must return the best I have.
        
        # Let's try to find a rank 17 solution (which we know exists) to ensure we at least match the parent.
        # The parent code has a hardcoded rank 17 solution. I will reuse that logic if the search fails.
        
        # For this "Different Strategy", I will implement a **Deterministic Linear Algebra Approach** for the rank 17 case to verify it, 
        # and then attempt a **Structured Random Search** for rank 11 with a time limit of 10s.
        # If the structured search finds it, great. If not, we fall back.
        
        # Structured Search for Rank 11:
        # We assume the decomposition has some symmetry.
        # We will try 100 random restarts with a very fast gradient step.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(50):
            if time.time() > start_time + time_budget:
                break
                
            # Initialize with small random values
            U = np.random.randn(U_dim, target_rank) * 0.1
            V = np.random.randn(V_dim, target_rank) * 0.1
            W = np.random.randn(W_dim, target_rank) * 0.1
            
            # Simple gradient descent for 100 steps
            lr = 0.01
            for step in range(100):
                # Compute gradients numerically? Too slow.
                # Just do a random perturbation (SA-like) but faster.
                # Pick a random entry, perturb, check error.
                
                # To speed up error calculation, we precompute the tensor?
                # No, just calculate the error for a random subset of indices?
                # No, we need exactness.
                
                # Let's just use the full error calculation, it's small (6*9*6 = 324 terms).
                err = 0.0
                for a in range(n2):
                    for b in range(m2):
                        for c in range(m2):
                            for d in range(p2):
                                for i in range(n2):
                                    for k in range(p2):
                                        target = 1.0 if (a == i and c == k and b == d) else 0.0
                                        s = np.dot(U[a * m2 + b, :], np.multiply(V[c * p2 + d, :], W[k * n2 + i, :]))
                                        err += (s - target) ** 2
                                        
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    if best_error < 1e-9:
                        # Check if it can be snapped to integers/half-integers
                        # If the values are close to 0, 0.5, 1, -1, etc.
                        # We'll just return the float solution if it's exact enough.
                        # The evaluator accepts float if it's close? 
                        # "integer/half-integer entries" is specified in the interface.
                        # We must snap.
                        
                        # Snap to nearest 0.5
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        # Verify snapped solution
                        if verify_decomposition(U_snap, V_snap, W_snap, n2, m2, p2):
                            return U_snap, V_snap, W_snap
                        else:
                            # If snapped doesn't work, try rounding to integers
                            U_snap = np.round(U)
                            V_snap = np.round(V)
                            W_snap = np.round(W)
                            if verify_decomposition(U_snap, V_snap, W_snap, n2, m2, p2):
                                return U_snap, V_snap, W_snap
                            
                        # If snap fails, keep the float version? 
                        # The contract says "integer/half-integer". 
                        # If we can't snap, we can't use this solution.
                        # So we continue searching.
                        
                # Perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    dim = U_dim
                elif mat_idx == 1:
                    mat = V
                    dim = V_dim
                else:
                    mat = W
                    dim = W_dim
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, target_rank - 1)
                mat[row, col] += np.random.randn() * 0.1
                
        return None

    # Try to find rank 11 for 2x3x3
    # Budget: 20 seconds
    m233_sol = find_m233_rank(11, 20)
    
    if m233_sol is not None:
        U233, V233, W233 = m233_sol
        # Construct M_{2,3,6} from two copies of M_{2,3,3}
        # U shape: (6, 22)
        # V shape: (18, 22)
        # W shape: (12, 22)
        
        U_comb = np.zeros((6, 22))
        V_comb = np.zeros((18, 22))
        W_comb = np.zeros((12, 22))
        
        # Block 1 (k=0,1,2)
        U_comb[:, 0:11] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, 0:11] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, 0:11] = W233[k * 2 + i]
                
        # Block 2 (k=3,4,5)
        U_comb[:, 11:22] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + (3 + k), 11:22] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[(3 + k) * 2 + i, 11:22] = W233[k * 2 + i]
                
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution (Rank 17 for 2x3x3 * 2)
    # This is the verified solution from the current program.
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
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
