import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 3, 6
    
    # Strategy: Construct a Rank 30 decomposition for M_{2,3,6} by combining:
    # 1. A Rank 14 decomposition for the 2x3x3 part (using Strassen for 2x2x2 + 6 rank-1 terms).
    # 2. A Rank 9 decomposition for the 2x3x3 part (using Strassen for 2x2x2 + 2 rank-1 terms + 1 clever term).
    # Total Rank = 14 + 9 = 23? No, wait.
    # M_{2,3,6} = M_{2,3,3} \oplus M_{2,3,3}.
    # If we can find a Rank 11 decomposition for M_{2,3,3}, we get Rank 22.
    # The inspiration says "Uses two copies of a Rank 11 decomposition for 2x3x3".
    # Let's try to construct a Rank 11 decomposition for M_{2,3,3}.
    
    # M_{2,3,3}: A(2x3), B(3x3), C(2x3).
    # Split A into A1(2x2) and A2(2x1).
    # Split B into B1(2x3) and B2(1x3).
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2x3.
    # A2 B2 is 2x1x3.
    
    # Rank of 2x2x3:
    # Schoolbook: 12.
    # Strassen for 2x2 part: 7 mults for 2x2x2.
    # The 3rd column of B is just a vector.
    # Let B = [B_left(2x2) b(2x1)].
    # A1 B1 = A1 B_left + A1 b.
    # A1 B_left (2x2x2) -> Rank 7 (Strassen).
    # A1 b (2x2x1) -> Rank 2?
    # C_col2 = A1 * b.
    # C[0,2] = A[0,0]b[0] + A[0,1]b[1].
    # C[1,2] = A[1,0]b[0] + A[1,1]b[1].
    # This is 2x2x1. Rank 4?
    # Actually, we can reuse the Strassen products?
    # Strassen products M1..M7 are linear combinations of A entries times linear combinations of B entries.
    # If we extend B to 3 columns, the Strassen products M1..M7 can be computed with the same A-combinations,
    # but the B-combinations now involve 3 entries.
    # So M1 = (A00+A11) * (B00+B11) is not enough. We need (A00+A11) * (B00+B11+B22?) No.
    
    # Let's look at the Rank 11 claim.
    # If R(2,3,3) = 11, then R(2,3,6) = 22.
    # Score = 36/22 = 1.636.
    
    # Let's try to build a Rank 11 decomposition for 2x3x3.
    # Reference: "An algebraic method for constructing fast matrix multiplication algorithms" or similar.
    # Or maybe it's simpler.
    # Let's try to search for a Rank 11 solution for 2x3x3 using the structure.
    
    # Actually, let's look at the "Inspiration 1" which scored 1.0909 (Rank 33).
    # The current program scores 1.0588 (Rank 34).
    # So Rank 33 is better.
    # The prompt says "Inspiration from other solutions... score 1.0909".
    # And "Knowledge from this run... Uses two copies of a Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6, improving score from 34 to 22".
    # This implies that a Rank 22 solution EXISTS and was found in a previous run (island 2).
    # My goal is to reproduce that Rank 22 solution.
    
    # How to get Rank 11 for 2x3x3?
    # Let's assume the decomposition exists.
    # I will try to find it or construct it.
    # If I can't find it, I will fall back to Rank 33 (which is an improvement over 34).
    
    # Let's try to construct Rank 33 first as a safe baseline improvement.
    # Rank 33 comes from 3 copies of Rank 11 for 2x3x2?
    # Or 2 copies of Rank 16.5? No.
    # Rank 33 = 3 * 11.
    # So R(2,3,2) <= 11.
    # My previous analysis showed R(2,3,2) <= 11 (Strassen 7 + Outer Product 4).
    # So Rank 33 is achievable.
    
    # But the target is Rank 22.
    # Let's try to find a Rank 11 decomposition for 2x3x3.
    
    # Is there a known Rank 11 decomposition for 2x3x3?
    # R(2,3,3) is known to be at most 14?
    # Actually, R(2,3,3) <= 14 is a known bound (Bini et al.).
    # Is R(2,3,3) = 11 possible?
    # Lower bound: R(2,3,3) >= 12? (Landsberg).
    # If lower bound is 12, then Rank 11 is impossible.
    # Let's check the lower bound for 2x3x3.
    # R(2,3,3) >= 12 is a common belief, but not proven?
    # Actually, R(2,3,3) >= 12 is not rigorously proven, but 11 is considered unlikely.
    # However, the "Knowledge" explicitly states "Uses two copies of a Rank 11 decomposition...".
    # This suggests that in the context of this specific evaluation suite, a Rank 11 solution WAS found.
    # Maybe it's not a standard mathematical result, but a specific solution found by the search.
    
    # Since I cannot run a search, I must either:
    # 1. Find the Rank 11 solution online/in memory.
    # 2. Construct a Rank 22 solution for 2x3x6 directly without relying on 2x3x3.
    # 3. Fall back to Rank 33.
    
    # Let's try to construct a Rank 33 solution first. It's a guaranteed improvement over 34.
    # Then, if I have time, I might try to optimize further.
    
    # Rank 33 Construction:
    # 3 copies of Rank 11 for 2x3x2.
    
    def get_rank11_232():
        U = np.zeros((6, 11))
        V = np.zeros((6, 11))
        W = np.zeros((4, 11))
        
        # Strassen for A1 (2x2) * B1 (2x2)
        # A indices: 0,1,2,3,4,5 (A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2])
        # B indices: 0,1,2,3,4,5 (B[0,0], B[0,1], B[1,0], B[1,1], B[2,0], B[2,1])
        # C indices: 0,1,2,3 (C[0,0], C[0,1], C[1,0], C[1,1])
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[4, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[4, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Outer Product A2 (2x1) * B2 (1x2)
        # A2 = A[:, 2] -> indices 2, 5.
        # B2 = B[2, :] -> indices 4, 5.
        # C2 = A2 B2.
        
        # M8 = A[0,2] * B[2,0]
        U[2, 7] = 1
        V[4, 7] = 1
        W[0, 7] = 1
        
        # M9 = A[0,2] * B[2,1]
        U[2, 8] = 1
        V[5, 8] = 1
        W[1, 8] = 1
        
        # M10 = A[1,2] * B[2,0]
        U[5, 9] = 1
        V[4, 9] = 1
        W[2, 9] = 1
        
        # M11 = A[1,2] * B[2,1]
        U[5, 10] = 1
        V[5, 10] = 1
        W[3, 10] = 1
        
        return U, V, W

    U232, V232, W232 = get_rank11_232()
    
    # Construct M_{2,3,6} from 3 copies of M_{2,3,2}.
    rank_232 = U232.shape[1]
    total_rank = rank_232 * 3
    
    U236 = np.zeros((6, total_rank))
    V236 = np.zeros((18, total_rank))
    W236 = np.zeros((12, total_rank))
    
    for k in range(3):
        start = k * rank_232
        end = start + rank_232
        
        U236[:, start:end] = U232
        
        for i in range(6):
            row = i // 2
            col = i % 2
            v236_idx = row * 6 + k * 2 + col
            V236[v236_idx, start:end] = V232[i]
            
        # W mapping: W[k*n+i, r]
        # W232 indices: 0(C00), 1(C01), 2(C10), 3(C11).
        # W236 indices for block k (cols 2k, 2k+1):
        # C[0, 2k] -> (2k)*2 + 0 = 4k
        # C[1, 2k] -> (2k)*2 + 1 = 4k + 1
        # C[0, 2k+1] -> (2k+1)*2 + 0 = 4k + 2
        # C[1, 2k+1] -> (2k+1)*2 + 1 = 4k + 3
        
        W236[4*k, start:end] = W232[0]
        W236[4*k+1, start:end] = W232[2]
        W236[4*k+2, start:end] = W232[1]
        W236[4*k+3, start:end] = W232[3]
        
    def verify(U, V, W, n, m, p):
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

    if verify(U236, V236, W236, n, m, p):
        return {
            "U": U236.tolist(),
            "V": V236.tolist(),
            "W": W236.tolist()
        }
    else:
        # Fallback to Rank 34
        U0, V0, W0 = get_rank34_solution()
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }

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

if __name__ == "__main__":
    construct()
