import time
import math
import numpy as np

_START_TIME = time.time()
DEADLINE = _START_TIME + 115

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    n, m, p = 3, 3, 4
    
    # Schoolbook fallback (rank 36)
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a rank-26 decomposition using a structured approach
    # Strategy: Use the fact that we can decompose 3x3x4 as a combination
    # of structured factors. We'll use a direct algebraic construction.
    
    # Approach: Hardcode a known rank-26 decomposition if we can find one,
    # or use a smarter search.
    
    # For now, let's try a direct construction based on the idea that
    # we can use the Winograd-like approach for 3x3x3 (rank 23) and add
    # 3 more for the 4th column.
    
    # Actually, let's just try to find a rank-26 solution using a more
    # efficient search than the current one.
    
    # The key insight is that we need to find U (9x26), V (12x26), W (12x26)
    # with half-integer entries such that for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's use a smarter initialization and search.
    
    # First, let's try a direct algebraic construction.
    # We know that rank(3x3x3) <= 23 (Winograd).
    # So rank(3x3x4) <= 23 + 3 = 26.
    
    # Let's construct this explicitly.
    
    # For the 3x3x3 part, we use the Winograd factors.
    # For the 3x3x1 part, we use schoolbook (rank 3).
    
    # The Winograd factors for 3x3x3 (rank 23) are complex to hardcode.
    # Instead, let's use a different approach.
    
    # We'll use the fact that the multiplication tensor for 3x3x4 can be
    # decomposed using a combination of structured factors.
    
    # Let's try a rank-26 construction:
    # Use 23 factors for the first 3 columns (Winograd-like)
    # Use 3 factors for the 4th column (schoolbook)
    
    # But we need to be careful about the indexing.
    
    # Actually, let's just try to find any rank-26 solution using a
    # more efficient search algorithm.
    
    # We'll use a local search with a better objective function.
    
    # Define the objective: we want to minimize the number of (i,j,k)
    # triples where the sum is not 1.
    
    # Use a random initialization with a structured bias.
    
    # Let's try multiple random initializations and local searches.
    
    found = False
    for attempt in range(50):
        if _time_remaining() < 5:
            break
            
        # Random initialization with ±1 entries
        rng = np.random.RandomState(attempt * 12345 + 6789)
        r = 26
        U = rng.choice([-1, 1], size=(n*m, r))
        V = rng.choice([-1, 1], size=(m*p, r))
        
        # Compute the "badness" - number of (i,j,k) where sum != 1
        def compute_badness(U, V):
            badness = 0
            M = np.zeros((n*p, r), dtype=int)
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        s = 0
                        for j in range(m):
                            s += U[i*m+j, c] * V[j*p+k, c]
                        M[i*p+k, c] = s
                    # Check if there exists a half-integer W row that makes
                    # the dot product equal to 1
                    # M_row dot W_row = 1, where W_row has half-integer entries
                    # This is solvable if and only if gcd of M_row entries
                    # (that are non-zero) divides 2 (since W entries are k/2)
                    # Actually, we need sum_c M[c] * W[c] = 1 where W[c] = k/2
                    # So sum_c M[c] * k/2 = 1 => sum_c M[c] * k = 2
                    # This is a linear Diophantine equation.
                    # For simplicity, check if there's a sparse solution.
                    solvable = False
                    for c in range(r):
                        if M[i*p+k, c] == 1:
                            solvable = True
                            break
                        if M[i*p+k, c] == 2:
                            solvable = True
                            break
                        if M[i*p+k, c] == -1:
                            solvable = True
                            break
                        if M[i*p+k, c] == -2:
                            solvable = True
                            break
                    if not solvable:
                        badness += 1
            return badness, M
        
        badness, M = compute_badness(U, V)
        
        # Local search: flip random entries to reduce badness
        max_iters = 500
        for it in range(max_iters):
            if badness == 0:
                break
            if _time_remaining() < 2:
                break
                
            # Try flipping a random entry in U or V
            flip_U = rng.random() < 0.5
            if flip_U:
                idx = rng.randint(n*m)
                col = rng.randint(r)
                U[idx, col] *= -1
            else:
                idx = rng.randint(m*p)
                col = rng.randint(r)
                V[idx, col] *= -1
                
            new_badness, new_M = compute_badness(U, V)
            if new_badness < badness:
                badness = new_badness
                M = new_M
            else:
                # Revert
                if flip_U:
                    U[idx, col] *= -1
                else:
                    V[idx, col] *= -1
                    
        if badness == 0:
            # Construct W
            W = np.zeros((p*n, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        if M[i*p+k, c] == 1:
                            W[k*n+i, c] = 1.0
                            break
                        elif M[i*p+k, c] == 2:
                            W[k*n+i, c] = 0.5
                            break
                        elif M[i*p+k, c] == -1:
                            W[k*n+i, c] = -1.0
                            break
                        elif M[i*p+k, c] == -2:
                            W[k*n+i, c] = -0.5
                            break
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = []
            for row in W:
                W_list.append([round(x*2)/2.0 for x in row])
            
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
                
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Found decomposition of rank {rank}")
