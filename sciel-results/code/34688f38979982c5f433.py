import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Verification using numpy for speed
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But realistically higher. Try from 50 to 104.
    
    def try_rank(R, num_trials=200, seed=42):
        """Try to find a valid decomposition with rank R."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with small integers/half-integers
            # Use values from {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, R))
            
            # Check if this works
            if verify(U.tolist(), V.tolist(), W.tolist(), R):
                return U.tolist(), V.tolist(), W.tolist()
            
            # Local refinement: try to fix violations by adjusting entries
            # This is a heuristic - for each violated (i,j,k), try flipping
            # or adjusting entries in U, V, W for that column pattern
            U_f = U.copy()
            V_f = V.copy()
            W_f = W.copy()
            
            for _ in range(50):  # 50 refinement steps
                if time.time() > deadline - 5:
                    return None
                
                # Find violated triples
                violated = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_f[i * m + j], V_f[j * p + k] * W_f[k * n + i])
                            if abs(val - 1.0) > 1e-9:
                                violated.append((i, j, k, val))
                
                if not violated:
                    if verify(U_f.tolist(), V_f.tolist(), W_f.tolist(), R):
                        return U_f.tolist(), V_f.tolist(), W_f.tolist()
                    break
                
                # Pick a random violated triple and try to fix it
                i, j, k, val = rng.choice(len(violated), size=1)[0]
                i, j, k, val = violated[i]
                
                # Try adjusting one entry at a time
                fixed = False
                # Try adjusting U
                for r in range(R):
                    old = U_f[i * m + j, r]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old:
                            continue
                        U_f[i * m + j, r] = new_val
                        new_val_check = np.dot(U_f[i * m + j], V_f[j * p + k] * W_f[k * n + i])
                        if abs(new_val_check - 1.0) < abs(val - 1.0):
                            fixed = True
                            break
                        U_f[i * m + j, r] = old
                    if fixed:
                        break
                
                if not fixed:
                    # Try adjusting V
                    for r in range(R):
                        old = V_f[j * p + k, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old:
                                continue
                            V_f[j * p + k, r] = new_val
                            new_val_check = np.dot(U_f[i * m + j], V_f[j * p + k] * W_f[k * n + i])
                            if abs(new_val_check - 1.0) < abs(val - 1.0):
                                fixed = True
                                break
                            V_f[j * p + k, r] = old
                        if fixed:
                            break
                
                if not fixed:
                    # Try adjusting W
                    for r in range(R):
                        old = W_f[k * n + i, r]
                        for new_val in [-1, -0.5, 0, 0.5, 1]:
                            if new_val == old:
                                continue
                            W_f[k * n + i, r] = new_val
                            new_val_check = np.dot(U_f[i * m + j], V_f[j * p + k] * W_f[k * n + i])
                            if abs(new_val_check - 1.0) < abs(val - 1.0):
                                fixed = True
                                break
                            W_f[k * n + i, r] = old
                        if fixed:
                            break
        
        return None
    
    # Try ranks from 50 to 104
    for R in range(50, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, num_trials=100, seed=R)
        if result is not None:
            U, V, W = result
            best_solution = {"U": U, "V": V, "W": W}
            best_rank = R
            break  # Found the lowest rank we can try in order
    
    return best_solution
# EVOLVE-BLOCK-END
