import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 20
    
    # Try to construct rank-20 decomposition
    # Start from Strassen (rank-21) and try to merge one component
    
    # Strassen for 2x2: 7 rank-1 terms
    # For 3 blocks, we get 21 terms
    # Try to find if one term can be expressed as combination of others
    
    # Actually, for block-diagonal tensor, rank = sum of ranks = 21
    # So rank-20 is NOT possible for pure block-diagonal structure
    
    # But wait - maybe we can use a non-trivial decomposition
    # Let me try to use the structure more cleverly
    
    # For 2x2x6, let's think of it as 2x2x2 + 2x2x2 + 2x2x2
    # Each 2x2x2 has rank 7 (Strassen)
    # Total = 21
    
    # To get rank 20, we need to exploit cross-block structure
    # This is non-trivial. Let me try a different approach.
    
    # Alternative: Use the fact that p=6 = 3*2, and try to find
    # a decomposition where some rank-1 terms share structure
    
    # Let me try to construct it directly
    # U: shape (4, rank), V: shape (12, rank), W: shape (12, rank)
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # Strassen coefficients for each 2x2 block
    # s=1..7
    S1 = {'U': [1,0,0,1], 'V': [1,0,0,1], 'W': [1,0,0,1]}
    S2 = {'U': [0,0,1,1], 'V': [1,0,0,0], 'W': [0,1,0,-1]}
    S3 = {'U': [1,0,0,0], 'V': [0,1,0,-1], 'W': [0,0,1,1]}
    S4 = {'U': [0,0,0,1], 'V': [-1,0,1,0], 'W': [1,1,0,0]}
    S5 = {'U': [1,1,0,0], 'V': [0,0,0,1], 'W': [-1,0,1,0]}
    S6 = {'U': [-1,0,1,0], 'V': [1,1,0,0], 'W': [0,0,0,1]}
    S7 = {'U': [0,1,0,-1], 'V': [0,0,1,1], 'W': [1,0,0,0]}
    
    strassen = [S1, S2, S3, S4, S5, S6, S7]
    
    # For block b (0,1,2), the V indices are [2b, 2b+1, 6+2b, 6+2b+1]
    # W indices are [4b, 4b+1, 4b+2, 4b+3]
    
    # First, build the rank-21 Strassen decomposition
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    
    # Now try to see if we can reduce rank by 1
    # Check if any column is a linear combination of others
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor is T = sum_s U_s ⊗ V_s ⊗ W_s
    # We want to check if one term can be removed
    
    # For a valid decomposition, we need:
    # For all (i,j,k): sum_s U[i,s]*V[j,s]*W[k,s] = T[i,j,k]
    
    # Let's verify the rank-21 first, then try to reduce
    
    # Actually, let me just try rank-20 by dropping one column and checking
    # if we can adjust the remaining columns
    
    # This is getting complex. Let me try a different strategy:
    # Use the known result that 2x2x6 might have rank 20
    
    # Let me try to construct it by hand based on symmetry
    
    # Alternative approach: use bilinear complexity bounds
    # For 2x2xn, the rank is known to be 7n - something for large n
    # For n=2, rank=7; for n=6, maybe rank=20?
    
    # Let me try a constructive approach
    # Use 3 Strassen decompositions but merge the "diagonal" terms
    
    # The idea: in Strassen, P1 = (A00+A11)(B00+B11) gives diagonal contributions
    # If we can share some of these across blocks...
    
    # Actually, let me just try to solve it numerically
    # Set up the linear system for rank-20
    
    if time.time() > deadline - 50:
        # Fallback to rank-21
        rank = 21
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(12)]
        W = [[0] * rank for _ in range(12)]
        for b in range(3):
            for s in range(7):
                col = b * 7 + s
                sc = strassen[s]
                for idx in range(4):
                    if sc['U'][idx] != 0:
                        U[idx][col] = sc['U'][idx]
                v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
                for idx in range(4):
                    if sc['V'][idx] != 0:
                        V[v_idx[idx]][col] = sc['V'][idx]
                w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
                for idx in range(4):
                    if sc['W'][idx] != 0:
                        W[w_idx[idx]][col] = sc['W'][idx]
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank-20 by solving linear system
    # We need to find U (4x20), V (12x20), W (12x20) such that
    # for all (i,j,k) in support: sum_s U[i,s]*V[j,s]*W[k,s] = 1
    # and for all other (i,j,k): sum_s U[i,s]*V[j,s]*W[k,s] = 0
    
    # This is a bilinear system. Let me try a heuristic approach.
    
    # Start from rank-21 and try to nullify one column
    # by adjusting others
    
    # Try dropping column 0 and solving for adjustments
    # This is complex. Let me try a simpler heuristic.
    
    # Use the fact that the tensor has symmetries
    # Try to find a decomposition with rank 20
    
    # I'll try a random search approach
    np.random.seed(42)
    
    # Initialize with small random values
    U_rand = np.random.randn(4, 20) * 0.1
    V_rand = np.random.randn(12, 20) * 0.1
    W_rand = np.random.randn(12, 20) * 0.1
    
    # The target tensor T[i,j,k] = 1 if j is between 0 and 1 and k = j + 2*(i) ... 
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to C[i,k]
    # i.e., T[i,j,k] = 1 for all valid (i,j,k) where i in [0,1], j in [0,1], k in [0,5]
    # and the contribution is A[i,j]*B[j,k] -> C[i,k]
    
    # So the tensor has 1s at positions (i,j,k) where i in {0,1}, j in {0,1}, k in {0,...,5}
    # and the value is 1 (it's a 0-1 tensor for the structure)
    
    # Wait, the tensor is T[i,j,k] = 1 if the output C[i,k] depends on A[i,j]*B[j,k]
    # For standard matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,5}
    # That's 2*2*6 = 24 non-zero entries, all equal to 1
    
    # So we need: for all (i,j,k), sum_s U[i,s]*V[j,s]*W[k,s] = 1
    
    # This is a system of 24 equations with 20*3 = 60 unknowns (but bilinear)
    
    # Let me try to solve it using alternating least squares
    
    # Fix V and W, solve for U (linear in U)
    # Fix U and W, solve for V (linear in V)  
    # Fix U and V, solve for W (linear in W)
    
    # Initialize
    U_sol = np.zeros((4, 20))
    V_sol = np.zeros((12, 20))
    W_sol = np.zeros((12, 20))
    
    # Initialize with Strassen-like structure
    for b in range(3):
        for s in range(7):
            col = min(b * 7 + s, 19)  # Cap at 20
            if col < 20:
                sc = strassen[s]
                for idx in range(4):
                    U_sol[idx, col] = sc['U'][idx]
                v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
                for idx in range(4):
                    V_sol[v_idx[idx], col] = sc['V'][idx]
                w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
                for idx in range(4):
                    W_sol[w_idx[idx], col] = sc['W'][idx]
    
    # Now try to optimize
    # The objective: minimize sum_{i,j,k} (sum_s U[i,s]*V[j,s]*W[k,s] - 1)^2
    
    # Alternating least squares
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        # Solve for U: for each i, U[i,:] = argmin sum_{j,k} (sum_s U[i,s]*V[j,s]*W[k,s] - 1)^2
        # This is linear in U[i,:]
        # Let M[i] be a 20x20 matrix where M[i][s,s'] = sum_{j,k} V[j,s]*W[k,s']*V[j,s']... 
        # Actually, for fixed i, we want to minimize:
        # sum_{j,k} (sum_s U[i,s] * (V[j,s]*W[k,s]) - 1)^2
        # Let a[j,k,s] = V[j,s]*W[k,s]
        # Then we want to minimize sum_{j,k} (U[i,:] @ a[j,k,:] - 1)^2
        # This is a least squares problem: A[i] @ U[i,:] = b[i]
        # where A[i][j*k, s] = a[j,k,s] and b[i][j*k] = 1
        
        A_U = np.zeros((24, 20))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    row = i*12 + j*6 + k
                    for s in range(20):
                        A_U[row, s] = V_sol[j, s] * W_sol[k, s]
        
        b_U = np.ones(24)
        
        # Solve for U[i,:] for each i
        # But the equations are coupled across i because the same s index is shared
        # Actually no, U[i,:] only appears in equations for that i
        # Wait, the equation for (i,j,k) involves U[i,s] for all s
        # So for each i, we have 12 equations (j=0,1 and k=0..5)
        
        for i in range(2):
            A_i = A_U[i*12:(i+1)*12, :]
            b_i = b_U[i*12:(i+1)*12]
            # Solve A_i @ U[i,:] = b_i
            U_sol[i, :] = np.linalg.lstsq(A_i, b_i, rcond=None)[0]
        
        # Solve for V: similar
        A_V = np.zeros((24, 20))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    row = i*12 + j*6 + k
                    for s in range(20):
                        A_V[row, s] = U_sol[i, s] * W_sol[k, s]
        
        b_V = np.ones(24)
        
        for j in range(2):
            # V[j,:] appears in equations where j is fixed
            # Equations: for i in {0,1}, k in {0..5}
            rows = []
            for i in range(2):
                for k in range(6):
                    rows.append(i*12 + j*6 + k)
            A_j = A_V[rows, :]
            b_j = b_V[rows]
            V_sol[j, :] = np.linalg.lstsq(A_j, b_j, rcond=None)[0]
        
        # Solve for W: similar
        A_W = np.zeros((24, 20))
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    row = i*12 + j*6 + k
                    for s in range(20):
                        A_W[row, s] = U_sol[i, s] * V_sol[j, s]
        
        b_W = np.ones(24)
        
        for k in range(6):
            rows = []
            for i in range(2):
                for j in range(2):
                    rows.append(i*12 + j*6 + k)
            A_k = A_W[rows, :]
            b_k = b_W[rows]
            W_sol[k, :] = np.linalg.lstsq(A_k, b_k, rcond=None)[0]
    
    # Check the solution
    max_err = 0
    for i in range(2):
        for j in range(2):
            for k in range(6):
                val = sum(U_sol[i, s] * V_sol[j, s] * W_sol[k, s] for s in range(20))
                err = abs(val - 1)
                max_err = max(max_err, err)
    
    if max_err < 1e-6:
        # Convert to lists with half-integer rounding
        U_out = [[round(U_sol[i, s] * 2) / 2 for s in range(20)] for i in range(4)]
        V_out = [[round(V_sol[j, s] * 2) / 2 for s in range(20)] for j in range(12)]
        W_out = [[round(W_sol[k, s] * 2) / 2 for s in range(20)] for k in range(12)]
        
        # Verify with rounded values
        max_err_rounded = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    val = sum(U_out[i][s] * V_out[j][s] * W_out[k][s] for s in range(20))
                    err = abs(val - 1)
                    max_err_rounded = max(max_err_rounded, err)
        
        if max_err_rounded < 1e-9:
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to rank-21
    rank = 21
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    if res:
        print("Rank:", len(res["U"][0]))
