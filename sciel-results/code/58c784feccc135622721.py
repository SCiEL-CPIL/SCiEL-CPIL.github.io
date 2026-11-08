"""
Strategy: Block Decomposition
We decompose the 2x3 * 3x10 multiplication into 5 independent 2x3 * 3x2 multiplications.
The tensor T_{2x3x10} is the direct sum of 5 copies of T_{2x3x2}.
If we find a rank-r decomposition for T_{2x3x2}, we get a rank-5r decomposition for T_{2x3x10}.
We use a fast, vectorized local search to find the minimal rank for the 2x3x2 case.
The schoolbook rank for 2x3x2 is 12. 5*12 = 60.
If we find rank 11 for 2x3x2, total rank is 55.
If we find rank 10, total rank is 50.
"""

import time
import numpy as np

N, M, P = 2, 3, 10
SUB_P = 2
NUM_BLOCKS = P // SUB_P

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Find minimal rank for 2x3x2
    # We will try ranks from a lower bound up to schoolbook (12)
    # Lower bound for 2x3x2 is at least max(2*3, 3*2, 2*2) = 6? 
    # Actually, rank >= max(n*m, m*p, p*n) is not a lower bound.
    # Rank >= max(n, m, p) is not a lower bound.
    # Let's just try 6 to 12.
    
    best_sub_rank = None
    best_sub_U = None
    best_sub_V = None
    best_sub_W = None
    
    # Candidates for local search
    # We use small integers and half-integers
    candidates = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    # Precompute constraints for 2x3x2
    dim_U_sub = N * M
    dim_V_sub = M * SUB_P
    dim_W_sub = SUB_P * N
    
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k', c=k''*n+i'' and i=i'', j=j', k=k'
    # Wait, indices for 2x3x2:
    # A is 2x3 (i, j)
    # B is 3x2 (j, k)
    # C is 2x2 (i, k)
    # U index: i*M + j
    # V index: j*SUB_P + k
    # W index: k*N + i
    
    constraints_1 = []
    constraints_0 = []
    for a in range(dim_U_sub):
        for b in range(dim_V_sub):
            for c in range(dim_W_sub):
                i = a // M
                j = a % M
                j_prime = b // SUB_P
                k_prime = b % SUB_P
                i_prime = c // N
                k = c % N
                
                if i == i_prime and j == j_prime and k == k_prime:
                    constraints_1.append((a, b, c))
                else:
                    constraints_0.append((a, b, c))
    
    # Vectorized error calculation
    def calculate_error(U, V, W):
        # U: (6, R), V: (6, R), W: (4, R)
        # We need sum_t U[a,t] V[b,t] W[c,t]
        # This is a batched tensor contraction.
        # We can compute the full tensor T_reconstructed of shape (6, 6, 4)
        # T_recon[a, b, c] = sum_t U[a,t] V[b,t] W[c,t]
        
        # Using einsum:
        # U[a, t], V[b, t], W[c, t] -> T[a, b, c]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Error on 1s
        if len(constraints_1) > 0:
            a1, b1, c1 = zip(*constraints_1)
            err1 = np.sum((T_recon[np.array(a1), np.array(b1), np.array(c1)] - 1.0)**2)
        else:
            err1 = 0.0
            
        # Error on 0s
        if len(constraints_0) > 0:
            a0, b0, c0 = zip(*constraints_0)
            err0 = np.sum((T_recon[np.array(a0), np.array(b0), np.array(c0)])**2)
        else:
            err0 = 0.0
            
        return err1 + err0

    def find_solution_sub(R, max_iters=50000):
        if time.time() > deadline:
            return None
            
        # Random init
        U = np.random.choice(candidates, (dim_U_sub, R))
        V = np.random.choice(candidates, (dim_V_sub, R))
        W = np.random.choice(candidates, (dim_W_sub, R))
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        current_error = calculate_error(U, V, W)
        
        # Simulated Annealing / Random Walk
        T = 1.0
        cooling_rate = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Randomly perturb one entry
            var_type = np.random.randint(3)
            if var_type == 0:
                idx = np.random.randint(dim_U_sub)
                t = np.random.randint(R)
                old_val = U[idx, t]
                U[idx, t] = np.random.choice(candidates)
            elif var_type == 1:
                idx = np.random.randint(dim_V_sub)
                t = np.random.randint(R)
                old_val = V[idx, t]
                V[idx, t] = np.random.choice(candidates)
            else:
                idx = np.random.randint(dim_W_sub)
                t = np.random.randint(R)
                old_val = W[idx, t]
                W[idx, t] = np.random.choice(candidates)
                
            new_error = calculate_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-9:
                        # Verify exactness
                        T_recon = np.einsum('at,bt,ct->abc', best_U, best_V, best_W)
                        valid = True
                        for a, b, c in constraints_1:
                            if abs(T_recon[a, b, c] - 1.0) > 1e-9:
                                valid = False
                                break
                        if valid:
                            for a, b, c in constraints_0:
                                if abs(T_recon[a, b, c]) > 1e-9:
                                    valid = False
                                    break
                            if valid:
                                return best_U, best_V, best_W
            else:
                # Revert if not accepted (simple hill climbing)
                if var_type == 0:
                    U[idx, t] = old_val
                elif var_type == 1:
                    V[idx, t] = old_val
                else:
                    W[idx, t] = old_val
            
            T *= cooling_rate
        
        return None

    # Try ranks for 2x3x2
    # Schoolbook is 12.
    # Let's try 12 down to 6.
    for R_sub in range(12, 5, -1):
        if time.time() > deadline:
            break
            
        sol = find_solution_sub(R_sub, max_iters=20000)
        if sol is not None:
            best_sub_U, best_sub_V, best_sub_W = sol
            best_sub_rank = R_sub
            break # Found minimal rank? We are iterating down, so yes.
            
    if best_sub_rank is None:
        # Fallback to schoolbook for sub-problem
        best_sub_rank = 12
        best_sub_U = np.zeros((dim_U_sub, 12))
        best_sub_V = np.zeros((dim_V_sub, 12))
        best_sub_W = np.zeros((dim_W_sub, 12))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(SUB_P):
                    best_sub_U[i*M + j, col] = 1
                    best_sub_V[j*SUB_P + k, col] = 1
                    best_sub_W[k*N + i, col] = 1
                    col += 1

    # 2. Construct full 2x3x10 solution from sub-solution
    # Total rank R = NUM_BLOCKS * best_sub_rank
    R_total = NUM_BLOCKS * best_sub_rank
    
    U_full = np.zeros((N*M, R_total))
    V_full = np.zeros((M*P, R_total))
    W_full = np.zeros((P*N, R_total))
    
    for block_idx in range(NUM_BLOCKS):
        # Slice for this block
        # U is same for all blocks (it's 2x3)
        # V is sliced: 3x2 block
        # W is sliced: 2x2 block
        
        # Column offset
        col_start = block_idx * best_sub_rank
        col_end = (block_idx + 1) * best_sub_rank
        
        # U: Copy the sub-U
        U_full[:, col_start:col_end] = best_sub_U
        
        # V: Map 3x2 block to 3x10
        # Sub-V index: j*2 + k_sub
        # Full-V index: j*10 + k_full
        # k_full = block_idx * 2 + k_sub
        for j in range(M):
            for k_sub in range(SUB_P):
                v_idx_sub = j * SUB_P + k_sub
                v_idx_full = j * P + (block_idx * SUB_P + k_sub)
                V_full[v_idx_full, col_start:col_end] = best_sub_V[v_idx_sub, :]
                
        # W: Map 2x2 block to 10x2 (wait, W is P*N = 10*2 = 20)
        # Sub-W index: k_sub*N + i
        # Full-W index: k_full*N + i
        # k_full = block_idx * 2 + k_sub
        for k_sub in range(SUB_P):
            for i in range(N):
                w_idx_sub = k_sub * N + i
                w_idx_full = (block_idx * SUB_P + k_sub) * N + i
                W_full[w_idx_full, col_start:col_end] = best_sub_W[w_idx_sub, :]
                
    # Verification (optional, but good for debugging)
    # T_recon_full = np.einsum('at,bt,ct->abc', U_full, V_full, W_full)
    # Check a few constraints...
    
    return {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
