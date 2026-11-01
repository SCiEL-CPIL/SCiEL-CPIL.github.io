"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (U_size, V_size, W_size)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    T = np.zeros((U_size, V_size, W_size))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (size, rank)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def objective(x, rank):
        """
        Objective function for scipy.optimize.minimize.
        x is a flattened vector of U, V, W.
        """
        U_size_r = U_size * rank
        V_size_r = V_size * rank
        W_size_r = W_size * rank
        
        U_x = x[:U_size_r].reshape(U_size, rank)
        V_x = x[U_size_r:U_size_r+V_size_r].reshape(V_size, rank)
        W_x = x[U_size_r+V_size_r:].reshape(W_size, rank)
        
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U_x[:, c], V_x[:, c], W_x[:, c])
        
        return np.sum((T_calc - T)**2)

    # Strassen-based structured initialization for 2x2x2 block
    def strassen_2x2x2_init():
        """
        Return a rank-7 decomposition for 2x2x2 matrix multiplication.
        Strassen's 7 products for C = A*B where A,B are 2x2.
        A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        
        M1 = (a11+a22)*(b11+b22)
        M2 = (a21+a22)*b11
        M3 = a11*(b12-b22)
        M4 = a22*(b21-b11)
        M5 = (a11+a12)*b22
        M6 = (a21-a11)*(b11+b12)
        M7 = (a12-a22)*(b21+b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # For 2x2x2: n=2, m=2, p=2
        # U index: i*2 + j, V index: j*2 + k, W index: k*2 + i
        # U_size=4, V_size=4, W_size=4
        
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Column 0: M1 = (a11+a22)*(b11+b22)
        # U: rows 0(a11), 3(a22) -> U[0,0]=1, U[3,0]=1
        # V: cols 0(b11), 1(b12)... wait, let me be careful
        # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
        # U idx: i*2+j: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V idx: j*2+k: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W idx: k*2+i: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # M1 = (a11+a22)*(b11+b22)
        # a11 is U[0], a22 is U[3]
        # b11 is V[0], b22 is V[3]
        # C11 = W[0] (k=0,i=0), C22 = W[3] (k=1,i=1)
        # C11 += M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)*b11
        # a21 is U[2], a22 is U[3], b11 is V[0]
        # C21 += M2 -> W[2] (k=1,i=0)? No, C21 is (i=1,j=0) -> W idx = k*2+i for output... 
        # Wait, W index is k*n+i where n=2. So W[k*2+i].
        # C[i,j] = sum_k A[i,k]*B[k,j]. For 2x2x2, output C is 2x2.
        # W index: k*n + i = k*2 + i. So W[0]=C[0,0] (k=0,i=0), W[1]=C[0,1] (k=0,i=1)?
        # No wait. The tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i.
        # For 2x2x2: u=i*2+j, v=j*2+k, w=k*2+i.
        # T[u,v,w]=1 means A[i,j]*B[j,k] contributes to C[i,k].
        # So w=k*2+i encodes (k,i) which is (row of B, col of A) -> output C[i,k].
        # So W[w] corresponds to C[i,k] where w=k*2+i.
        # W[0] = C[0,0] (k=0,i=0), W[1] = C[1,0] (k=0,i=1), W[2] = C[0,1] (k=1,i=0), W[3] = C[1,1] (k=1,i=1)
        
        # M2 = (a21+a22)*b11: a21=U[2], a22=U[3], b11=V[0]
        # Contributes to C[1,0] -> W[1] (k=0,i=1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3 = a11*(b12-b22): a11=U[0], b12=V[1], b22=V[3]
        # Contributes to C[0,1] -> W[2] (k=1,i=0)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4 = a22*(b21-b11): a22=U[3], b21=V[2], b11=V[0]
        # Contributes to C[0,0] -> W[0], and C[1,0] -> W[1]
        # C11 += M4, C21 += M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (a11+a12)*b22: a11=U[0], a12=U[1], b22=V[3]
        # Contributes to C[0,0] -> W[0] (with -1) and C[0,1] -> W[2]
        # C11 -= M5, C12 += M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (a21-a11)*(b11+b12): a21=U[2], a11=U[0], b11=V[0], b12=V[1]
        # Contributes to C[1,1] -> W[3]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)*(b21+b22): a12=U[1], a22=U[3], b21=V[2], b22=V[3]
        # Contributes to C[0,0] -> W[0] and C[1,1] -> W[3]
        # C11 += M7, C22 += M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Try to extend Strassen 2x2x2 to 2x3x3 by adding terms for the extra column/row
    def try_strassen_extension():
        """
        Use Strassen 2x2x2 as base, then add terms for the 3rd column of A and 3rd column of B.
        This gives a structured initial guess.
        """
        U2, V2, W2 = strassen_2x2x2_init()
        
        # For 2x3x3, we have U_size=6, V_size=9, W_size=6
        # U indices: i*3+j for i in {0,1}, j in {0,1,2}
        #   i=0: j=0->0, j=1->1, j=2->2
        #   i=1: j=0->3, j=1->4, j=2->5
        # V indices: j*3+k for j in {0,1,2}, k in {0,1,2}
        #   j=0: k=0->0, k=1->1, k=2->2
        #   j=1: k=0->3, k=1->4, k=2->5
        #   j=2: k=0->6, k=1->7, k=2->8
        # W indices: k*2+i for k in {0,1,2}, i in {0,1}
        #   k=0: i=0->0, i=1->1
        #   k=1: i=0->2, i=1->3
        #   k=2: i=0->4, i=1->5
        
        # Map 2x2x2 indices to 2x3x3:
        # 2x2x2 U: i*2+j -> 2x3x3 U: i*3+j (for j in {0,1})
        # 2x2x2 V: j*2+k -> 2x3x3 V: j*3+k (for j in {0,1}, k in {0,1})
        # 2x2x2 W: k*2+i -> 2x3x3 W: k*2+i (same, since n=2)
        
        base_rank = 7
        # Start with base rank 7, then we'll need more for the extra dimensions
        # The extra terms involve j=2 (3rd row of A) and k=2 (3rd col of B)
        
        # Let's try rank 13-14 with structured initialization
        for extra_rank in range(6, 10):
            total_rank = base_rank + extra_rank
            if total_rank > 18:
                break
            
            U = np.zeros((U_size, total_rank))
            V = np.zeros((V_size, total_rank))
            W = np.zeros((W_size, total_rank))
            
            # Copy Strassen base (map indices)
            for c in range(base_rank):
                for u_idx_222 in range(4):
                    i = u_idx_222 // 2
                    j = u_idx_222 % 2
                    u_idx_233 = i * 3 + j
                    U[u_idx_233, c] = U2[u_idx_222, c]
                
                for v_idx_222 in range(4):
                    j = v_idx_222 // 2
                    k = v_idx_222 % 2
                    v_idx_233 = j * 3 + k
                    V[v_idx_233, c] = V2[v_idx_222, c]
                
                # W indices are the same
                for w_idx in range(4):
                    W[w_idx, c] = W2[w_idx, c]
            
            # Add extra terms for the 3rd column/row
            # These need to handle: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2}
            # and A[i,j]*B[j,2] for i in {0,1}, j in {0,1}
            # and A[i,2]*B[2,2]
            
            # Simple approach: add schoolbook-like terms for the new indices
            extra_col = base_rank
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Skip if already covered by Strassen base
                        if j < 2 and k < 2:
                            continue
                        if extra_col >= total_rank:
                            break
                        U[i * m + j, extra_col] = 1
                        V[j * p + k, extra_col] = 1
                        W[k * n + i, extra_col] = 1
                        extra_col += 1
            
            # Add small noise
            U += np.random.randn(U_size, total_rank) * 0.05
            V += np.random.randn(V_size, total_rank) * 0.05
            W += np.random.randn(W_size, total_rank) * 0.05
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(lambda x: objective(x, total_rank), x0, method='L-BFGS-B', 
                              options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:U_size*total_rank].reshape(U_size, total_rank)
                    V_opt = x_opt[U_size*total_rank:U_size*total_rank+V_size*total_rank].reshape(V_size, total_rank)
                    W_opt = x_opt[U_size*total_rank+V_size*total_rank:].reshape(W_size, total_rank)
                    
                    result = snap_and_verify(U_opt, V_opt, W_opt)
                    if result:
                        U_final, V_final, W_final = result
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
            except Exception:
                continue
        
        return None

    # Generate initial guesses by merging columns of the schoolbook decomposition
    def generate_initial_guesses(rank, num_guesses):
        """
        Generate initial guesses by merging columns of the schoolbook decomposition.
        """
        schoolbook_rank = n * m * p
        U_sb = np.zeros((U_size, schoolbook_rank))
        V_sb = np.zeros((V_size, schoolbook_rank))
        W_sb = np.zeros((W_size, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        
        guesses = []
        if rank >= schoolbook_rank:
            return guesses
            
        for _ in range(num_guesses):
            if rank > schoolbook_rank:
                continue
                
            leaders = np.random.choice(schoolbook_rank, rank, replace=False)
            groups = {l: [l] for l in leaders}
            
            remaining = [i for i in range(schoolbook_rank) if i not in leaders]
            for idx in remaining:
                leader = np.random.choice(leaders)
                groups[leader].append(idx)
                
            U_new = np.zeros((U_size, rank))
            V_new = np.zeros((V_size, rank))
            W_new = np.zeros((W_size, rank))
            
            for new_col_idx, leader in enumerate(leaders):
                cols = groups[leader]
                U_new[:, new_col_idx] = np.sum(U_sb[:, cols], axis=1)
                V_new[:, new_col_idx] = np.sum(V_sb[:, cols], axis=1)
                W_new[:, new_col_idx] = np.sum(W_sb[:, cols], axis=1)
                
            U_new += np.random.randn(U_size, rank) * 0.1
            V_new += np.random.randn(V_size, rank) * 0.1
            W_new += np.random.randn(W_size, rank) * 0.1
            
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            guesses.append(x0)
            
        return guesses

    # First try Strassen extension
    if time.time() < deadline:
        result = try_strassen_extension()
        if result:
            return result

    # Try ranks from 14 upwards with schoolbook-based initializations
    for rank in range(14, 19):
        if time.time() > deadline:
            break
            
        max_restarts = 30 if rank < 17 else 10
        
        guesses = generate_initial_guesses(rank, max_restarts)
        
        for x0 in guesses:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(lambda x: objective(x, rank), x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success or res.fun < 1e-8:
                    x_opt = res.x
                    U_opt = x_opt[:U_size*rank].reshape(U_size, rank)
                    V_opt = x_opt[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                    W_opt = x_opt[U_size*rank+V_size*rank:].reshape(W_size, rank)
                    
                    result = snap_and_verify(U_opt, V_opt, W_opt)
                    if result:
                        U_final, V_final, W_final = result
                        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
            except Exception:
                continue

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
