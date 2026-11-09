import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U, 'V': V, 'W': W}

    def strassen_222_decomp():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return {'U': U, 'V': V, 'W': W}

    def combine_232_decomp():
        d_222 = strassen_222_decomp()
        d_212 = schoolbook_decomp(2, 1, 2)
        
        U_full = np.zeros((6, 11))
        V_full = np.zeros((6, 11))
        W_full = np.zeros((4, 11))
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_full[i*3 + j, c] = d_222['U'][i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V_full[j*2 + k, c] = d_222['V'][j*2 + k, c]
            for i in range(2):
                for k in range(2):
                    W_full[k*2 + i, c] = d_222['W'][k*2 + i, c]

        for c in range(4):
            for i in range(2):
                U_full[i*3 + 2, c + 7] = d_212['U'][i, c]
            for k in range(2):
                V_full[4 + k, c + 7] = d_212['V'][k, c]
            for i in range(2):
                for k in range(2):
                    W_full[k*2 + i, c + 7] = d_212['W'][k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    def combine_233_decomp():
        d_222 = strassen_222_decomp()
        d_221 = schoolbook_decomp(2, 2, 1)
        
        U_223 = np.zeros((4, 11))
        V_223 = np.zeros((6, 11))
        W_223 = np.zeros((6, 11))
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_223[i*2 + j, c] = d_222['U'][i*2 + j, c]
            for j in range(2):
                for k in range(2):
                    V_223[j*2 + k, c] = d_222['V'][j*2 + k, c]
            for i in range(2):
                for k in range(2):
                    W_223[k*2 + i, c] = d_222['W'][k*2 + i, c]

        for c in range(4):
            for i in range(2):
                for j in range(2):
                    U_223[i*2 + j, c + 7] = d_221['U'][i*2 + j, c]
            for j in range(2):
                V_223[j*3 + 2, c + 7] = d_221['V'][j, c]
            for i in range(2):
                W_223[4 + i, c + 7] = d_221['W'][i, c]

        d_223 = {'U': U_223, 'V': V_223, 'W': W_223}
        d_213 = schoolbook_decomp(2, 1, 3)
        
        U_full = np.zeros((6, 17))
        V_full = np.zeros((9, 17))
        W_full = np.zeros((6, 17))
        
        for c in range(11):
            for i in range(2):
                for j in range(2):
                    U_full[i*3 + j, c] = d_223['U'][i*2 + j, c]
            for j in range(2):
                for k in range(3):
                    V_full[j*3 + k, c] = d_223['V'][j*3 + k, c]
            for i in range(2):
                for k in range(3):
                    W_full[k*2 + i, c] = d_223['W'][k*2 + i, c]

        for c in range(6):
            for i in range(2):
                U_full[i*3 + 2, c + 11] = d_213['U'][i, c]
            for k in range(3):
                V_full[2*3 + k, c + 11] = d_213['V'][k, c]
            for i in range(2):
                for k in range(3):
                    W_full[k*2 + i, c + 11] = d_213['W'][k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    def final_decomp():
        d_232 = combine_232_decomp()
        d_233 = combine_233_decomp()
        
        U_full = np.zeros((6, 28))
        V_full = np.zeros((15, 28))
        W_full = np.zeros((10, 28))
        
        for c in range(11):
            for i in range(2):
                for j in range(3):
                    U_full[i*3 + j, c] = d_232['U'][i*3 + j, c]
            for j in range(3):
                for k in range(2):
                    V_full[j*5 + k, c] = d_232['V'][j*2 + k, c]
            for i in range(2):
                for k in range(2):
                    W_full[k*2 + i, c] = d_232['W'][k*2 + i, c]

        for c in range(17):
            for i in range(2):
                for j in range(3):
                    U_full[i*3 + j, c + 11] = d_233['U'][i*3 + j, c]
            for j in range(3):
                for k in range(3):
                    V_full[j*5 + (2 + k), c + 11] = d_233['V'][j*3 + k, c]
            for i in range(2):
                for k in range(3):
                    W_full[(2 + k)*2 + i, c + 11] = d_233['W'][k*2 + i, c]

        return {'U': U_full, 'V': V_full, 'W': W_full}

    def verify_decomp(U, V, W):
        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.linalg.norm(T_calc - T_target) < 1e-8

    def discrete_search(r, num_restarts=10, max_iters=500):
        """
        Discrete local search for half-integer/integer solutions.
        """
        if time.time() > deadline - 15:
            return None
            
        nU, nV, nW = 6, 15, 10
        # Candidate values: small integers and half-integers
        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        best_err = np.inf
        best_state = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart)
            
            # Initialize with random candidates
            U = np.array(np.random.choice(candidates, size=(nU, r)))
            V = np.array(np.random.choice(candidates, size=(nV, r)))
            W = np.array(np.random.choice(candidates, size=(nW, r)))
            
            T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
            curr_err = np.sum((T_calc - T_target) ** 2)
            
            if curr_err < best_err:
                best_err = curr_err
                best_state = (U.copy(), V.copy(), W.copy())
                
            if curr_err < 1e-8:
                if verify_decomp(U, V, W):
                    return {'U': U, 'V': V, 'W': W}
            
            # Local search: try changing one element at a time
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Shuffle order of updates to avoid bias
                indices_U = np.random.permutation(nU * r)
                indices_V = np.random.permutation(nV * r)
                indices_W = np.random.permutation(nW * r)
                
                for idx in indices_U:
                    i, j = divmod(idx, r)
                    old_val = U[i, j]
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        U[i, j] = new_val
                        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                        new_err = np.sum((T_calc - T_target) ** 2)
                        if new_err < curr_err:
                            curr_err = new_err
                            improved = True
                            break
                        else:
                            U[i, j] = old_val
                            
                for idx in indices_V:
                    i, j = divmod(idx, r)
                    old_val = V[i, j]
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        V[i, j] = new_val
                        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                        new_err = np.sum((T_calc - T_target) ** 2)
                        if new_err < curr_err:
                            curr_err = new_err
                            improved = True
                            break
                        else:
                            V[i, j] = old_val
                            
                for idx in indices_W:
                    i, j = divmod(idx, r)
                    old_val = W[i, j]
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        W[i, j] = new_val
                        T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
                        new_err = np.sum((T_calc - T_target) ** 2)
                        if new_err < curr_err:
                            curr_err = new_err
                            improved = True
                            break
                        else:
                            W[i, j] = old_val

            if curr_err < best_err:
                best_err = curr_err
                best_state = (U.copy(), V.copy(), W.copy())
                
            if curr_err < 1e-8:
                if verify_decomp(U, V, W):
                    return {'U': U, 'V': V, 'W': W}

        # Check if best state is a valid solution
        if best_state is not None and best_err < 1e-8:
            U, V, W = best_state
            if verify_decomp(U, V, W):
                return {'U': U, 'V': V, 'W': W}
                
        return None

    # First, get the baseline rank-28 solution
    try:
        result = final_decomp()
        U, V, W = result['U'], result['V'], result['W']
        if verify_decomp(U, V, W):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = 28
        else:
            sb = schoolbook_decomp(2, 3, 5)
            best_result = {"U": sb['U'].tolist(), "V": sb['V'].tolist(), "W": sb['W'].tolist()}
            best_rank = 30
    except Exception:
        sb = schoolbook_decomp(2, 3, 5)
        best_result = {"U": sb['U'].tolist(), "V": sb['V'].tolist(), "W": sb['W'].tolist()}
        best_rank = 30

    # Try to find lower-rank decompositions using discrete search
    for r in range(27, 10, -1):
        if time.time() > deadline - 20:
            break
        
        res = discrete_search(r, num_restarts=5, max_iters=100)
        if res is not None:
            best_result = {"U": res['U'].tolist(), "V": res['V'].tolist(), "W": res['W'].tolist()}
            best_rank = r
            break

    return best_result

# EVOLVE-BLOCK-END
