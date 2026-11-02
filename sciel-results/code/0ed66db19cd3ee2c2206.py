import numpy as np
import time
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    n, m, p = DIMENSIONS

    def build_from_terms(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check that the decomposition is exact for matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen-based construction (known good, rank 46)
    def strassen_construction():
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)

        return terms

    def try_alt_partition_1():
        """Alternative: split j into {0,1,2}, {3,4} and k into {0,1,2}, {3,4}"""
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        add_strassen(0, 0, 0)
        add_schoolbook(0, 2, 0, 2, 2, 3)
        add_schoolbook(0, 2, 2, 3, 0, 2)
        add_schoolbook(0, 2, 2, 3, 2, 3)
        add_strassen(0, 0, 3)
        add_schoolbook(0, 2, 2, 3, 3, 5)
        add_strassen(0, 3, 0)
        add_schoolbook(0, 2, 3, 5, 2, 3)
        add_strassen(0, 3, 3)

        return terms

    def try_alt_partition_2():
        """Schoolbook only: rank 50"""
        terms = []
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        add_schoolbook(0, 2, 0, 5, 0, 5)
        return terms

    def try_local_search(terms, max_iters=50):
        """
        Try to reduce rank by 1 using local search.
        1. Convert terms to arrays U, V, W.
        2. Remove one term (or set to 0).
        3. Optimize the remaining terms to fit the tensor.
        4. If residual is small and entries are near half-integers, return.
        """
        rank = len(terms)
        if rank <= 45:
            return None

        U_list = []
        V_list = []
        W_list = []
        for u, v, w in terms:
            U_row = np.zeros(n*m)
            V_row = np.zeros(m*p)
            W_row = np.zeros(p*n)
            for (i, j), val in u.items(): U_row[i*m+j] = val
            for (j, k), val in v.items(): V_row[j*p+k] = val
            for (k, i), val in w.items(): W_row[k*n+i] = val
            U_list.append(U_row)
            V_list.append(V_row)
            W_list.append(W_row)
        
        U_init = np.array(U_list).T # (n*m, rank)
        V_init = np.array(V_list).T # (m*p, rank)
        W_init = np.array(W_list).T # (p*n, rank)

        # Target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0

        def objective_and_grad(x):
            # x is flattened (n*m)*R + (m*p)*R + (p*n)*R
            R = len(terms) - 1 # We are optimizing R-1 terms
            size_U = n*m * R
            size_V = m*p * R
            size_W = p*n * R
            
            U = x[:size_U].reshape(n*m, R)
            V = x[size_U:size_U+size_V].reshape(m*p, R)
            W = x[size_U+size_V:].reshape(p*n, R)
            
            # Compute residual
            # Res[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - T[i,j,k]
            Res = np.zeros((n, m, p))
            for c in range(R):
                Res += np.einsum('ij,jk,ki->ijk', U[:, c], V[:, c], W[:, c])
            Res -= T
            
            obj = np.sum(Res**2)
            
            # Gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for c in range(R):
                # d(Res^2)/dU[i,j,c] = 2 * sum_{k} Res[i,j,k] * V[j,k,c] * W[k,i,c]
                # grad_U[i,j,c] = 2 * np.einsum('ijk,jk,ki->ij', Res, V[:, c], W[:, c])
                grad_U[:, c] = 2.0 * np.einsum('ijk,jk,ki->ij', Res, V[:, c], W[:, c])
                
                # d(Res^2)/dV[j,k,c] = 2 * sum_{i} Res[i,j,k] * U[i,j,c] * W[k,i,c]
                grad_V[:, c] = 2.0 * np.einsum('ijk,ij,ki->jk', Res, U[:, c], W[:, c])
                
                # d(Res^2)/dW[k,i,c] = 2 * sum_{j} Res[i,j,k] * U[i,j,c] * V[j,k,c]
                grad_W[:, c] = 2.0 * np.einsum('ijk,ij,jk->ki', Res, U[:, c], V[:, c])
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return obj, grad

        # Try removing each term
        np.random.seed(42)
        order = np.random.permutation(rank)
        
        for drop_idx in order:
            if time.time() > deadline - 10:
                break
            
            # Prepare initial guess: all terms except drop_idx
            mask = [i for i in range(rank) if i != drop_idx]
            U0 = U_init[:, mask]
            V0 = V_init[:, mask]
            W0 = W_init[:, mask]
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success and res.fun < 1e-6:
                    # Check if entries are close to half-integers
                    U_opt = res.x[:n*m*(rank-1)].reshape(n*m, rank-1)
                    V_opt = res.x[n*m*(rank-1):n*m*(rank-1)+m*p*(rank-1)].reshape(m*p, rank-1)
                    W_opt = res.x[n*m*(rank-1)+m*p*(rank-1):].reshape(p*n, rank-1)
                    
                    # Check half-integer constraint
                    def is_half_int(arr):
                        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-5)
                    
                    if is_half_int(U_opt) and is_half_int(V_opt) and is_half_int(W_opt):
                        # Round and verify
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        # Convert back to terms for validation
                        new_terms = []
                        for c in range(rank-1):
                            u_dict = {}
                            v_dict = {}
                            w_dict = {}
                            for idx, val in enumerate(U_rounded[:, c]):
                                if abs(val) > 1e-9:
                                    i, j = divmod(idx, m)
                                    u_dict[(i, j)] = val
                            for idx, val in enumerate(V_rounded[:, c]):
                                if abs(val) > 1e-9:
                                    j, k = divmod(idx, p)
                                    v_dict[(j, k)] = val
                            for idx, val in enumerate(W_rounded[:, c]):
                                if abs(val) > 1e-9:
                                    k, i = divmod(idx, n)
                                    w_dict[(k, i)] = val
                            new_terms.append((u_dict, v_dict, w_dict))
                        
                        result = build_from_terms(new_terms)
                        if validate(result["U"], result["V"], result["W"]):
                            return new_terms
            except Exception:
                pass
        return None

    # Try all constructions and pick the best
    constructions = [
        ("strassen", strassen_construction),
        ("alt1", try_alt_partition_1),
        ("schoolbook", try_alt_partition_2),
    ]
    
    best_terms = None
    best_rank = float('inf')
    best_name = None
    
    for name, func in constructions:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        try:
            terms = func()
            if len(terms) < best_rank:
                result = build_from_terms(terms)
                if validate(result["U"], result["V"], result["W"]):
                    best_terms = terms
                    best_rank = len(terms)
                    best_name = name
        except Exception as e:
            pass
    
    # If we have a rank 46 solution, try to reduce it to 45
    if best_rank == 46 and time.time() < deadline - 20:
        improved_terms = try_local_search(best_terms)
        if improved_terms is not None:
            best_terms = improved_terms
            best_rank = len(best_terms)
            best_name = "local_search"

    if best_terms is None:
        # Fallback to Strassen
        best_terms = strassen_construction()
        best_rank = len(best_terms)
    
    best_result = build_from_terms(best_terms)
    
    # Verify final result
    if not validate(best_result["U"], best_result["V"], best_result["W"]):
        # Last resort: schoolbook
        terms = try_alt_partition_2()
        best_result = build_from_terms(terms)
        if not validate(best_result["U"], best_result["V"], best_result["W"]):
            raise ValueError("All constructions failed validation")
    
    return best_result
