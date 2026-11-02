import numpy as np
import time
from scipy.optimize import least_squares
from scipy.optimize import linprog
import itertools

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from scipy.optimize import linprog
import itertools

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety
    
    def build_tensor_equations(R):
        """Build the linear system A*x = b for tensor decomposition.
        
        x = [U_flat, V_flat, W_flat] where each is R*nm, R*mp, R*pn
        
        The tensor equation: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        This is nonlinear, but we can use the fact that for rank-1 terms,
        each term contributes U[i,j,r]*V[j,k,r]*W[k,i,r].
        
        For MILP, we'll use a different approach: enumerate support patterns.
        """
        pass
    
    def try_rank_with_support_search(R, max_patterns=200):
        """Try to find a valid decomposition of rank R by searching over support patterns.
        
        For each rank-r term, the support is determined by which (i,j), (j,k), (k,i) 
        entries are non-zero. We search over sparse support patterns.
        """
        # For each rank term r, the U factor has non-zero entries at some (i,j) positions
        # V factor at some (j,k) positions, W factor at some (k,i) positions
        
        # Generate candidate support patterns
        # U supports: subsets of {0..nm-1}
        # V supports: subsets of {0..mp-1}
        # W supports: subsets of {0..pn-1}
        
        # For efficiency, limit to small supports
        max_support_size = 3
        
        U_supports = []
        V_supports = []
        W_supports = []
        
        for size in range(1, max_support_size + 1):
            for combo in itertools.combinations(range(nm), size):
                U_supports.append(set(combo))
            if time.time() - start_time > timeout:
                return None
        
        for size in range(1, max_support_size + 1):
            for combo in itertools.combinations(range(mp), size):
                V_supports.append(set(combo))
            if time.time() - start_time > timeout:
                return None
        
        for size in range(1, max_support_size + 1):
            for combo in itertools.combinations(range(pn), size):
                W_supports.append(set(combo))
            if time.time() - start_time > timeout:
                return None
        
        # For each combination of R support patterns, solve for values
        # This is still combinatorially large, so use random sampling
        
        n_attempts = max(50, max_patterns // 2)
        
        for attempt in range(n_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Randomly select support patterns
            U_supps = [np.random.choice(U_supports, size=R, replace=True)]
            V_supps = [np.random.choice(V_supports, size=R, replace=True)]
            W_supps = [np.random.choice(W_supps, size=R, replace=True)]
            
            U_supp_list = U_supps[0]
            V_supp_list = V_supps[0]
            W_supp_list = W_supps[0]
            
            # Build the linear system for this support pattern
            # Unknowns: values at non-zero positions
            # For each r, U has |U_supp[r]| unknowns, V has |V_supp[r]|, W has |W_supp[r]|
            
            # Total unknowns
            n_unknowns = sum(len(s) for s in U_supp_list) + \
                         sum(len(s) for s in V_supp_list) + \
                         sum(len(s) for s in W_supp_list)
            
            if n_unknowns == 0:
                continue
            
            # Map unknown indices
            U_idx = {}
            V_idx = {}
            W_idx = {}
            
            idx = 0
            for r in range(R):
                for pos in U_supp_list[r]:
                    U_idx[(r, pos)] = idx
                    idx += 1
            for r in range(R):
                for pos in V_supp_list[r]:
                    V_idx[(r, pos)] = idx
                    idx += 1
            for r in range(R):
                for pos in W_supp_list[r]:
                    W_idx[(r, pos)] = idx
                    idx += 1
            
            # Build equations: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # This is still nonlinear! The product of three variables.
            
            # Alternative: use the fact that we can fix some values and solve linearly
            # Or use a different approach: iterate over possible value assignments
            
            # For now, skip this and use a simpler approach
            pass
        
        return None
    
    def solve_with_fixed_support(R, U_supp_list, V_supp_list, W_supp_list):
        """Solve for values given fixed support patterns.
        
        Uses iterative linearization: fix two factors, solve linear for third.
        """
        # Initialize values
        U_vals = {}
        V_vals = {}
        W_vals = {}
        
        for r in range(R):
            for pos in U_supp_list[r]:
                U_vals[(r, pos)] = 1.0
            for pos in V_supp_list[r]:
                V_vals[(r, pos)] = 1.0
            for pos in W_supp_list[r]:
                W_vals[(r, pos)] = 1.0
        
        # Iterate: fix U and V, solve for W linearly
        for iteration in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Fix U and V, solve for W
            # For each (k,i), W[k,i,r] appears in equations where that (k,i) is in W_supp[r]
            # Equation: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
            
            # Build linear system for W
            n_w_unknowns = sum(len(s) for s in W_supp_list)
            if n_w_unknowns == 0:
                continue
            
            W_idx = {}
            idx = 0
            for r in range(R):
                for pos in W_supp_list[r]:
                    W_idx[(r, pos)] = idx
                    idx += 1
            
            # For each equation (i,j,k), coefficient of W[k,i,r] is U[i,j,r]*V[j,k,r]
            A = np.zeros((n*m*p, n_w_unknowns))
            b = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            pos_w = k * n + i
                            if (r, pos_w) in W_idx:
                                u_val = U_vals.get((r, i * m + j), 0.0)
                                v_val = V_vals.get((r, j * p + k), 0.0)
                                A[eq_idx, W_idx[(r, pos_w)]] = u_val * v_val
            
            # Solve A * w = b
            try:
                w_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Update W values
                for r in range(R):
                    for pos in W_supp_list[r]:
                        W_vals[(r, pos)] = w_sol[W_idx[(r, pos)]]
                        
            except:
                continue
            
            # Fix V and W, solve for U
            n_u_unknowns = sum(len(s) for s in U_supp_list)
            if n_u_unknowns == 0:
                continue
            
            U_idx = {}
            idx = 0
            for r in range(R):
                for pos in U_supp_list[r]:
                    U_idx[(r, pos)] = idx
                    idx += 1
            
            A = np.zeros((n*m*p, n_u_unknowns))
            b = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            pos_u = i * m + j
                            if (r, pos_u) in U_idx:
                                v_val = V_vals.get((r, j * p + k), 0.0)
                                w_val = W_vals.get((r, k * n + i), 0.0)
                                A[eq_idx, U_idx[(r, pos_u)]] = v_val * w_val
            
            try:
                u_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                for r in range(R):
                    for pos in U_supp_list[r]:
                        U_vals[(r, pos)] = u_sol[U_idx[(r, pos)]]
                        
            except:
                continue
            
            # Fix U and W, solve for V
            n_v_unknowns = sum(len(s) for s in V_supp_list)
            if n_v_unknowns == 0:
                continue
            
            V_idx = {}
            idx = 0
            for r in range(R):
                for pos in V_supp_list[r]:
                    V_idx[(r, pos)] = idx
                    idx += 1
            
            A = np.zeros((n*m*p, n_v_unknowns))
            b = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            pos_v = j * p + k
                            if (r, pos_v) in V_idx:
                                u_val = U_vals.get((r, i * m + j), 0.0)
                                w_val = W_vals.get((r, k * n + i), 0.0)
                                A[eq_idx, V_idx[(r, pos_v)]] = u_val * w_val
            
            try:
                v_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                for r in range(R):
                    for pos in V_supp_list[r]:
                        V_vals[(r, pos)] = v_sol[V_idx[(r, pos)]]
                        
            except:
                continue
        
        # Check if solution is valid and snap to half-integers
        U_flat = np.zeros(R * nm)
        V_flat = np.zeros(R * mp)
        W_flat = np.zeros(R * pn)
        
        for r in range(R):
            for pos in U_supp_list[r]:
                U_flat[r * nm + pos] = U_vals[(r, pos)]
            for pos in V_supp_list[r]:
                V_flat[r * mp + pos] = V_vals[(r, pos)]
            for pos in W_supp_list[r]:
                W_flat[r * pn + pos] = W_vals[(r, pos)]
        
        # Snap to half-integers
        U_rounded = np.round(U_flat * 2) / 2
        V_rounded = np.round(V_flat * 2) / 2
        W_rounded = np.round(W_flat * 2) / 2
        
        # Verify
        U3 = U_rounded.reshape(n, m, R)
        V3 = V_rounded.reshape(m, p, R)
        W3 = W_rounded.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        
        if np.allclose(T, 1.0, atol=1e-10):
            return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from lower bound upward
    for R in range(18, 31):
        if time.time() - start_time > timeout:
            break
        
        # Try several random support patterns
        n_patterns = 30 if R < 22 else 10
        
        for pattern_idx in range(n_patterns):
            if time.time() - start_time > timeout:
                break
            
            # Generate random sparse support patterns
            U_supp_list = []
            V_supp_list = []
            W_supp_list = []
            
            for r in range(R):
                # Random small support for U
                n_u_nonzero = np.random.randint(1, 4)
                u_positions = np.random.choice(nm, n_u_nonzero, replace=False)
                U_supp_list.append(set(u_positions.tolist()))
                
                # Random small support for V
                n_v_nonzero = np.random.randint(1, 4)
                v_positions = np.random.choice(mp, n_v_nonzero, replace=False)
                V_supp_list.append(set(v_positions.tolist()))
                
                # Random small support for W
                n_w_nonzero = np.random.randint(1, 4)
                w_positions = np.random.choice(pn, n_w_nonzero, replace=False)
                W_supp_list.append(set(w_positions.tolist()))
            
            result = solve_with_fixed_support(R, U_supp_list, V_supp_list, W_supp_list)
            
            if result is not None:
                U_rounded, V_rounded, W_rounded = result
                return {
                    "U": U_rounded.reshape(nm, R).tolist(),
                    "V": V_rounded.reshape(mp, R).tolist(),
                    "W": W_rounded.reshape(pn, R).tolist()
                }
        
        # Also try the original least_squares approach for this rank
        for attempt in range(10):
            if time.time() - start_time > timeout:
                break
            
            x0 = np.random.uniform(-1, 1, R * (nm + mp + pn))
            
            def residual(params, R):
                size_U = nm * R
                size_V = mp * R
                size_W = pn * R
                
                U_flat = params[:size_U]
                V_flat = params[size_U:size_U+size_V]
                W_flat = params[size_U+size_V:]
                
                U3 = U_flat.reshape(n, m, R)
                V3 = V_flat.reshape(m, p, R)
                W3 = W_flat.reshape(p, n, R)
                
                T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
                return (T - 1.0).flatten()
            
            try:
                result = least_squares(residual, x0, args=(R,), method='lm', max_nfev=200)
                
                if result.cost < 1e-4:
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    U_rounded = np.round(U_flat * 2) / 2
                    V_rounded = np.round(V_flat * 2) / 2
                    W_rounded = np.round(W_flat * 2) / 2
                    
                    U3 = U_rounded.reshape(n, m, R)
                    V3 = V_rounded.reshape(m, p, R)
                    W3 = W_rounded.reshape(p, n, R)
                    
                    T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
                    
                    if np.allclose(T, 1.0, atol=1e-10):
                        return {
                            "U": U_rounded.reshape(nm, R).tolist(),
                            "V": V_rounded.reshape(mp, R).tolist(),
                            "W": W_rounded.reshape(pn, R).tolist()
                        }
                        
            except Exception:
                continue
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
