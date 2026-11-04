import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s deadline, 5s margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    def check_solution(U, V, W, tol=1e-6):
        """Check if the tensor decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # For all i, j, k: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
        # For all other index combos: sum = 0
        # Full constraint: sum_r W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2)*delta(k,k2)*delta(j,j2)
        for i in range(n):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                val = 0.0
                                for r in range(rank):
                                    val += W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r]
                                expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True
    
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    def compute_residuals(x, rank):
        """Compute all constraint residuals from flattened parameters."""
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        residuals = []
        for i in range(n):
            for i2 in range(n):
                for j in range(m):
                    for j2 in range(m):
                        for k in range(p):
                            for k2 in range(p):
                                val = 0.0
                                for r in range(rank):
                                    val += W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r]
                                expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                                residuals.append(val - expected)
        return np.array(residuals)
    
    def objective(x, rank):
        """Sum of squared residuals."""
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        # Compute all constraints
        # sum_r W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r]
        # Vectorize using einsum
        # T[i2, j, j2, k2, i, k] = sum_r W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r]
        # We need T[i2,j,j2,k2,i,k] = delta(i,i2)*delta(k,k2)*delta(j,j2)
        
        # Reshape W to (k2, i2, r), U to (i2, j, r), V to (j2, k2, r)
        W_reshaped = W.reshape(p, n, rank)  # W[k2*n+i2, r] -> W_reshaped[k2, i2, r]
        U_reshaped = U.reshape(n, m, rank)  # U[i2*m+j, r] -> U_reshaped[i2, j, r]
        V_reshaped = V.reshape(m, p, rank)  # V[j2*p+k2, r] -> V_reshaped[j2, k2, r]
        
        # T[i2, j, j2, k2] = sum_r W_reshaped[k2, i2, r] * U_reshaped[i2, j, r] * V_reshaped[j2, k2, r]
        # Using einsum: T[i2,j,j2,k2] = sum_r W[k2,i2,r] * U[i2,j,r] * V[j2,k2,r]
        T = np.einsum('kir,jir,lkr->ijkl', W_reshaped, U_reshaped, V_reshaped)
        # Wait, that's not right. Let me be more careful.
        # W_reshaped[k2, i2, r], U_reshaped[i2, j, r], V_reshaped[j2, k2, r]
        # T[i2, j, j2, k2] = sum_r W_reshaped[k2, i2, r] * U_reshaped[i2, j, r] * V_reshaped[j2, k2, r]
        T = np.einsum('kir,jir,lkr->ijkl', W_reshaped, U_reshaped, V_reshaped)
        # Here: k->k2, i->i2, r->r for W; j->j, i->i2, r->r for U; l->j2, k->k2, r->r for V
        # Output: i2, j, j2, k2
        
        # Expected: E[i2, j, j2, k2] = 1 if there exist i,k such that i2==i, k2==k, j==j2
        # Actually the constraint is: for the "diagonal" terms where i2=i, k2=k, j=j2, sum should be 1
        # For all other combinations, sum should be 0
        # The constraint is: sum_r W[k*n+i, r]*U[i*m+j, r]*V[j*p+k, r] = 1 for all i,j,k
        # And for any "off-diagonal" combination, it should be 0.
        
        # Let me reconsider. The full set of constraints:
        # For all (i, i2, j, j2, k, k2): 
        #   sum_r W[k2*n+i2, r] * U[i2*m+j, r] * V[j2*p+k2, r] = delta(i,i2)*delta(j,j2)*delta(k,k2)
        
        # So T[i2, j, j2, k2] (which is sum_r W[k2*n+i2,r]*U[i2*m+j,r]*V[j2*p+k2,r]) should equal:
        # delta(i,i2)*delta(j,j2)*delta(k,k2) for some i,k
        # But i and k are free variables in the constraint... 
        # Actually, the constraint says: for ALL i, i2, j, j2, k, k2, the sum equals delta(i,i2)*delta(j,j2)*delta(k,k2)
        # This means: if i2 != i OR j2 != j OR k2 != k, the sum must be 0.
        # If i2 == i AND j2 == j AND k2 == k, the sum must be 1.
        
        # So for fixed (i2, j, j2, k2): the sum should be 1 if there exists (i,k) with i2==i, k2==k, j==j2,
        # which is always true when j==j2 (since we can pick i=i2, k=k2). Wait no.
        
        # Let me re-read: the constraint is that for ALL six indices, the sum equals the product of deltas.
        # So: sum_r W[k2*n+i2,r]*U[i2*m+j,r]*V[j2*p+k2,r] = 1 iff i2==i AND j2==j AND k2==k
        # Since i and k range over all values, this means:
        # - When j2==j: sum_r W[k2*n+i2,r]*U[i2*m+j,r]*V[j2*p+k2,r] = 1 (for any i2,k2, since we can set i=i2, k=k2)
        # Wait, that's not right. The delta is delta(i,i2)*delta(j,j2)*delta(k,k2).
        # For a fixed (i2,j,j2,k2), the RHS depends on i and k.
        # The constraint must hold for ALL i, k as well.
        # So for fixed (i2,j,j2,k2) and for ALL i,k:
        #   sum_r W[k2*n+i2,r]*U[i2*m+j,r]*V[j2*p+k2,r] = delta(i,i2)*delta(j,j2)*delta(k,k2)
        # LHS doesn't depend on i or k! So this can only be satisfied if:
        # - delta(j,j2) = 0, i.e., j != j2, in which case RHS = 0 for all i,k, so LHS must be 0.
        # - If j == j2, then RHS = delta(i,i2)*delta(k,k2), which varies with i,k.
        #   But LHS is constant in i,k. So this can only work if delta(i,i2)*delta(k,k2) is constant in i,k,
        #   which is impossible unless n=1 and p=1.
        
        # I think I'm misunderstanding the constraint. Let me re-derive from scratch.
        
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_l V[l*p+k, r] B[l,k])
        # = sum_r sum_j sum_l W[k*n+i, r] U[i*m+j, r] V[l*p+k, r] A[i,j] B[l,k]
        
        # For this to equal sum_j A[i,j] B[j,k] for all A, B:
        # Coefficient of A[i',j'] B[l',k'] in C[i,k] must be delta(i',i)*delta(l',k)*delta(j',l')
        # From decomposition, coefficient of A[i',j'] B[l',k'] in C[i,k] is:
        #   sum_r W[k*n+i, r] U[i*m+j', r] V[l'*p+k, r]
        # So we need: sum_r W[k*n+i, r] U[i*m+j', r] V[l'*p+k, r] = delta(i',i)*delta(k',k)*delta(j',l')
        # Wait, A[i',j'] and B[l',k'] appear in C[i,k] only if i'=i and k'=k (since U depends on i and V depends on k).
        # More carefully: the term A[i',j'] appears in the sum over j only when the index matches.
        # Let me just collect: C[i,k] = sum_r sum_j sum_l W[k*n+i,r] U[i*m+j,r] V[l*p+k,r] A[i,j] B[l,k]
        # The coefficient of A[a,b] B[c,d] in C[i,k] is:
        #   sum_r W[k*n+i,r] U[i*m+b,r] V[c*p+k,r] * (1 if a==i else 0) * (1 if d==k else 0)
        # = delta(a,i)*delta(d,k) * sum_r W[k*n+i,r] U[i*m+b,r] V[c*p+k,r]
        
        # For C = A*B, the coefficient of A[a,b] B[c,d] in C[i,k] is:
        #   delta(a,i)*delta(c,b)*delta(d,k)
        
        # So we need: delta(a,i)*delta(d,k) * sum_r W[k*n+i,r] U[i*m+b,r] V[c*p+k,r] = delta(a,i)*delta(c,b)*delta(d,k)
        # For all a,b,c,d,i,k.
        # When a==i and d==k: sum_r W[k*n+i,r] U[i*m+b,r] V[c*p+k,r] = delta(b,c)
        # When a!=i or d!=k: both sides are 0, so no constraint.
        
        # So the actual constraints are: for all i in [0,n), k in [0,p), b in [0,m), c in [0,m):
        #   sum_r W[k*n+i,r] U[i*m+b,r] V[c*p+k,r] = delta(b,c)
        
        # That's n*p*m*m = 2*9*3*3 = 162 constraints! Much simpler than I thought.
        
        pass  # Will implement below
    
    # Simplified constraint check
    def check_solution_simple(U, V, W, tol=1e-6):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        val = 0.0
                        for r in range(rank):
                            val += W[k*n+i, r] * U[i*m+b, r] * V[c*p+k, r]
                        expected = 1.0 if b == c else 0.0
                        if abs(val - expected) > tol:
                            return False
        return True
    
    def objective_simple(x, rank):
        n, m, p = DIMENSIONS
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        U_reshaped = U.reshape(n, m, rank)  # U[i, b, r]
        V_reshaped = V.reshape(m, p, rank)  # V[c, k, r]
        W_reshaped = W.reshape(p, n, rank)  # W[k, i, r]
        
        # T[i, b, c, k] = sum_r W[k,i,r] * U[i,b,r] * V[c,k,r]
        T = np.einsum('kir,ibr,ckr->ibck', W_reshaped, U_reshaped, V_reshaped)
        # T[i,b,c,k] should equal delta(b,c)
        
        # Residuals: T[i,b,c,k] - delta(b,c)
        residuals = np.zeros((n, m, m, p))
        for i in range(n):
            for k in range(p):
                for b in range(m):
                    for c in range(m):
                        residuals[i, b, c, k] = T[i, b, c, k] - (1.0 if b == c else 0.0)
        
        return np.sum(residuals ** 2)
    
    def gradient_simple(x, rank):
        """Compute gradient of objective using finite differences or analytic."""
        # Use central differences for simplicity
        eps = 1e-5
        grad = np.zeros_like(x)
        n_params = len(x)
        for idx in range(n_params):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            grad[idx] = (objective_simple(x_plus, rank) - objective_simple(x_minus, rank)) / (2 * eps)
        return grad
    
    # Try ranks from lower bound up
    lower_bound = max(n*p, m*p, n*m)
    
    for target_rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        n_params = (n*m + m*p + p*n) * target_rank
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            np.random.seed(attempt * 1000 + target_rank)
            U_init = np.random.randn(n*m, target_rank) * 0.5
            V_init = np.random.randn(m*p, target_rank) * 0.5
            W_init = np.random.randn(p*n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Try L-BFGS-B
            try:
                result = minimize(objective_simple, x0, args=(target_rank,), method='L-BFGS-B',
                                  jac=None, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if result.fun < 1e-8:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = x_opt[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = x_opt[n*m*target_rank+m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_h = to_half_int(U_opt)
                    V_h = to_half_int(V_opt)
                    W_h = to_half_int(W_opt)
                    
                    if check_solution_simple(U_h, V_h, W_h, tol=1e-6):
                        best_rank = target_rank
                        U, V, W = U_h, V_h, W_h
                        break
                    else:
                        # Try without rounding
                        if check_solution_simple(U_opt, V_opt, W_opt, tol=1e-6):
                            best_rank = target_rank
                            U, V, W = U_opt, V_opt, W_opt
                            break
            except Exception:
                pass
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
