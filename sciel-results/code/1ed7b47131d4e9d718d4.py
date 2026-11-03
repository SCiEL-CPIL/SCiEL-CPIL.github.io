"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor T[i, j, k] should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check a random set of (i, j, k) triples
    # Actually, the tensor is T_{i,j,k} = delta_{i,j,k} only for the identity?
    # No. The matrix multiplication tensor T has entries T[i, j, k] = 1 if (i,j,k) corresponds to a term in C = AB.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor is a multilinear form: sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}.
    # In the decomposition, C_{i,k} = sum_r W_{k*n+i, r} (sum_s U_{i*m+s, r} A_{i,s}) (sum_t V_{s*p+k, r} B_{s,k})?
    # Wait. The standard tensor decomposition for matrix multiplication (n x m) (m x p) -> (n x p) is:
    # C_{i,k} = sum_{r=1}^R sum_{j=1}^m U_{i,j,r} A_{i,j} V_{j,k,r} B_{j,k} W_{i,k,r} ?
    # Let's look at the seed code.
    # Seed: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # Rank = n*m*p.
    # This corresponds to the schoolbook algorithm where each product A_{i,j} B_{j,k} is a separate term.
    # The reconstruction is: C_{i,k} = sum_c W_{k*n+i, c} * (sum_{i', j'} U_{i'*m+j', c} A_{i', j'}) * (sum_{j', k'} V_{j'*p+k', c} B_{j', k'})
    # In the seed, for a specific c corresponding to (i, j, k):
    # U_{i*m+j, c} = 1, others 0. So the first factor is A_{i,j}.
    # V_{j*p+k, c} = 1, others 0. So the second factor is B_{j,k}.
    # W_{k*n+i, c} = 1.
    # So C_{i,k} gets contribution 1 * A_{i,j} * B_{j,k} if c is the triple (i,j,k).
    # Summing over c, C_{i,k} = sum_j A_{i,j} B_{j,k}. This is correct.
    
    # To verify, we can check if the tensor matches.
    # T[i, j, k] is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the tensor definition used by these tasks, usually:
    # T_{i, j, k} = 1.
    # And the decomposition is: T_{i, j, k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # Let's verify this against the seed.
    # For c=(i0, j0, k0):
    # U[i*m+j, c] is 1 iff i=i0 and j=j0.
    # V[j*p+k, c] is 1 iff j=j0 and k=k0.
    # W[k*n+i, c] is 1 iff k=k0 and i=i0.
    # So the product is 1 iff i=i0, j=j0, k=k0.
    # Summing over c, for a fixed (i, j, k), there is exactly one c (i, j, k) where all three are 1.
    # So sum is 1.
    # Thus T_{i, j, k} = 1 for all i, j, k.
    
    # So we just need to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i, j, k.
    # Since n=2, m=5, p=6, total 60 entries.
    # We can construct the full tensor and check.
    
    T_expected = np.ones((n, m, p))
    T_computed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_computed[i, j, k] = s
    return np.allclose(T_computed, T_expected, atol=tol)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. Try to find a low-rank decomposition using a greedy/randomized approach or structured search.
    # 2. Given the small dimensions (2x5x6), we can try to exploit the structure.
    #    The rank is at most 60.
    #    Known lower bounds: max(nm, mp, np) = 12.
    #    Strassen for 2x2x2 is rank 7.
    #    We can view the 2x5x6 multiplication as a block multiplication.
    #    A is 2x5, B is 5x6.
    #    We can pad A to 2x6 and B to 6x6? Or 2x2 and 2x2 blocks?
    #    Since n=2, we can split A into A_left (2x2) and A_right (2x3).
    #    Split B into B_left (2x6) and B_right (3x6).
    #    C = A_left B_left + A_right B_right.
    #    This doesn't immediately reduce rank unless we use Strassen on the blocks.
    #    However, the blocks are not square.
    
    # Let's try a more direct approach: Randomized search for low rank.
    # Or, use the fact that the rank of (2x5x6) is likely close to the product of ranks of sub-problems.
    # But the best known rank for 2x5x6 is not trivially known.
    # Let's try to find a decomposition of rank R.
    # We can formulate it as a polynomial system.
    # For each (i, j, k): sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
    
    # Since the search space is huge, we might not find a rank < 60 easily.
    # However, the seed is 60.
    # Can we do better than 60?
    # For 2x2x2, Strassen gives 7 < 8.
    # For 2x5x6, we can embed 2x2x2.
    # Let's try to construct a decomposition using Strassen-like ideas.
    
    # Actually, let's just return the schoolbook decomposition if we can't find better quickly.
    # But we should try to find a better one.
    
    # A simple improvement: If we can find a rank R < 60, we win.
    # Let's try a few random initializations and use gradient descent to minimize the error.
    # But the objective is nonsmooth (we want exact equality).
    # We can minimize the sum of squares of the errors: sum_{i,j,k} (sum_r ... - 1)^2.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank using randomized search with gradient descent
    # We will try ranks from best_rank - 1 downwards, but that's slow.
    # Instead, let's try to find a rank R that is a multiple of some block size?
    # No, let's just try a few random restarts with R = best_rank - 1, best_rank - 2, etc.
    # Actually, reducing rank by 1 requires satisfying 60 constraints with 3*60 - 1 variables?
    # Degrees of freedom: 3*R - (scaling invariance).
    # For R=60, 180 vars, 60 constraints. Very overdetermined? No, 180 > 60.
    # For R=59, 177 vars, 60 constraints.
    # It is likely that a decomposition exists for R < 60.
    
    # Let's try R = 50, 40, 30, 20, 12.
    # We will use a simple coordinate descent or gradient descent.
    
    def try_rank(R, num_iters=100, num_restarts=5):
        best_err = float('inf')
        best_params = None
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                return None, None, None, float('inf')
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize to prevent overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(num_iters):
                if time.time() > deadline:
                    return None, None, None, float('inf')
                
                # Compute error and gradient
                # E = sum_{i,j,k} (sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} - 1)^2
                # This is expensive to compute fully.
                # We can use a stochastic approach: sample a few (i,j,k) and update.
                
                # Alternatively, compute the full error tensor
                T_comp = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_comp[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                
                err_tensor = T_comp - 1.0
                err = np.sum(err_tensor**2)
                
                if err < best_err:
                    best_err = err
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-9:
                    return U, V, W, err
                
                # Gradient update (simple step)
                # dU[i*m+j, r] = sum_{k} 2 * err_tensor[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                # dV[j*p+k, r] = sum_{i} 2 * err_tensor[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                # dW[k*n+i, r] = sum_{j} 2 * err_tensor[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
                
                lr = 0.01
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = err_tensor[i, j, k]
                            if abs(e) < 1e-12:
                                continue
                            dU[i*m+j] += 2 * e * V[j*p+k] * W[k*n+i]
                            dV[j*p+k] += 2 * e * U[i*m+j] * W[k*n+i]
                            dW[k*n+i] += 2 * e * U[i*m+j] * V[j*p+k]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Renormalize
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
        
        if best_params:
            return best_params + (best_err,)
        return None, None, None, float('inf')
    
    # Try a few ranks
    for R in [best_rank - 1, best_rank - 5, best_rank - 10, 50, 40, 30]:
        if time.time() > deadline - 20:
            break
        U, V, W, err = try_rank(R, num_iters=50, num_restarts=2)
        if U is not None and err < 1e-9:
            best_U, best_V, best_W = U, V, W
            best_rank = R
            # Try to go lower
            for R2 in [R - 1, R - 5]:
                if time.time() > deadline - 10:
                    break
                U2, V2, W2, err2 = try_rank(R2, num_iters=50, num_restarts=2)
                if U2 is not None and err2 < 1e-9:
                    best_U, best_V, best_W = U2, V2, W2
                    best_rank = R2
    
    # Convert to lists of integers/half-integers if possible
    # The task requires integer/half-integer entries.
    # If the solution is exact, the entries should be close to such values.
    # We can round them.
    
    U_list = [[0]*best_rank for _ in range(n*m)]
    V_list = [[0]*best_rank for _ in range(m*p)]
    W_list = [[0]*best_rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r in range(best_rank):
            val = best_U[i, r]
            # Round to nearest half-integer
            rounded = round(val * 2) / 2
            U_list[i][r] = rounded
    for j in range(m*p):
        for r in range(best_rank):
            val = best_V[j, r]
            rounded = round(val * 2) / 2
            V_list[j][r] = rounded
    for k in range(p*n):
        for r in range(best_rank):
            val = best_W[k, r]
            rounded = round(val * 2) / 2
            W_list[k][r] = rounded
    
    # Verify the rounded solution
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # If rounding failed, return the schoolbook (which is always valid)
        U, V, W = schoolbook()
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
