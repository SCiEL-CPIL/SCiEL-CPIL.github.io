"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the trivial rank-75 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
    return U, V, W


def _round_half(U, V, W):
    """Round entries to nearest half-integer."""
    return (np.round(2 * U) / 2.0,
            np.round(2 * V) / 2.0,
            np.round(2 * W) / 2.0)


def _verify(U, V, W):
    """Check exact decomposition: sum_c U[:,c] V[:,c] W[:,c] == target tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Target tensor T[i,j,k] = 1 if (i,j,k) in [n]x[m]x[p] else 0
    # Actually target is the matrix-mult tensor: T[i,j,k] = 1 for all valid indices.
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # Wait, the tensor for matrix multiplication A(n x m) * B(m x p) -> C(n x p) is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The trilinear form: sum_{i,j,k} A[i,j] B[j,k] C[k,i] (with appropriate indexing)
    # Actually the standard: the tensor T has entries T[i,j,k] such that
    # C[i,k] = sum_j A[i,j] B[j,k], so the multiplication tensor is
    # T = sum_{i,j,k} e_{i,k} ⊗ e_{i,j} ⊗ e_{j,k} ... 
    # More precisely, the rank of the tensor: we need
    # For the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is: sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}
    # So T[i,k, i,j, j,k] ... Let me use the standard form.
    # The decomposition: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c should equal 1 for each (i,j,k)
    # Wait no. Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n x m * m x p:
    # We want C[i,k] = sum_j A[i,j] B[j,k]
    # The trilinear form: sum_{i,j,k} A[i,j] * B[j,k] * C[k,i]  (note: C index order)
    # Actually C[i,k], so the third factor should give C[i,k].
    # W has shape (p*n, rank), indexed as W[k*n + i, c]
    # So W[k*n+i, c] corresponds to C[i,k] = C[k,i]? No, W[k*n+i] means index k*n+i.
    # Let me think: W row index = k*n + i, so for a given (k,i), W[k*n+i, c] is the coefficient.
    # The product: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Summed over c, this should equal the tensor entry T[i,j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Wait, that's not right either. The tensor is not all-ones.
    
    # Let me reconsider. The matrix multiplication map:
    # (A, B) -> C where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can write this as a trilinear form:
    # sum_{i,k,j} C[i,k] A[i,j] B[j,k]
    # 
    # The tensor T has entries T[i, j, k] such that the trilinear form is
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # (using C[k,i] to match W[k*n+i])
    # 
    # For this to equal sum_{i,k} C[i,k] (sum_j A[i,j] B[j,k]), we need:
    # T[i,j,k] = 1 if the indices are "consistent" in the right way.
    # Actually, T[i,j,k] = 1 for all i,j,k because:
    # sum_{i,j,k} A[i,j] B[j,k] C[k,i] with T[i,j,k]=1 gives
    # sum_i sum_j sum_k A[i,j] B[j,k] C[k,i] = sum_i,k C[k,i] (sum_j A[i,j] B[j,k])
    # = sum_i,k C[k,i] C_target[i,k] which is not what we want.
    
    # I think the correct interpretation is:
    # The tensor entries are T[i, j, k] = 1 for all valid i, j, k.
    # And the decomposition satisfies:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i, j, k
    #
    # Wait, that would mean the tensor is all-ones, which has rank 1, not 75.
    # That can't be right.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The standard matrix multiplication tensor T_{n,m,p} has entries:
    # T[i, j, k] = 1 if the indices form a valid "path" i->j->k, else 0.
    # For matrix multiplication A(n x m) * B(m x p):
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's all ones.
    
    # Actually, I think the tensor is defined such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{valid}
    # where the "valid" condition means this equals 1 for each (i,j,k).
    # But that's just the all-ones tensor which has rank 1.
    
    # I'm confused. Let me look at the schoolbook code:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) with unique c.
    # So sum_c ... = 1 for each (i,j,k) because only one c contributes.
    # So yes, the tensor is all-ones? That seems wrong for rank minimization.
    
    # OH WAIT. I think I misread the problem. The "matrix multiplication tensor" is NOT
    # the all-ones tensor. It's the tensor that encodes the bilinear map.
    # Let me reconsider the indexing.
    
    # Actually, I think the correct interpretation is:
    # The tensor T has T[i, j, k] = 1 for all i, j, k.
    # No, that's rank 1.
    
    # Let me just verify with the schoolbook: it sets each (i,j,k) to have its own column
    # with all ones. So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    # This IS the all-ones tensor. But the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # has rank 1, not n*m*p.
    
    # I think there's a misunderstanding in my tensor definition. Let me re-examine.
    
    # Perhaps the tensor is NOT all-ones. Perhaps:
    # T[i, j, k] = 1 if and only if the three indices are "compatible" in some way.
    # For matrix multiplication, the only constraint is that j is shared between A and B.
    # But in the flattened indices, i*m+j is the A index, j*p+k is the B index, k*n+i is the C index.
    # The "tensor" for matrix multiplication might actually be defined as:
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where j = j' and i = i' and k = k'
    # i.e., the indices must be consistent.
    
    # So T[a, b, c] = 1 if a, b, c are consistent (same i, j, k), else 0.
    # That means T is a sparse tensor with 1s only at (i*m+j, j*p+k, k*n+i) for valid i,j,k.
    # This tensor has rank... well, the schoolbook gives rank n*m*p by using one column per entry.
    # Can we do better? Yes, potentially, by using linear combinations.
    
    # So the correct check is:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other (a,b,c) combinations: sum_c U[a, c] * V[b, c] * W[c2, c] = 0
    # Wait no, the tensor is only defined at the "diagonal" entries.
    
    # Actually, I think the decomposition only needs to satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i, j, k
    # There are no other constraints because the tensor is implicitly defined by these equations.
    # The "tensor" is the multilinear map, and we just need to represent it.
    
    # Hmm, but if that's all we need, then the all-ones vectors would work with rank 1:
    # U = ones((n*m, 1)), V = ones((m*p, 1)), W = ones((p*n, 1))
    # Then sum_c = 1*1*1 = 1 for all (i,j,k). Rank 1!
    
    # That can't be right either. The rank of matrix multiplication is known to be between
    # various bounds, definitely not 1.
    
    # I think I'm fundamentally misunderstanding the tensor. Let me re-read the task once more.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix by a 5 by 5 matrix."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    
    # OK so I think the correct formulation is:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a BILINEAR map, not trilinear. To make it trilinear, we introduce C as a variable:
    # sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = 0 is NOT what we want.
    #
    # The standard tensor formulation: the matrix multiplication tensor T is defined such that
    # the rank of T equals the number of scalar multiplications needed.
    #
    # T is a tensor of size (n*p) x (n*m) x (m*p) with entries:
    # T[i*p+k, i*m+j, j*p+k] = 1 for all i, j, k, and 0 otherwise.
    #
    # But the task says W has shape (p*n, rank) with index W[k*n+i, c].
    # So W index is k*n + i, which corresponds to C[i,k] if we think of it as C[k*n+i].
    #
    # So the tensor entry is: T[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k.
    #
    # And the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # AND sum_c U[a, c] * V[b, c] * W[d, c] = 0 for all (a,b,d) not of the form (i*m+j, j'*p+k, k'*n+i') with j=j', i=i', k=k'.
    
    # So the full check is:
    # For all a in [n*m], b in [m*p], d in [p*n]:
    #   sum_c U[a,c] V[b,c] W[d,c] = 1 if (a,b,d) is a "consistent" triple, else 0.
    
    # A triple (a,b,d) is consistent if:
    #   a = i*m + j for some i, j
    #   b = j'*p + k for some j', k  
    #   d = k'*n + i' for some k', i'
    #   AND j = j', i = i', k = k'
    
    # Let me implement this check.
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Compute the product tensor
    # product[a, b, d] = sum_c U[a,c] * V[b,c] * W[d,c]
    # This is a 3D tensor of size nm x mp x pn
    
    # We can compute this efficiently using einsum or broadcasting
    # product[a, b, d] = sum_c U[a,c] V[b,c] W[d,c]
    
    # Use einsum: 'ac,bc,dc->abd'
    product = np.einsum('ac,bc,dc->abd', U, V, W)
    
    # Now check: for all i,j,k: product[i*m+j, j*p+k, k*n+i] == 1
    # And for all other (a,b,d): product[a,b,d] == 0
    
    # First check the diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                if abs(product[a, b, d] - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    # Create a set of consistent triples
    consistent = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                consistent.add((i * m + j, j * p + k, k * n + i))
    
    for a in range(nm):
        for b in range(mp):
            for d in range(pn):
                if (a, b, d) not in consistent:
                    if abs(product[a, b, d]) > 1e-9:
                        return False
    
    return True


def _random_try(U, V, W, target_rank, seed, max_iter=1000, tol=1e-6):
    """Try to find a decomposition of target_rank by random initialization + optimization."""
    np.random.seed(seed)
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def objective(x):
        U_flat = x[:nm * target_rank].reshape(nm, target_rank)
        V_flat = x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
        W_flat = x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
        
        # Compute product tensor
        product = np.einsum('ac,bc,dc->abd', U_flat, V_flat, W_flat)
        
        # Target: 1 at consistent positions, 0 elsewhere
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    error += (product[a, b, d] - 1.0) ** 2
        
        # Off-diagonal errors
        consistent = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    consistent.add((i * m + j, j * p + k, k * n + i))
        
        for a in range(nm):
            for b in range(mp):
                for d in range(pn):
                    if (a, b, d) not in consistent:
                        error += product[a, b, d] ** 2
        
        return error
    
    x0 = np.random.randn(nm * target_rank + mp * target_rank + pn * target_rank) * 0.5
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iter})
        if result.fun < tol:
            U_flat = result.x[:nm * target_rank].reshape(nm, target_rank)
            V_flat = result.x[nm * target_rank : nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_flat = result.x[nm * target_rank + mp * target_rank :].reshape(pn, target_rank)
            U_r, V_r, W_r = _round_half(U_flat, V_flat, W_flat)
            if _verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
    except Exception:
        pass
    return None


def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook()
    
    # Try to reduce rank
    for target_rank in range(74, 40, -1):
        if time.time() - start_time > deadline - 10:
            break
        result = _random_try(U0, V0, W0, target_rank, seed=target_rank * 42, max_iter=500)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
