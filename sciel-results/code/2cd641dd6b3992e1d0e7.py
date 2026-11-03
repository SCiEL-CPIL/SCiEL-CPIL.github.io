import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 120s limit, aim for 110s safety margin
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists of floats (or ints)
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    # Construct schoolbook decomposition
    # The tensor for matrix multiplication C = AB is T_{i,j,k} = A_{i,l} B_{l,k}
    # Decomposition: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # For schoolbook, each term corresponds to a specific index triple (i,j,k) in the summation index?
    # No, schoolbook is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Here i in [0,n), k in [0,p), j in [0,m).
    # The output W index is given as W[k*n+i, r].
    # The input U index is U[i*m+j, r].
    # The input V index is V[j*p+k, r].
    
    # We need to match A_{i,j} * B_{j,k} to produce C_{i,k}.
    # So for each term r corresponding to (i, j, k):
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1
    # W[k*n+i, r] = 1
    
    # Let's verify indices:
    # U row: i*m+j. i in 0..n-1, j in 0..m-1. Correct.
    # V row: j*p+k. j in 0..m-1, k in 0..p-1. Correct.
    # W row: k*n+i. k in 0..p-1, i in 0..n-1. Correct.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    if time.time() > deadline:
        # If we somehow took too long (unlikely for this construction), return what we have
        pass

    # Optional: Verification step to ensure correctness
    # We can check a random subset or the full tensor if fast enough
    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # This should equal 1 if the index mapping in the problem statement implies
    # that the tensor T is defined such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}?
    # No, the standard tensor for matmul M_{n,m,p} has entries T_{i,j,k}
    # such that (AB)_{i,k} = sum_l A_{i,l} B_{l,k}.
    # Wait, the indices in U, V, W are mixed.
    # Let's re-read the contract carefully.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # "U (n*m, rank), V (m*p, rank), W (p*n, rank)"
    
    # Standard decomposition:
    # (AB)_{i,k} = sum_{l=0}^{m-1} A_{i,l} B_{l,k}
    # We want sum_{r} U_{i,l,r} V_{l,k,r} W_{k,i,r} ? No, W is p*n.
    # If W is indexed by (k, i), then the term is U_{i,l,r} V_{l,k,r} W_{k,i,r}.
    # This would sum over r.
    # For schoolbook:
    # Term r corresponds to a specific l? No, a specific (i, l, k) triple?
    # If we set U_{i,l,r}=1, V_{l,k,r}=1, W_{k,i,r}=1 for r=(i,l,k),
    # Then sum_r U_{i,l',r} V_{l',k',r} W_{k',i',r}
    # Non-zero only if i=i', l=l', k=k'.
    # Then term is 1 * 1 * 1 = 1.
    # This gives (AB)_{i,k} = sum_l (1 if l matches the summation index? No.)
    
    # Let's trace:
    # We want C_{i,k} = sum_l A_{i,l} B_{l,k}.
    # The decomposition is C_{i,k} = sum_r (sum_l A_{i,l} U_{i,l,r}) (sum_l B_{l,k} V_{l,k,r}) W_{k,i,r} ?
    # No, the definition is usually:
    # C_{i,k} = sum_r (U^r A)_{i, ?} ...
    
    # Let's look at the shapes again.
    # U: (n*m, rank). Index (i*m+j).
    # V: (m*p, rank). Index (j*p+k).
    # W: (p*n, rank). Index (k*n+i).
    
    # If we sum over r:
    # C_{i,k} = sum_r ( sum_j A_{i,j} U_{i*m+j, r} ) * ( sum_j B_{j,k} V_{j*p+k, r} ) * W_{k*n+i, r}
    # This structure is:
    # Let U_r be a vector of length n*m (reshaped to n x m).
    # Let V_r be a vector of length m*p (reshaped to m x p).
    # Let W_r be a scalar (or vector of length p*n, reshaped to p x n).
    # Actually, W has shape (p*n, rank). So for each r, W[:, r] is a vector of length p*n.
    # Reshaped to p x n.
    
    # So, C_{i,k} = sum_r W_r[k, i] * (sum_j A_{i,j} U_r[i, j]) * (sum_j B_{j,k} V_r[j, k]) ?
    # This doesn't look like standard matrix multiplication unless U_r and V_r are deltas.
    
    # Let's test the schoolbook assignment again with this formula:
    # For r corresponding to (i0, j0, k0):
    # U_r[i, j] = 1 if i=i0, j=j0 else 0.
    # V_r[j, k] = 1 if j=j0, k=k0 else 0.
    # W_r[k, i] = 1 if k=k0, i=i0 else 0.
    
    # Term for r=(i0, j0, k0) in the sum for C_{i, k}:
    # W_{k0, i0} * (sum_j A_{i, j} U_{i0, j}) * (sum_j B_{j, k} V_{j, k0})
    # = (1 if k=k0, i=i0 else 0) * (A_{i, j0} if i=i0 else 0) * (B_{j0, k} if k=k0 else 0)
    # = (1 if i=i0 and k=k0) * A_{i0, j0} * B_{j0, k0}
    # = A_{i0, j0} B_{j0, k0} if i=i0, k=k0.
    
    # Summing over all r=(i0, j0, k0):
    # C_{i, k} = sum_{i0, j0, k0} [ (1 if i=i0 and k=k0) A_{i0, j0} B_{j0, k0} ]
    # = sum_{j0} A_{i, j0} B_{j0, k}
    # = (AB)_{i, k}.
    
    # So the schoolbook construction IS correct for this interface.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
