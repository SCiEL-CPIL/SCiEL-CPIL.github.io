"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 48
    # We will try to find a lower rank decomposition.
    
    # The matrix multiplication tensor M_{n,m,p} has entries T_{i,j,k} = delta_{i,k} delta_{j,l} ... wait, standard notation:
    # (A * B)_{i,l} = sum_j A_{i,j} B_{j,l}
    # Tensor T has indices i (row of A), j (col of A / row of B), l (col of B)
    # T_{i,j,l} = 1 if we are looking at the (i,l) entry of the product, which depends on A_{i,j} and B_{j,l}.
    # So the tensor is T_{i,j,l} = 1 for all i,j,l? No.
    # The tensor of matrix multiplication is defined such that:
    # (A B)_{i,l} = sum_{j} A_{i,j} B_{j,l}
    # This can be written as a tensor contraction.
    # Let U_{i,j,c}, V_{j,l,c}, W_{l,i,c} be the factors.
    # The decomposition requires:
    # delta_{i,i'} delta_{l,l'} = sum_c U_{i,j,c} V_{j,l,c} W_{l',i',c} ? No.
    
    # Standard formulation:
    # We want to compute C_{i,l} = sum_j A_{i,j} B_{j,l}.
    # This is equivalent to finding tensors U, V, W such that:
    # T_{i,j,l} = sum_c U_{i,j,c} V_{j,l,c} W_{l,i,c}  -- this is not right either.
    
    # Let's use the standard definition from Strassen's paper or similar.
    # The tensor M_{n,m,p} is in U^* \otimes V^* \otimes W^*.
    # Its components are T_{i,j,k} where i indexes rows of A (size n), j indexes cols of A / rows of B (size m), k indexes cols of B (size p).
    # The multiplication is: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T has T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # Actually, T is a 3-tensor with indices i, j, k.
    # T_{i,j,k} = 1 for all i,j,k? No, that would be the all-ones tensor.
    # The matrix multiplication tensor is defined by:
    # T(A,B) = A B
    # In tensor notation, T \in U^* \otimes V^* \otimes W^* where dim(U)=n, dim(V)=m, dim(W)=p.
    # The action of T on A \in U, B \in V is (T(A,B))_w = sum_{u,v} T_{u,v,w} A_u B_v.
    # Here u corresponds to (i,j), v corresponds to (j,k), w corresponds to (i,k).
    # So the indices are:
    # U index: i (row of A), j (col of A) -> size n*m
    # V index: j (row of B), k (col of B) -> size m*p
    # W index: i (row of C), k (col of C) -> size n*p
    
    # The task says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)  <-- Note: p*n, not n*p.
    # Output factor follows W[k*n+i, r]. This means W is indexed by k (col of C) and i (row of C).
    # So W_{k*n+i, r} corresponds to the entry C_{i,k}.
    
    # The decomposition condition is:
    # For all i, j, k:
    # delta_{i,i'} delta_{k,k'} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k'*n+i', r} ? No.
    
    # Let's look at the schoolbook decomposition in the prompt:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # So the tensor T has components T_{(i,j), (j,k), (k,i)} = 1?
    # No, the indices of the tensor are (i,j) for U, (j,k) for V, and (k,i) for W?
    # Wait, W is indexed by k*n+i. So the third index is (k,i).
    # The tensor is T_{i,j,k} such that:
    # T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # Actually, the tensor is defined by its action:
    # (T \cdot (A \otimes B))_{(i,k)} = sum_{j} A_{i,j} B_{j,k}
    # This means the tensor T has components T_{(i,j), (j,k), (i,k)} = 1.
    # Let's verify with the schoolbook:
    # U_{(i,j), r} = 1, V_{(j,k), r} = 1, W_{(k,i), r} = 1.
    # Then sum_r U_{(i,j), r} V_{(j,k), r} W_{(k',i'), r} = 1 if (k,i) = (k',i'), else 0.
    # This gives delta_{k,k'} delta_{i,i'}.
    # This is exactly the identity tensor in the output space.
    # So the condition is:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k? No.
    # The sum is over r. The result is a tensor in U \otimes V \otimes W.
    # The component at ((i,j), (j,k), (k,i)) is sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # This must equal 1 for all i,j,k?
    # No, the tensor T is the matrix multiplication tensor.
    # T_{(i,j), (j,k), (i,k)} = 1.
    # All other components are 0.
    # But (i,j), (j,k), (i,k) are the only non-zero components?
    # Yes, because A_{i,j} and B_{j,k} only contribute to C_{i,k}.
    # So the tensor is:
    # T_{(i,j), (j,k), (i,k)} = 1
    # T_{(i',j'), (j'',k'), (k'',i'')} = 0 otherwise.
    
    # We need to find U, V, W such that:
    # sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = T_{(i,j), (j,k), (i,k)}
    # for all (i,j), (j,k), (k,i).
    
    # This is a rank minimization problem.
    # We can use the "border rank" or "approximate rank" techniques, but we need exact rank.
    # For small dimensions, we can use numerical methods to find a low-rank decomposition.
    
    # Strategy:
    # 1. Construct the full tensor T of size (n*m) x (m*p) x (p*n).
    # 2. Use a numerical algorithm to find a low-rank decomposition.
    #    - We can use the "tensor train" or "CP decomposition" algorithms.
    #    - Or we can use a simple iterative method: start with a random rank-r decomposition and optimize.
    
    # Given the time limit, we will try a few fixed ranks below 48 and see if we can find a valid decomposition.
    # Known results:
    # - 2x2x2 matrix multiplication has rank 7 (Strassen).
    # - 3x3x3 matrix multiplication has rank 23 (Landsberg).
    # - 4x4x4 matrix multiplication has rank 49? No, schoolbook is 64. Best known is 49.
    # But we have 3x4x4.
    
    # Let's try to use the Strassen-like approach for 2x2 blocks.
    # We can partition the 3x4 matrix into 2x2 blocks, but 3 is not divisible by 2.
    # So we can use a 2x2 block for the first 2 rows/cols and handle the 3rd row/col separately.
    
    # Alternatively, we can use a generic numerical solver.
    # Let's implement a simple CP decomposition solver using gradient descent or ALS.
    
    # Due to time constraints, we will first return the schoolbook decomposition.
    # Then, if we have time, we can try to improve it.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for the required output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
