"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per seed interface
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over all i, j, k to define each rank-1 term
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i (Note: Task says W[k*n+i, r] but standard trace is often k*n+i? 
    # Let's re-read carefully: "W[k*n+i,r]"
    # In the seed: W[k * n + i][column] = 1.
    # My derivation above used W[j*n+i] where j was col of B/row of C?
    # Let's check the indices in the seed code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i * m + j][column] = 1
    #       V[j * p + k][column] = 1
    #       W[k * n + i][column] = 1
    # 
    # Let's verify the tensor product:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # If we set U_{i,j,r}=1, V_{j,k,r}=1, W_{k,i,r}=1 for r corresponding to (i,j,k),
    # Then T_{i,j,k} gets a contribution of 1.
    # All other T_{i',j',k'} are 0 because the indices don't match.
    # So this constructs the standard matrix multiplication tensor.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    assert rank == n * m * p
    assert len(res["U"]) == n * m
    assert len(res["V"]) == m * p
    assert len(res["W"]) == p * n
    print("Construct successful")
