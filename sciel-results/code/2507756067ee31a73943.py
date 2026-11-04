import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Constructs a rank-50 decomposition for the 2x5 by 5x5 matrix multiplication tensor.
    
    This implementation uses a direct Python loop to generate the factor matrices 
    U, V, and W without relying on NumPy for the final output structure, ensuring 
    exact integer entries and fast execution. It models the multiplication tensor 
    T_{i,j,k} = \sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    
    The 'schoolbook' decomposition assigns a unique rank index r to each (i, j, k) triple.
    """
    n, m, p = DIMENSIONS
    # Total dimensions for the matrix multiplication problem
    dim_a = n * m  # 10
    dim_b = m * p  # 25
    dim_c = p * n  # 10
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists of zeros
    U = [[0] * rank for _ in range(dim_a)]
    V = [[0] * rank for _ in range(dim_b)]
    W = [[0] * rank for _ in range(dim_c)]
    
    # Generate the decomposition
    # For each (i, j, k), we create a rank slice that isolates the term A[i,j] * B[j,k]
    # This contributes to C[i,k]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U row index: i * m + j
                u_row = i * m + j
                # V row index: j * p + k
                v_row = j * p + k
                # W row index: k * n + i
                w_row = k * n + i
                
                # Set the values to 1 at the specific rank index r
                U[u_row][r] = 1
                V[v_row][r] = 1
                W[w_row][r] = 1
                
                r += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Constructed rank 50 decomposition.")
