import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Vectorized construction of the schoolbook decomposition
    # This approach uses NumPy broadcasting and advanced indexing instead of Python loops.
    
    # Create index arrays
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # We need to create matrices U, V, W of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # Each column c corresponds to a unique (i, j, k) triple.
    
    # Generate all (i, j, k) triples
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    
    # Flatten them to get the column index for each term
    # The order of columns doesn't matter for the decomposition, as long as it's consistent.
    # Let's map (i, j, k) to column c = i*m*p + j*p + k
    cols = (i_vals * m * p + j_vals * p + k_vals).flatten()
    
    # Flatten the indices for rows
    u_rows = (i_vals * m + j_vals).flatten()
    v_rows = (j_vals * p + k_vals).flatten()
    w_rows = (k_vals * n + i_vals).flatten()
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Set the 1s
    # U[u_rows, cols] = 1
    # Since cols are unique 0..rank-1 and u_rows are the corresponding rows,
    # we can use advanced indexing.
    U[u_rows, cols] = 1
    V[v_rows, cols] = 1
    W[w_rows, cols] = 1
    
    # Convert to lists of lists to match the expected output format
    # The task says "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't strictly require lists of lists, but the seed program returned lists.
    # The interface says `construct() -> {'U', 'V', 'W'}`.
    # I will return NumPy arrays or lists? The seed returned lists.
    # "Keep the function names and signatures... do not rename functions or change what they return."
    # The seed returned `{"U": U, "V": V, "W": W}` where U, V, W were lists of lists.
    # I should probably return lists of lists to be safe, or check if the evaluator accepts arrays.
    # Given "Python standard library, NumPy and SciPy are available", and the seed used lists,
    # I will convert back to lists to ensure compatibility.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
