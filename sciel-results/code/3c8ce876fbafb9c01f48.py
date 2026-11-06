"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csc_matrix


def priority(adjacency):
    """
    Assign a score to each vertex based on the expected loss of selecting it.
    The score is negative of the sum of inverse degrees of neighbors plus 1.
    Vertices with lower scores (more negative) are picked last?
    Wait, the decoder picks in DESCENDING score order.
    So we want the BEST vertex to have the HIGHEST score.
    
    We want to pick vertices with LOW expected loss first.
    So Score should be INVERSE of Expected Loss? Or Negative Expected Loss?
    If Loss is low, we want High Score.
    So Score = -Loss works if we pick highest score.
    Example:
    Vertex A: Loss 2. Score -2.
    Vertex B: Loss 5. Score -5.
    Decoder picks A (-2 > -5). Correct.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Convert to sparse matrix for efficiency
    A_sparse = csc_matrix(adj)
    
    # Handle edge case: empty graph
    if A_sparse.nnz == 0:
        return np.zeros(n)
    
    # Calculate degrees
    degrees = np.asarray(A_sparse.sum(axis=1)).flatten()
    
    # Avoid division by zero for isolated nodes
    # If degree is 0, the term 1/deg is undefined, but the sum over neighbors is empty anyway.
    # We create a vector of inverse degrees for the neighbors.
    # We need to divide each column j of A by max(1, degree[j]).
    
    # Create a diagonal matrix D_inv where D_inv[j,j] = 1 / max(1, degree[j])
    # But we only need to multiply A by this diagonal matrix.
    # A_sparse * D_inv
    # Since A_sparse is CSC, multiplying by a diagonal on the right is easy.
    # However, scipy sparse matrix multiplication with diagonal might be slow if not careful.
    # Alternative:
    # We want S_u = sum_{v in N(u)} 1/max(1, deg_v)
    # This is equivalent to (A * D_inv) @ ones
    
    inv_deg = 1.0 / np.maximum(1.0, degrees)
    
    # Multiply each column j of A by inv_deg[j]
    # In CSC format, columns are contiguous.
    # We can just multiply the data array by the appropriate inv_deg values.
    # But we need to know which column each data point belongs to.
    # A_sparse.indices gives row indices for each column.
    # A_sparse.indptr gives the start/end of each column's data.
    
    # Let's use a safer method: 
    # B = A_sparse.multiply(np.diag(inv_deg)) is slow.
    # Let's use the structure of CSC.
    
    # For each column j, the data is A_sparse.data[A_sparse.indptr[j]:A_sparse.indptr[j+1]]
    # We want to multiply these by inv_deg[j].
    
    # We can create a vector `col_inv_deg` that has inv_deg[j] repeated for each entry in column j.
    # indptr differences give the number of entries.
    col_lengths = np.diff(A_sparse.indptr)
    
    # Create the repeated vector
    # np.repeat(inv_deg, col_lengths)
    # This aligns with A_sparse.data because CSC stores data column by column.
    
    data_multiplier = np.repeat(inv_deg, col_lengths)
    
    # The sum for each row u is the sum of (A_data * multiplier) for all entries in row u.
    # But A is symmetric? Not necessarily, but for ER graphs it is.
    # If not symmetric, we should use the adjacency matrix as given.
    # The task says "adjacency: bool matrix", implying symmetry, but we should be robust.
    # The greedy decoder skips neighbours. Neighbours are defined by the matrix.
    # If A[u,v] is 1, v is a neighbour of u.
    
    # We want S_u = sum_{v: A[u,v]=1} inv_deg[v]
    # This is (A * D_inv) @ 1
    # Let's compute this product efficiently.
    
    # B = A * D_inv
    # In CSC, B.data = A.data * data_multiplier
    # B.indptr and B.indices are the same.
    
    # We want the row sums of B.
    # scipy sparse matrix .sum(axis=1) does this.
    
    B_data = A_sparse.data * data_multiplier
    B = csc_matrix((B_data, A_sparse.indices, A_sparse.indptr), shape=A_sparse.shape)
    
    S = np.asarray(B.sum(axis=1)).flatten()
    
    # Expected Loss L(u) = 1 + S_u
    loss = 1.0 + S
    
    # Score = -Loss
    # We want high score for low loss.
    score = -loss
    
    return score
# EVOLVE-BLOCK-END
