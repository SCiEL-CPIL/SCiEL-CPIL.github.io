"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a block-partitioned matrix multiplication decomposition.
    Splits the 3x8 matrix into two 3x4 blocks and attempts rank minimization.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Split p=8 into two blocks of size 4
    p1 = 4
    p2 = 4
    
    # Schoolbook rank for each block
    rank_block = n * m * p1  # 2*3*4 = 24
    
    # Try to find lower rank decomposition for each block
    # Start with schoolbook for each block
    U1 = [[0] * rank_block for _ in range(n * m)]
    V1 = [[0] * rank_block for _ in range(m * p1)]
    W1 = [[0] * rank_block for _ in range(p1 * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U1[i * m + j][column] = 1
                V1[j * p1 + k][column] = 1
                W1[k * n + i][column] = 1
                column += 1
                if time.time() > deadline:
                    break
    
    # Second block
    U2 = [[0] * rank_block for _ in range(n * m)]
    V2 = [[0] * rank_block for _ in range(m * p2)]
    W2 = [[0] * rank_block for _ in range(p2 * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p2):
                U2[i * m + j][column] = 1
                V2[j * p2 + k][column] = 1
                W2[k * n + i][column] = 1
                column += 1
                if time.time() > deadline:
                    break
    
    # Attempt to reduce rank by finding linear dependencies
    # Convert to numpy for efficient computation
    U1_np = np.array(U1, dtype=float)
    V1_np = np.array(V1, dtype=float)
    W1_np = np.array(W1, dtype=float)
    
    U2_np = np.array(U2, dtype=float)
    V2_np = np.array(V2, dtype=float)
    W2_np = np.array(W2, dtype=float)
    
    # Try to find rank-reducing transformations
    # For each block, attempt to find if we can reduce rank by sharing structure
    
    # Strategy: Try to find a common factorization pattern
    # If both blocks have the same structure, we might be able to share some rank
    
    # First, check if we can reduce rank of individual blocks
    # Use SVD to find approximate low-rank decompositions
    
    best_rank = rank_block * 2  # Start with schoolbook total rank
    best_U = U1_np
    best_V = np.hstack([V1_np, V2_np])
    best_W = np.vstack([W1_np, W2_np])
    
    # Try various rank reductions
    for target_rank in range(rank_block, 0, -1):
        if time.time() > deadline:
            break
        
        # Try to decompose each block with target_rank
        try:
            # For block 1
            U1_reduced, V1_reduced, W1_reduced = _try_reduce_rank(
                U1_np, V1_np, W1_np, target_rank, deadline
            )
            
            if U1_reduced is not None:
                # For block 2
                U2_reduced, V2_reduced, W2_reduced = _try_reduce_rank(
                    U2_np, V2_np, W2_np, target_rank, deadline
                )
                
                if U2_reduced is not None:
                    # Combine
                    total_rank = target_rank * 2
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_U = np.hstack([U1_reduced, U2_reduced])
                        best_V = np.hstack([V1_reduced, V2_reduced])
                        best_W = np.vstack([W1_reduced, W2_reduced])
                        
                        # Check if we can do better by sharing
                        # Try to find a joint decomposition
                        if time.time() < deadline - 30:
                            joint_result = _try_joint_decomposition(
                                U1_reduced, V1_reduced, W1_reduced,
                                U2_reduced, V2_reduced, W2_reduced,
                                target_rank, deadline
                            )
                            if joint_result is not None:
                                best_rank = joint_result[0]
                                best_U = joint_result[1]
                                best_V = joint_result[2]
                                best_W = joint_result[3]
        except Exception:
            continue
    
    # Convert back to list format
    U = best_U.tolist()
    V = best_V.tolist()
    W = best_W.tolist()
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, target_rank, deadline):
    """
    Try to reduce the rank of a tensor decomposition.
    Returns (U_reduced, V_reduced, W_reduced) or (None, None, None) if not possible.
    """
    n, m, p = DIMENSIONS
    p1 = 4  # Block size
    
    # Check if target_rank is feasible
    if target_rank < n * m or target_rank < m * p1 or target_rank < p1 * n:
        return None, None, None
    
    # Use randomized SVD-like approach to find low-rank approximation
    # This is a heuristic approach
    
    # For now, return None to indicate we can't easily reduce rank
    # A more sophisticated approach would use algebraic methods
    return None, None, None

def _try_joint_decomposition(U1, V1, W1, U2, V2, W2, target_rank, deadline):
    """
    Try to find a joint decomposition that shares rank between blocks.
    Returns (rank, U, V, W) or None if not possible.
    """
    # This is a placeholder for a more sophisticated joint decomposition
    # In practice, this would involve finding linear dependencies between
    # the two block decompositions
    
    # For now, return None to indicate we can't easily find a joint decomposition
    return None
# EVOLVE-BLOCK-END

