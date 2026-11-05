import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Generate all blocks and triples
    all_blocks = list(combinations(range(v), k))
    num_blocks = len(all_blocks)
    all_triples = list(combinations(range(v), t))
    num_triples = len(all_triples)

    # Precompute which triples are covered by which blocks (incidence matrix)
    # M[b, i] = 1 if block b covers triple i
    # Using a sparse-friendly approach but building dense for milp
    # 346104 blocks * 2024 triples. This is large but manageable with uint8.
    # However, milp requires float or int arrays. Let's build the constraint matrix.
    
    # To speed up precomputation, we can use numpy.
    # For each block, we have C(k, t) triples.
    # We can create a large array of indices and use advanced indexing.
    
    # Let's map triples to indices
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Build incidence matrix as a list of sparse rows first, then convert to CSR for milp
    # scipy.sparse is better for large matrices.
    from scipy import sparse
    
    # Precompute the triples for each block to avoid repeated computation
    # This might take some time, but it's necessary.
    block_triples_indices = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in combinations(b, t)]
        block_triples_indices.append(tris)
        
    # Build sparse incidence matrix A (num_triples x num_blocks)
    # Constraint: A @ x >= 1, x in {0, 1}
    # Variables x are blocks.
    # A is the transpose of the matrix we described earlier.
    
    row_indices = []
    col_indices = []
    data = []
    
    for j, tris in enumerate(block_triples_indices):
        for i in tris:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A_sparse = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_triples, num_blocks))
    
    # Objective: minimize sum(x)
    c = np.ones(num_blocks)
    
    # Constraints: A @ x >= 1
    # milp uses A @ x >= lb and A @ x <= ub
    # So we set lb = 1, ub = inf
    constraints = LinearConstraint(A_sparse.toarray(), lb=1, ub=np.inf)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integality: x is integer
    integrality = np.ones(num_blocks)
    
    # Solve using HiGHS
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            selected_blocks = [i for i in range(num_blocks) if x[i] > 0.5]
            
            # Verify and Prune
            # The LP solution should be a valid covering.
            # Let's do a quick prune just in case.
            final_blocks = _prune_covering(selected_blocks, all_blocks, block_triples_indices, triple_to_idx, v, k, t, deadline)
            
            if len(final_blocks) < len(selected_blocks):
                selected_blocks = final_blocks
                
            return [all_blocks[i] for i in selected_blocks]
        else:
            # If LP fails or times out, fallback to greedy
            pass
    except Exception:
        pass
    
    # Fallback: Greedy Heuristic
    return _greedy_covering(v, k, t, all_blocks, block_triples_indices, triple_to_idx, deadline)

def _prune_covering(block_indices, all_blocks, block_triples_indices, triple_to_idx, v, k, t, deadline):
    """
    Remove redundant blocks from a valid covering.
    """
    if not block_indices:
        return []
        
    num_triples = len(triple_to_idx)
    # Count coverage for each triple
    counts = np.zeros(num_triples, dtype=np.int32)
    
    for idx in block_indices:
        for tri_idx in block_triples_indices[idx]:
            counts[tri_idx] += 1
            
    new_indices = []
    for idx in block_indices:
        if time.time() > deadline - 5:
            break
        # Check if all triples in this block are covered by more than 1 block
        is_redundant = True
        for tri_idx in block_triples_indices[idx]:
            if counts[tri_idx] == 1:
                is_redundant = False
                break
        if is_redundant:
            # Remove this block
            for tri_idx in block_triples_indices[idx]:
                counts[tri_idx] -= 1
        else:
            new_indices.append(idx)
            
    # Second pass if time permits
    if time.time() < deadline - 10 and len(new_indices) > 0:
        counts = np.zeros(num_triples, dtype=np.int32)
        for idx in new_indices:
            for tri_idx in block_triples_indices[idx]:
                counts[tri_idx] += 1
        final_indices = []
        for idx in new_indices:
            is_redundant = True
            for tri_idx in block_triples_indices[idx]:
                if counts[tri_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for tri_idx in block_triples_indices[idx]:
                    counts[tri_idx] -= 1
            else:
                final_indices.append(idx)
        return final_indices
        
    return new_indices

def _greedy_covering(v, k, t, all_blocks, block_triples_indices, triple_to_idx, deadline):
    """
    Standard greedy set cover heuristic.
    """
    num_triples = len(triple_to_idx)
    uncovered = np.ones(num_triples, dtype=np.uint8)
    selected_blocks = []
    
    # Precompute gains? No, dynamic.
    # To speed up, we can use the fact that we only care about uncovered triples.
    # However, scanning 346k blocks every step is slow in Python.
    # Let's try to vectorize the gain calculation if possible, or accept the time cost
    # if the number of iterations is small (~70-80).
    
    # 70 iterations * 346k blocks * ~35 triples per block = ~850M operations.
    # This is likely too slow for pure Python loops.
    
    # Alternative: Use the sparse matrix from the LP setup?
    # We can rebuild it or pass it. Let's rebuild a CSR matrix for gain calc.
    from scipy import sparse
    
    row_indices = []
    col_indices = []
    data = []
    for j, tris in enumerate(block_triples_indices):
        for i in tris:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A_sparse = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(num_triples, len(all_blocks)))
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        # Calculate gain for each block: number of newly covered triples
        # gain[j] = sum(uncovered[i] for i in A[:, j])
        # This is (uncovered.T @ A) if A is CSR?
        # Actually, if A is (triples x blocks), then A.T is (blocks x triples).
        # gain = A.T @ uncovered
        A_T = A_sparse.T
        gains = (A_T @ uncovered).toarray().flatten()
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks.append(best_idx)
        
        # Update uncovered
        # uncovered[rows covered by best_idx] = 0
        # Get rows for best_idx from A_sparse
        # In CSR, we need to find the non-zero rows in column best_idx.
        # A_sparse.T is CSC? No, A_sparse.T is CSC if A is CSR.
        # Let's just use the precomputed list of triples for this block.
        for tri_idx in block_triples_indices[best_idx]:
            uncovered[tri_idx] = 0
            
    # Prune
    final_indices = _prune_covering(selected_blocks, all_blocks, block_triples_indices, triple_to_idx, v, k, t, deadline)
    
    return [all_blocks[i] for i in final_indices]
