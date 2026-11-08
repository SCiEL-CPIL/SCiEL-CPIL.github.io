"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

D = 11
DEADLINE = 145.0  # safety margin before 150s kill


def _validate(vectors):
    """Check min pairwise dist >= max norm."""
    if len(vectors) == 0:
        return True
    norms2 = np.sum(vectors * vectors, axis=1)
    max_norm2 = norms2.max()
    # min pairwise squared distance
    # For efficiency, check incrementally in the main loop
    return True


def _min_pairwise_dist2(vectors):
    """Compute minimum squared pairwise distance."""
    n = len(vectors)
    if n < 2:
        return float('inf')
    # Use vectorized computation in chunks to avoid memory issues
    min_d2 = float('inf')
    chunk = 500
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        block = vectors[i:end]
        # distances from block to all vectors
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        block_norms2 = np.sum(block * block, axis=1)[:, None]
        all_norms2 = np.sum(vectors * vectors, axis=1)[None, :]
        dots = block @ vectors.T
        d2 = block_norms2 + all_norms2 - 2 * dots
        # exclude self-distances
        for idx in range(len(block)):
            d2[idx, i + idx] = float('inf')
        min_d2 = min(min_d2, d2.min())
    return min_d2


def _try_add_vectors(base_vectors, max_time=10.0):
    """Try to add more vectors to base set via greedy random search."""
    vectors = list(base_vectors)
    start_time = time.time()
    
    # Precompute norms
    arr = np.array(vectors, dtype=np.int64)
    norms2 = np.sum(arr * arr, axis=1)
    max_norm2 = norms2.max()
    
    added = 0
    attempts = 0
    
    while time.time() - start_time < max_time:
        attempts += 1
        if attempts > 200000:
            break
        
        # Generate random candidate: sparse integer vector
        # Try different sparsity levels
        sparsity = random.choice([2, 3, 3, 4, 4, 5])
        if sparsity >= D:
            sparsity = D - 1
        
        indices = random.sample(range(D), sparsity)
        signs = [random.choice([1, -1]) for _ in range(sparsity)]
        # Values: mostly 1, sometimes 2
        values = [random.choice([1, 1, 1, 2]) for _ in range(sparsity)]
        
        v = [0] * D
        for idx, s, val in zip(indices, signs, values):
            v[idx] = s * val
        
        v_arr = np.array(v, dtype=np.int64)
        v_norm2 = int(np.sum(v_arr * v_arr))
        
        if v_norm2 == 0:
            continue
        
        # Check constraint: for all existing vectors a: ||v - a||^2 >= max(v_norm2, a_norm2)
        # i.e., v_norm2 + a_norm2 - 2*v.a >= max(v_norm2, a_norm2)
        # If v_norm2 >= a_norm2: need a_norm2 - 2*v.a >= 0, i.e., v.a <= a_norm2/2
        # If v_norm2 < a_norm2: need v_norm2 - 2*v.a >= 0, i.e., v.a <= v_norm2/2
        
        dots = arr @ v_arr  # shape (n,)
        # constraint: dots[i] <= min(v_norm2, norms2[i]) / 2
        threshold = np.minimum(v_norm2, norms2) // 2
        # Actually need: v_norm2 + norms2[i] - 2*dots[i] >= max(v_norm2, norms2[i])
        # => 2*dots[i] <= min(v_norm2, norms2[i])
        # => dots[i] <= min(v_norm2, norms2[i]) // 2  (integer)
        
        if np.all(dots <= threshold):
            vectors.append(v)
            arr = np.vstack([arr, v_arr])
            norms2 = np.append(norms2, v_norm2)
            max_norm2 = max(max_norm2, v_norm2)
            added += 1
    
    return vectors, added


def _local_search(vectors, max_time=5.0):
    """Try to swap out some vectors for better ones to allow more additions."""
    start_time = time.time()
    arr = np.array(vectors, dtype=np.int64)
    norms2 = np.sum(arr * arr, axis=1)
    max_norm2 = norms2.max()
    n = len(arr)
    
    # Try removing one vector and adding two
    improved = False
    while time.time() - start_time < max_time:
        # Pick a random vector to remove
        idx = random.randint(0, n - 1)
        remaining = np.delete(arr, idx, axis=0)
        rem_norms2 = np.delete(norms2, idx)
        
        # Try to add two new vectors
        new_vecs = []
        for _ in range(2):
            sparsity = random.choice([2, 3, 3, 4])
            indices = random.sample(range(D), min(sparsity, D - 1))
            signs = [random.choice([1, -1]) for _ in range(len(indices))]
            values = [random.choice([1, 1, 1, 2]) for _ in range(len(indices))]
            v = [0] * D
            for i, s, val in zip(indices, signs, values):
                v[i] = s * val
            v_arr = np.array(v, dtype=np.int64)
            v_norm2 = int(np.sum(v_arr * v_arr))
            if v_norm2 == 0:
                continue
            
            # Check against remaining + new_vecs
            check_arr = np.vstack([remaining] + [np.array(x) for x in new_vecs]) if new_vecs else remaining
            check_norms2 = np.append(rem_norms2, [int(np.sum(np.array(x)**2)) for x in new_vecs]) if new_vecs else rem_norms2
            
            dots = check_arr @ v_arr
            threshold = np.minimum(v_norm2, check_norms2) // 2
            if np.all(dots <= threshold):
                new_vecs.append(v)
        
        if len(new_vecs) == 2:
            # Verify full constraint
            new_arr = np.vstack([remaining] + [np.array(x) for x in new_vecs])
            new_norms2 = np.sum(new_arr * new_arr, axis=1)
            min_d2 = _min_pairwise_dist2(new_arr)
            if min_d2 >= new_norms2.max():
                arr = new_arr
                norms2 = new_norms2
                n = len(arr)
                improved = True
    
    return arr.tolist(), improved


def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    # Start with seed: all ±e_i ± e_j
    base_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    base_vectors.append(v)
    
    # Also add single basis vectors ±e_i (norm 1, should be compatible)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            base_vectors.append(v)
    
    vectors = base_vectors
    n_base = len(vectors)
    
    # Phase 1: Greedy addition of random vectors
    vectors, added1 = _try_add_vectors(vectors, max_time=60.0)
    
    if time.time() - start_time > DEADLINE - 20:
        return np.array(vectors, dtype=np.int64)
    
    # Phase 2: More aggressive random search with different distributions
    vectors, added2 = _try_add_vectors(vectors, max_time=40.0)
    
    if time.time() - start_time > DEADLINE - 10:
        return np.array(vectors, dtype=np.int64)
    
    # Phase 3: Local search to improve
    vectors, improved = _local_search(vectors, max_time=15.0)
    
    if time.time() - start_time > DEADLINE - 5:
        return np.array(vectors, dtype=np.int64)
    
    # Final validation
    arr = np.array(vectors, dtype=np.int64)
    # Remove zero vectors
    norms2 = np.sum(arr * arr, axis=1)
    nonzero_mask = norms2 > 0
    arr = arr[nonzero_mask]
    
    # Validate: min pairwise dist >= max norm
    if len(arr) >= 2:
        min_d2 = _min_pairwise_dist2(arr)
        max_n2 = norms2[nonzero_mask].max()
        if min_d2 < max_n2:
            # Invalid - fall back to base
            arr = np.array(base_vectors, dtype=np.int64)
    
    return arr
# EVOLVE-BLOCK-END
