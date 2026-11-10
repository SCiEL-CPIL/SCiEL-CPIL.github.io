import time
import itertools
import numpy as np

_DEADLINE = 145.0
_MARGIN = 5.0


def _tetracode(q, n):
    # Direct sum of the ternary tetracode (9 words, length 4, perfect radius-1)
    # with every word on the remaining n-4 coordinates: 9 * q**(n-4) codewords.
    if n < 4:
        # Fallback for small n, though task is n=8
        return [list(p) for p in itertools.product(range(q), repeat=n)]
        
    base = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    rest = n - 4
    out = []
    for h in base:
        for tail in itertools.product(range(q), repeat=rest):
            out.append(h + list(tail))
    return out


def _build_coverage_matrix(q, n, code):
    """
    Builds a matrix M of shape (q**n, len(code)).
    M[w, k] = 1 if word w is within Hamming distance 1 of code[k].
    """
    total = q ** n
    k = len(code)
    M = np.zeros((total, k), dtype=np.uint8)
    
    # Precompute base for index conversion
    base = np.array([q ** i for i in range(n)], dtype=np.int64)
    
    # Convert code to numpy array for faster access? 
    # code is list of lists.
    code_arr = np.array(code, dtype=np.int64)
    
    # For each word w, find which codewords cover it.
    # Instead of iterating w, iterate codewords and their neighbors?
    # Total operations: k * (1 + n*(q-1))
    # 729 * (1 + 8*2) = 729 * 17 = 12393 operations. Very fast.
    
    for c_idx in range(k):
        c = code_arr[c_idx]
        # Mark c itself
        idx = int(np.dot(c, base))
        M[idx, c_idx] = 1
        
        # Mark neighbors
        for pos in range(n):
            orig_val = c[pos]
            for v in range(q):
                if v == orig_val:
                    continue
                # Create neighbor
                # We can just compute the index difference
                # diff = (v - orig_val) * base[pos]
                # new_idx = idx + diff
                diff = (v - orig_val) * base[pos]
                new_idx = idx + diff
                M[new_idx, c_idx] = 1
                
    return M


def _remove_redundant(M):
    """
    Reverse greedy set cover.
    M is (total_words, num_codewords).
    Returns list of indices of codewords to keep.
    """
    total_words, num_codewords = M.shape
    covered = M.sum(axis=1)  # (total_words,)
    active = np.ones(num_codewords, dtype=bool)
    
    # Optimization: If a row is covered by 0, we have a problem (should not happen if code is valid).
    # If a row is covered by 1, the column covering it MUST be kept.
    
    # We iterate until no more columns can be removed.
    # To speed up, we can process in batches.
    
    # Initial check: if any row has 0 coverage, return all (or handle error).
    if np.any(covered == 0):
         return list(range(num_codewords))
         
    while True:
        # Identify rows that are uniquely covered (covered == 1)
        unique_mask = (covered == 1)
        if not np.any(unique_mask):
            break
            
        # For each active column, check if it covers any unique row.
        # M_unique = M[unique_mask, :]  # (num_unique, num_codewords)
        # We only care about active columns.
        
        # Efficient check:
        # A column j is redundant if it does not intersect the set of unique rows.
        # i.e., M[unique_mask, j].sum() == 0
        
        # Let's compute the sum of M for unique rows for all columns.
        # col_sums = M[unique_mask, :].sum(axis=0)
        # This is (num_unique) * (num_codewords) additions.
        # num_unique can be large.
        
        # Alternative:
        # Iterate through unique rows? No, vectorized is better.
        
        # Get indices of unique rows
        unique_indices = np.nonzero(unique_mask)[0]
        
        # Compute which columns cover these rows
        # sub_M = M[unique_indices, :]
        # col_has_unique = sub_M.any(axis=0)
        
        # To save memory/time, we can do:
        # col_has_unique = np.zeros(num_codewords, dtype=bool)
        # for r in unique_indices:
        #    cols = np.nonzero(M[r])[0]
        #    col_has_unique[cols] = True
        # This is O(num_unique * avg_degree).
        
        # Let's try the vectorized way first, it's usually fast enough for small matrices.
        sub_M = M[unique_indices, :]
        col_has_unique = sub_M.any(axis=0)
        
        # A column is redundant if it is active AND does NOT have a unique row.
        # redundant = active & ~col_has_unique
        redundant_mask = active & ~col_has_unique
        
        if not np.any(redundant_mask):
            break
            
        # Remove redundant columns
        redundant_indices = np.nonzero(redundant_mask)[0]
        
        # Update covered
        # covered -= M[:, redundant_indices].sum(axis=1)
        # M[:, redundant_indices] is (total_words, num_redundant)
        # Summing this is efficient.
        covered -= M[:, redundant_indices].sum(axis=1)
        
        # Mark as inactive
        active[redundant_indices] = False
        
    return np.nonzero(active)[0]


def construct(q, n, radius):
    start = time.time()
    
    # Build initial code
    code = _tetracode(q, n)
    
    # Check time
    if time.time() - start > _DEADLINE - _MARGIN:
        return code
        
    total = q ** n
    
    # Build coverage matrix
    # Only feasible if total * len(code) is not too huge.
    # 6561 * 729 ~ 4.8M entries. Fine.
    M = _build_coverage_matrix(q, n, code)
    
    # Check time
    if time.time() - start > _DEADLINE - _MARGIN:
        return code
        
    # Prune
    keep_indices = _remove_redundant(M)
    
    # Construct result
    result = [code[i] for i in keep_indices]
    
    # Check time
    if time.time() - start > _DEADLINE - _MARGIN:
        return result
        
    return result
