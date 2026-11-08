import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n, m, p = 3, 5, 5
    
    # Try multiple Strassen block configurations and pick the one with lowest rank
    # A Strassen block at (i0, j0, k0) covers rows i0,i0+1; cols j0,j0+1; output k0,k0+1
    # It has rank 7 and covers 8 tensor entries
    
    def get_strassen_terms(i0, j0, k0):
        """Return the 7 rank-1 terms for a Strassen block at (i0,j0,k0)"""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'U': [(i0*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)],
            'V': [(j0*p+k0, 1.0), ((j0+1)*p+(k0+1), 1.0)],
            'W': [(k0*n+i0, 1.0), ((k0+1)*n+(i0+1), 1.0)]
        })
        # M2 = (A10+A11)B00
        terms.append({
            'U': [((i0+1)*m+j0, 1.0), ((i0+1)*m+(j0+1), 1.0)],
            'V': [(j0*p+k0, 1.0)],
            'W': [(k0*n+(i0+1), 1.0), ((k0+1)*n+(i0+1), -1.0)]
        })
        # M3 = A00(B01-B11)
        terms.append({
            'U': [(i0*m+j0, 1.0)],
            'V': [(j0*p+(k0+1), 1.0), ((j0+1)*p+(k0+1), -1.0)],
            'W': [((k0+1)*n+i0, 1.0), ((k0+1)*n+(i0+1), 1.0)]
        })
        # M4 = A11(B10-B00)
        terms.append({
            'U': [((i0+1)*m+(j0+1), 1.0)],
            'V': [((j0+1)*p+k0, 1.0), (j0*p+k0, -1.0)],
            'W': [(k0*n+i0, 1.0), (k0*n+(i0+1), 1.0)]
        })
        # M5 = (A00+A01)B11
        terms.append({
            'U': [(i0*m+j0, 1.0), (i0*m+(j0+1), 1.0)],
            'V': [((j0+1)*p+(k0+1), 1.0)],
            'W': [(k0*n+i0, -1.0), ((k0+1)*n+i0, 1.0)]
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'U': [((i0+1)*m+j0, 1.0), (i0*m+j0, -1.0)],
            'V': [(j0*p+k0, 1.0), (j0*p+(k0+1), 1.0)],
            'W': [((k0+1)*n+(i0+1), 1.0)]
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'U': [(i0*m+(j0+1), 1.0), ((i0+1)*m+(j0+1), -1.0)],
            'V': [((j0+1)*p+k0, 1.0), ((j0+1)*p+(k0+1), 1.0)],
            'W': [(k0*n+i0, 1.0)]
        })
        return terms
    
    def get_covered_terms(terms_list):
        """Get the set of (i,j,k) terms covered by the given rank-1 terms"""
        covered = set()
        for term in terms_list:
            # For each rank-1 term, the covered entries are all combinations
            # of the U indices, V indices, and W indices
            u_indices = [idx for idx, _ in term['U']]
            v_indices = [idx for idx, _ in term['V']]
            w_indices = [idx for idx, _ in term['W']]
            for u in u_indices:
                i, j = divmod(u, m)
                for v in v_indices:
                    jj, k = divmod(v, p)
                    if j == jj:  # Only if the column index matches
                        for w in w_indices:
                            kk, ii = divmod(w, n)
                            if k == kk:  # Only if the row index matches
                                covered.add((i, j, k))
        return covered
    
    # Generate all valid Strassen block positions
    valid_blocks = []
    for i0 in range(n-1):  # 0 to 1
        for j0 in range(m-1):  # 0 to 3
            for k0 in range(p-1):  # 0 to 3
                valid_blocks.append((i0, j0, k0))
    
    # Try different numbers of Strassen blocks and find the best configuration
    best_rank = n * m * p  # schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # We'll try configurations with 2, 3, 4, 5 Strassen blocks
    # For each, try different combinations and pick the one that minimizes total rank
    
    # For efficiency, let's try a greedy approach: add Strassen blocks one by one,
    # each time picking the block that covers the most uncovered terms
    
    all_terms = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.add((i, j, k))
    
    def compute_rank_for_config(block_positions):
        """Compute the total rank for a given set of Strassen block positions"""
        if time.time() - start_time > deadline - 5:
            return None, None, None, None
        
        all_terms_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    all_terms_set.add((i, j, k))
        
        covered = set()
        all_rank1_terms = []
        
        for (i0, j0, k0) in block_positions:
            terms = get_strassen_terms(i0, j0, k0)
            all_rank1_terms.extend(terms)
            # Update covered
            term_covered = get_covered_terms(terms)
            covered.update(term_covered)
        
        # Count uncovered terms
        uncovered = all_terms_set - covered
        total_rank = len(all_rank1_terms) + len(uncovered)
        
        # Build U, V, W
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        for term in all_rank1_terms:
            for idx, val in term['U']:
                U[idx][col] += val
            for idx, val in term['V']:
                V[idx][col] += val
            for idx, val in term['W']:
                W[idx][col] += val
            col += 1
        
        # Add schoolbook terms for uncovered
        for (i, j, k) in uncovered:
            U[i*m+j][col] += 1.0
            V[j*p+k][col] += 1.0
            W[k*n+i][col] += 1.0
            col += 1
        
        return total_rank, U, V, W, block_positions
    
    # Greedy: start with no blocks, add blocks one by one
    # Try multiple starting points and greedy orders
    
    best_result = None
    
    for num_blocks in [3, 4, 5, 6]:
        if time.time() - start_time > deadline - 10:
            break
        
        # Try random subsets of blocks of size num_blocks
        import random
        random.seed(42)
        
        # Also try systematic: all blocks, then remove some
        # For now, try random sampling
        
        num_trials = 200
        for trial in range(num_trials):
            if time.time() - start_time > deadline - 5:
                break
            
            # Random selection of blocks
            if len(valid_blocks) <= num_blocks:
                blocks_to_try = valid_blocks[:]
            else:
                blocks_to_try = random.sample(valid_blocks, num_blocks)
            
            rank, U, V, W, _ = compute_rank_for_config(blocks_to_try)
            
            if rank is not None:
                if best_result is None or rank < best_result[0]:
                    best_result = (rank, U, V, W)
    
    # If we found a good result, use it
    if best_result is not None:
        rank, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: original approach
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    total_strassen_rank = 4 * 7
    total_terms = n * m * p
    strassen_terms_count = 4 * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        terms = get_strassen_terms(i0, j0, k0)
        for t, term in enumerate(terms):
            c = col_offset + t
            for idx, val in term['U']:
                U[idx][c] += val
            for idx, val in term['V']:
                V[idx][c] += val
            for idx, val in term['W']:
                W[idx][c] += val
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1.0
                    V[j*p+k][col] += 1.0
                    W[k*n+i][col] += 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
