import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Initialization: Spectral Embedding ---
    # The QAP is related to the "Quadratic Assignment Problem" which can be approximated
    # by aligning the eigenvectors of A and B.
    
    # Center the matrices to remove mean effects
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    
    # Get eigenvectors
    # For symmetric matrices, use eigh. If not, use svd or eig.
    # QAPLIB instances are typically symmetric.
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
    except np.linalg.LinAlgError:
        eigvals_a, eigvecs_a = np.linalg.eig(a_centered)
        eigvals_b, eigvecs_b = np.linalg.eig(b_centered)
        # Sort by real part of eigenvalues
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
    else:
        # eigh returns eigenvalues in ascending order. We want largest magnitude.
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        eigvecs_a = eigvecs_a[:, idx_a]
        eigvecs_b = eigvecs_b[:, idx_b]
        
    # Use the first few eigenvectors to create a low-dimensional embedding
    # Usually the first 1-3 eigenvectors capture the most structure.
    # We project to 1D to get an ordering.
    # A common heuristic is to use the first principal component (FPC) or a combination.
    
    # Let's use the first eigenvector for both
    # If the first eigenvalue is negative, we might need to flip the sign to ensure
    # that "similar" items are close. However, for sorting, the relative order matters.
    
    # To handle the sign ambiguity and potential negative eigenvalues, 
    # we can use the absolute value of the projection or just sort by the coordinate.
    
    coord_a = np.real(eigvecs_a[:, 0])
    coord_b = np.real(eigvecs_b[:, 0])
    
    # If the dominant eigenvalue is negative, the "center" is actually the "periphery" in some sense?
    # Actually, for MDS, we usually take the eigenvalues with largest magnitude.
    # Let's try a more robust initialization: Rank Matching based on sums, 
    # but refined by spectral info if available.
    
    # Simple Rank Matching is often surprisingly competitive.
    # Let's combine: Sort facilities by a_sums, Sort locations by b_sums.
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by flow strength (descending)
    heavy_facilities = np.argsort(-a_sums, kind='stable')
    # Sort locations by distance centrality (ascending) - "central" locations have low sum of distances
    central_locations = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute some values for delta calculation?
    # The delta formula:
    # Delta(i, j) = (A_ii - A_jj)(B_YY - B_XX) + (A_ij - A_ji)(B_YX - B_XY) + Sum_{k!=i,j} (A_ik - A_jk)(B_Y, Pk - B_X, Pk)
    # Let X = p[i], Y = p[j].
    # Sum_{k} (A_ik - A_jk)(B_Y, Pk - B_X, Pk)
    # = Sum_k A_ik B_Y, Pk - Sum_k A_ik B_X, Pk - Sum_k A_jk B_Y, Pk + Sum_k A_jk B_X, Pk
    
    # Let V_Ai = A[i, :]
    # Let V_Aj = A[j, :]
    # Let V_BX = B[X, :]
    # Let V_BY = B[Y, :]
    # Note: Pk is the permutation. So B_Y, Pk is B[Y, p[k]].
    
    # We can precompute for each location l, the vector B[l, p].
    # But p changes.
    
    # Efficient Delta Calculation:
    # Let S_Ai = A[i, :] @ B[p[i], p]  -- This is not quite right because the index is p[k].
    # Actually, let's just compute the sum term using vectorization in the loop.
    # n=150, so O(n) per swap is 150 ops. 150*150/2 swaps = 11250 swaps per pass.
    # 11250 * 150 = ~1.7M ops per pass. Very fast in Python/NumPy.
    
    def get_delta(i, j, p, a, b):
        x = p[i]
        y = p[j]
        
        # Term 1: (A_ii - A_jj)(B_yy - B_xx)
        term1 = (a[i, i] - a[j, j]) * (b[y, y] - b[x, x])
        
        # Term 2: (A_ij - A_ji)(B_yx - B_xy)
        term2 = (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
        
        # Term 3: Sum_{k != i, j} (A_ik - A_jk)(B_y, p[k] - B_x, p[k])
        # Let diff_a = a[i, :] - a[j, :]
        # Let diff_b = b[y, p] - b[x, p]
        # Sum = dot(diff_a, diff_b)
        # But we must exclude k=i and k=j.
        
        diff_a = a[i, :] - a[j, :]
        diff_b = b[y, p] - b[x, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract k=i term: (a[i, i] - a[j, i]) * (b[y, p[i]] - b[x, p[i]])
        # p[i] is x. So b[y, x] - b[x, x].
        term_ik = (a[i, i] - a[j, i]) * (b[y, x] - b[x, x])
        
        # Subtract k=j term: (a[i, j] - a[j, j]) * (b[y, p[j]] - b[x, p[j]])
        # p[j] is y. So b[y, y] - b[x, y].
        term_jk = (a[i, j] - a[j, j]) * (b[y, y] - b[x, y])
        
        term3 = total_dot - term_ik - term_jk
        
        return term1 + term2 + term3

    while time.time() < deadline:
        improved = False
        # 2-Opt Local Search
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                delta = get_delta(i, j, p, a, b)
                
                if delta < 0:
                    # Apply swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
                    # Restart inner loop? Or continue?
                    # Restarting ensures we don't miss better moves in the same row.
                    # But it's slower. "First Fit" is standard.
                    break
        
        if not improved:
            # Local Minimum Reached. Perturb.
            # Random Reversal Perturbation
            if time.time() < deadline - 10:
                # Pick a random segment to reverse
                len_seg = np.random.randint(3, 15) # Reverse 3 to 14 elements
                if len_seg < n:
                    start = np.random.randint(0, n - len_seg)
                    end = start + len_seg
                    p[start:end+1] = p[start:end+1][::-1]
                    current_cost = compute_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                else:
                    # If segment too long, just do random swaps
                    for _ in range(10):
                        i = np.random.randint(n)
                        j = np.random.randint(n)
                        if i != j:
                            p[i], p[j] = p[j], p[i]
                    current_cost = compute_cost(p)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()

    return best_p.tolist()
