import numpy as np
import time
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Convert to float64 for numerical stability and speed in eigendecomposition
    # The final cost calculation should use integer arithmetic if possible, 
    # but for delta calculations, float64 is sufficient given the magnitudes.
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # 1. Spectral Initialization
    # For QAP, a common heuristic initialization is based on the principal eigenvectors.
    # We want to align the "heavy" facilities with "central" locations.
    # The leading eigenvector of A captures the main flow structure.
    # The leading eigenvector of B captures the main distance structure.
    
    # Use sparse eigensolver for speed on large n
    # A and B are dense, but we can treat them as sparse if we want, 
    # however eigsh on dense matrices converted to sparse is often fine or 
    # we can just use np.linalg.eigh for n=150 which is very fast.
    
    # np.linalg.eigh is O(n^3) ~ 3.4M ops for n=150. Very fast.
    try:
        # Get the eigenvector corresponding to the largest eigenvalue
        # Note: eigsh returns eigenvalues in ascending order by default if k is not specified?
        # Actually, eigsh(k=1) returns the smallest magnitude. 
        # We want the largest magnitude.
        # For QAP, we usually look at the leading eigenvector (largest eigenvalue).
        
        # Using scipy.sparse.linalg.eigsh with which='LM' (Largest Magnitude)
        A_sp = sp.csr_matrix(a_f)
        B_sp = sp.csr_matrix(b_f)
        
        # n=150 is small enough that dense eigh might be faster and more robust
        # Let's stick to dense eigh for simplicity and reliability at this size
        eigvals_A, eigvecs_A = np.linalg.eigh(a_f)
        eigvals_B, eigvecs_B = np.linalg.eigh(b_f)
        
        # Largest eigenvalue is the last one in ascending order
        v_A = eigvecs_A[:, -1]
        v_B = eigvecs_B[:, -1]
        
        # Sort facilities by their component in v_A (descending)
        # Sort locations by their component in v_B (ascending? or descending?)
        # The sign of the eigenvector is arbitrary. 
        # We need to ensure the assignment is consistent.
        # If we sort both in the same order (e.g., descending), we might get a "mirror" or correct alignment.
        # Let's try sorting both descending.
        
        idx_A = np.argsort(-v_A)
        idx_B = np.argsort(-v_B)
        
        p = np.empty(n, dtype=np.int64)
        p[idx_A] = idx_B
        
    except Exception:
        # Fallback to rank matching if spectral fails
        a_row_sums = a_f.sum(axis=1)
        b_row_sums = b_f.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central

    best_p = p.copy()
    best_cost = _calc_cost(a_f, b_f, best_p)
    
    # 2. Local Search with Vectorized Delta
    # We will implement a First-Improvement 2-Opt local search.
    # To speed it up, we precompute the "interaction" terms.
    
    # Let's define the cost C(p) = sum_{i,j} A[i,j] B[p[i], p[j]]
    # Delta for swapping i and j:
    # D = sum_{k != i,j} A[i,k](B[pj, pk] - B[pi, pk]) + A[k,i](B[pk, pj] - B[pk, pi])
    #   + A[j,k](B[pi, pk] - B[pj, pk]) + A[k,j](B[pk, pi] - B[pk, pj])
    #   + Corner terms for i,j
    
    # We can precompute for each facility f and location l:
    # R_f[l] = sum_k A[f,k] B[l, p[k]]  (Row interaction)
    # C_f[l] = sum_k A[k,f] B[p[k], l]  (Col interaction)
    
    # If we swap i and j:
    # Old cost contribution involving i: R_i[pi] + C_i[pi] - A[i,i]B[pi,pi] (double count correction? No, R and C include k=i term)
    # Let's stick to the explicit delta formula which is safer.
    
    # Vectorized approach for finding the best swap:
    # For a fixed i, we want to find j that minimizes Delta(i, j).
    # Delta(i, j) depends on p[i] and p[j].
    
    # We will iterate i from 0 to n-1.
    # For each i, we compute the delta for swapping i with all j > i.
    
    # Precomputation for fast delta:
    # Let B_p = B[p, :] and B_pT = B[:, p]
    # Actually, let's just compute the delta matrix for a given i.
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Precompute current B rows/cols for current permutation?
        # B[p, :] is the matrix of distances from assigned locations to all locations?
        # No, B is fixed. p is the permutation.
        # B[p[i], p[j]] is the distance between location of i and location of j.
        
        # Let's use the property that Delta(i, j) can be computed in O(n) for a specific pair.
        # To vectorize over j, we need to express Delta(i, j) as a function of p[j].
        
        # Delta(i, j) = 
        # Sum_{k != i,j} A[i,k] (B[pj, pk] - B[pi, pk]) 
        # + Sum_{k != i,j} A[k,i] (B[pk, pj] - B[pk, pi])
        # + Sum_{k != i,j} A[j,k] (B[pi, pk] - B[pj, pk])
        # + Sum_{k != i,j} A[k,j] (B[pk, pi] - B[pk, pj])
        # + Corners
        
        # Let's define vectors for fixed i:
        # u_k = A[i,k] - A[k,i]  (Not quite, terms are separated)
        
        # Term 1: A[i,k] B[pj, pk] - A[i,k] B[pi, pk]
        # Term 2: A[k,i] B[pk, pj] - A[k,i] B[pk, pi]
        # Term 3: -A[j,k] B[pi, pk] + A[j,k] B[pj, pk]
        # Term 4: -A[k,j] B[pk, pi] + A[k,j] B[pk, pj]
        
        # This is complex to fully vectorize over all pairs (i,j) at once.
        # Instead, we loop i, and vectorize over j.
        
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            
            pi = p[i]
            # Precompute vectors related to i
            # A_i = A[i, :]
            # A_col_i = A[:, i]
            
            # We need to evaluate Delta(i, j) for all j != i.
            # Let's compute the "Base" part that doesn't depend on j, and the "J-dependent" part.
            
            # Actually, a simpler and robust vectorized local search step:
            # Compute the full cost matrix of swaps? No, too memory heavy.
            
            # Let's try a "Best Improvement" strategy with vectorized delta for each i.
            
            # For fixed i, let's compute delta for swapping i with every j.
            # We can do this by constructing the delta vector for all j at once.
            
            # Let P be the current permutation.
            # B_pi_P = B[pi, P]  (Row of B corresponding to location pi, indexed by P)
            # B_P_pi = B[P, pi]  (Col of B corresponding to location pi, indexed by P)
            
            B_pi_P = b_f[pi, p]
            B_P_pi = b_f[p, pi]
            
            A_i = a_f[i, :]
            A_col_i = a_f[:, i]
            
            # For each candidate j, let pj = p[j].
            # Delta(i, j) involves B[pj, P] and B[P, pj].
            
            # Let's loop j in chunks or one by one? 
            # One by one is O(n^3) total. n=150 -> 3.4M ops per full pass. 
            # 3.4M * 150 (passes) = 500M ops. Python loop overhead is the killer.
            
            # Vectorized J-loop:
            # We want Delta(i, j) for all j.
            # Delta(i, j) = Sum_k [ A[i,k](B[pj, pk] - B[pi, pk]) + A[k,i](B[pk, pj] - B[pk, pi]) 
            #                    + A[j,k](B[pi, pk] - B[pj, pk]) + A[k,j](B[pk, pi] - B[pk, pj]) ] - Corrections
            
            # This is hard to vectorize cleanly because A[j,k] and A[k,j] change with j.
            
            # Alternative: Use the "Interaction" precomputation.
            # Let's precompute for each facility f:
            # R_f[l] = Sum_k A[f,k] B[l, p[k]]
            # C_f[l] = Sum_k A[k,f] B[p[k], l]
            
            # Cost = Sum_i (R_i[p[i]] + C_i[p[i]]) / 2 + Diagonal corrections?
            # Actually Cost = Sum_{i,j} A[i,j] B[p[i], p[j]]
            # = Sum_i Sum_j A[i,j] B[p[i], p[j]]
            
            # If we swap i and j:
            # New Cost = Old Cost - Cost_terms_involving_i_or_j + New_Cost_terms_involving_i_or_j
            
            # Let's stick to the explicit delta but optimize the inner loop.
            # Since n=150 is small, a pure Python loop with NumPy vectorization for the K-sum is acceptable if we don't do it too many times.
            
            # Optimization: 
            # We will perform a "Sweep".
            # For each i, we find the best j to swap with.
            # We compute the delta for all j > i in a vectorized way?
            # The term A[j,k] makes it hard to vectorize over j directly without storing A rows.
            
            # Let's try this:
            # For fixed i, compute the vector Delta_j for all j.
            # Delta_j = Sum_{k != i,j} ...
            
            # Let's approximate or use a simpler heuristic: 
            # Calculate the "gain" of moving facility i to any location l.
            # Gain(i, l) = Sum_k A[i,k] B[l, p[k]] + Sum_k A[k,i] B[p[k], l]
            #             - (A[i,i] B[l,l] + ... corrections)
            
            # If we just move i to l (shifting others), it's a 3-opt.
            # 2-opt swap is better.
            
            # Let's use a standard 2-opt loop but with NumPy for the K-sum.
            # To avoid O(n^3) Python overhead, we will batch the K-sums.
            
            # Actually, for n=150, O(n^3) = 3.4M operations.
            # If we do this in a Python loop, it's ~1ms per pass? No, NumPy overhead is high.
            # 150 * 150 pairs = 11250 pairs.
            # Each pair check involves a vector operation of size 150.
            # 11250 * 150 = 1.7M flops.
            # This is very fast in NumPy. The overhead is the Python loop over 11250 pairs.
            # 11250 iterations is fine.
            
            for j in range(i + 1, n):
                pj = p[j]
                
                # Vectorized Delta Calculation
                # B_pj_P = b_f[pj, p]
                # B_P_pj = b_f[p, pj]
                # B_pi_P, B_P_pi already computed
                
                # We need A[j, :] and A[:, j]
                A_j = a_f[j, :]
                A_col_j = a_f[:, j]
                
                # Differences
                diff_row = b_f[pj, p] - B_pi_P
                diff_col = b_f[p, pj] - B_P_pi
                
                # Sum over k
                # Terms:
                # A_i * diff_row  (for A[i,k])
                # A_col_i * diff_col (for A[k,i])
                # -A_j * diff_row (for A[j,k]) -- Note: B[pi, pk] - B[pj, pk] = -diff_row
                # -A_col_j * diff_col (for A[k,j]) -- Note: B[pk, pi] - B[pk, pj] = -diff_col
                
                delta_vec = (A_i * diff_row) + (A_col_i * diff_col) - (A_j * diff_row) - (A_col_j * diff_col)
                
                # Sum over all k
                delta_sum = np.sum(delta_vec)
                
                # Subtract k=i and k=j terms
                delta_sum -= delta_vec[i]
                delta_sum -= delta_vec[j]
                
                # Add corner terms
                # Old: A[i,i]B[pi,pi] + A[i,j]B[pi,pj] + A[j,i]B[pj,pi] + A[j,j]B[pj,pj]
                # New: A[i,i]B[pj,pj] + A[i,j]B[pj,pi] + A[j,i]B[pi,pj] + A[j,j]B[pi,pi]
                
                delta_corners = (
                    a_f[i, i] * (b_f[pj, pj] - b_f[pi, pi]) +
                    a_f[i, j] * (b_f[pj, pi] - b_f[pi, pj]) +
                    a_f[j, i] * (b_f[pi, pj] - b_f[pj, pi]) +
                    a_f[j, j] * (b_f[pi, pi] - b_f[pj, pj])
                )
                
                total_delta = delta_sum + delta_corners
                
                if total_delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    improved = True
                    # Update precomputed vectors for i?
                    # pi is now old pj.
                    # B_pi_P and B_P_pi are stale.
                    # We break inner loop and restart outer loop?
                    # Or update?
                    # Updating is O(n). Breaking and restarting is O(n^2) for the rest of the sweep.
                    # Let's break inner and continue outer with updated pi.
                    pi = p[i]
                    B_pi_P = b_f[pi, p]
                    B_P_pi = b_f[p, pi]
                    # We don't break outer, we just update pi and continue j loop?
                    # No, the vector delta_vec depends on pi. If pi changes, all previous j calculations were for the old pi.
                    # But we only accepted ONE swap.
                    # If we continue the j loop, we are checking swaps of the NEW i with subsequent j's.
                    # This is valid.
                    # However, the vectors A_i, A_col_i are fixed for facility i.
                    # B_pi_P and B_P_pi are updated.
                    # So we can continue the inner loop.
                    
                    # Wait, if we swap i and j, the position of j changes.
                    # The next j is j+1.
                    # The calculation for the next j uses the current p.
                    # So it is correct to continue.
                    
        if not improved:
            # Local optimum reached
            break
            
        # Update best
        current_cost = _calc_cost(a_f, b_f, p)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Perturbation for ILS (if we have time)
        # If we reached a local optimum, we might be stuck.
        # Let's add a simple perturbation: swap 2-3 random pairs.
        if time.time() < deadline - 10.0:
            # Perturb
            for _ in range(3):
                i1 = np.random.randint(0, n)
                i2 = np.random.randint(0, n)
                if i1 != i2:
                    p[i1], p[i2] = p[i2], p[i1]
                    
    return best_p.tolist()

def _calc_cost(a, b, p):
    # a, b are float64 arrays
    # p is int64 array
    # Cost = sum a[i,j] b[p[i], p[j]]
    # Vectorized:
    # B_p = b[p, p]
    # Cost = np.sum(a * B_p)
    B_p = b[p, p]
    return np.sum(a * B_p)
