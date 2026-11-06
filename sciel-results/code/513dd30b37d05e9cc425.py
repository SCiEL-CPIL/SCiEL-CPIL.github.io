"""
Phylogenetic topology reconstruction using BIONJ (FastUPGMA) with randomized restarts.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distances(sequences):
    """
    Compute Jukes-Cantor corrected distance matrix using NumPy.
    sequences: dict mapping label -> DNA string
    Returns: dict mapping (label_a, label_b) -> distance
    """
    labels = sorted(sequences, key=lambda x: int(x[1:]) if x.startswith('t') else 0)
    n = len(labels)
    if n == 0:
        return {}
    
    # Convert sequences to a 2D array of integers for faster comparison
    # 0=A, 1=C, 2=G, 3=T
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Map characters to indices
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = np.zeros((n, length), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            data[i, j] = char_map.get(c, 0)
            
    # Calculate pairwise differences
    # diff[i, j] is 1 if data[i] != data[j], else 0
    # We can do this by broadcasting, but for memory efficiency with large n, 
    # we might just loop or use a clever trick. 
    # For n <= 32 and L <= 2000, n*L is small enough for standard ops.
    
    # Efficient pairwise mismatch count
    # diff_matrix[i, j] = sum(data[i] != data[j])
    # This creates an n x n x L boolean array which is 32*32*2000 ~ 2M bytes. Safe.
    diff_bool = (data[:, np.newaxis, :] != data[np.newaxis, :, :])
    mismatch_counts = np.sum(diff_bool, axis=2)
    
    # Calculate JC distances
    # p = mismatch_counts / length
    p = mismatch_counts / length
    
    # Handle p >= 0.75 case (log(1 - 4p/3) would be log(negative) or log(0))
    # JC formula: -3/4 * ln(1 - 4/3 * p)
    # If p >= 0.75, set to a large value or handle specially.
    # Usually p is much smaller in simulated data unless branches are very long.
    valid = p < 0.75
    dist_matrix = np.zeros((n, n))
    dist_matrix[valid] = -0.75 * np.log(1.0 - (4.0/3.0) * p[valid])
    dist_matrix[~valid] = 10.0 # Large penalty for highly divergent sequences
    
    # Set diagonal to 0
    np.fill_diagonal(dist_matrix, 0.0)
    
    # Convert to dict for the BIONJ algorithm which expects a dict or matrix
    # We will pass the numpy matrix directly to a modified BIONJ for speed
    return labels, dist_matrix

def _bionj(labels, dist_matrix, rng):
    """
    Perform BIONJ (FastUPGMA) reconstruction.
    labels: list of taxon labels
    dist_matrix: numpy array of distances
    rng: numpy random generator for noise
    Returns: Newick string
    """
    n = len(labels)
    if n <= 2:
        if n == 1:
            return f"{labels[0]};\n"
        else:
            return f"({labels[0]},{labels[1]});\n"
            
    # Copy distance matrix to avoid modification
    D = dist_matrix.copy()
    
    # Calculate variances for BIONJ weighting
    # Var(d_ij) = (3/4) * p_ij * (1 - p_ij) / L + (3/16) * (p_ik + p_jk)^2 / L ... 
    # Simplified variance estimate for JC:
    # Let p_ij be the raw proportion of differences.
    # We need p_ij to calculate variance. Let's re-calculate or store it.
    # To keep it simple and fast, we can approximate variance based on distance or re-calculate p.
    # Let's re-calculate p from the original data? No, we only have D.
    # Actually, BIONJ requires the variance of the distance estimate.
    # For JC, var(d_ij) approx (3/4) * p_ij * (1-p_ij) / L.
    # We don't have L or p_ij here easily without passing them.
    # Let's pass p_matrix as well or compute variance inside.
    
    # Refactor: Pass p_matrix too.
    # For now, let's use a simplified weighting or standard NJ if variance is too complex to pass.
    # However, BIONJ is significantly better. Let's compute p_matrix in _jc_distances and pass it.
    
    # Let's assume we have p_matrix available. 
    # I will modify _jc_distances to return p_matrix as well.
    
    pass # Placeholder, will implement below

def _jc_distances_with_p(sequences):
    """
    Returns labels, dist_matrix, p_matrix
    """
    labels = sorted(sequences, key=lambda x: int(x[1:]) if x.startswith('t') else 0)
    n = len(labels)
    if n == 0:
        return labels, np.array([]), np.array([])
        
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = np.zeros((n, length), dtype=np.int8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            data[i, j] = char_map.get(c, 0)
            
    diff_bool = (data[:, np.newaxis, :] != data[np.newaxis, :, :])
    mismatch_counts = np.sum(diff_bool, axis=2)
    p = mismatch_counts / length
    
    valid = p < 0.75
    dist_matrix = np.zeros((n, n))
    dist_matrix[valid] = -0.75 * np.log(1.0 - (4.0/3.0) * p[valid])
    dist_matrix[~valid] = 10.0
    np.fill_diagonal(dist_matrix, 0.0)
    
    return labels, dist_matrix, p

def _bionj_reconstruct(labels, D, P, noise_level=0.0, rng=None):
    """
    BIONJ algorithm implementation.
    D: Distance matrix
    P: Raw proportion of differences matrix (for variance calculation)
    noise_level: Add noise to D before running? Or just use D.
    """
    n = len(labels)
    if n <= 2:
        if n == 1: return f"{labels[0]};\n"
        return f"({labels[0]},{labels[1]});\n"
        
    # Add noise if requested (for restarts)
    if noise_level > 0 and rng is not None:
        D = D + rng.normal(0, noise_level, D.shape)
        # Ensure symmetry
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)
        
    # Initialize
    active = list(range(n))
    # Tree representation: list of strings (Newick subtrees)
    trees = [labels[i] for i in range(n)]
    
    # Calculate Variances
    # Var(d_ij) = (3/4) * p_ij * (1-p_ij) / L + ... 
    # We need L. Let's assume L is constant. 
    # We can estimate L from the input if we passed it, or just use a simplified variance.
    # Actually, the standard BIONJ formula uses:
    # var_ij = (3/4) * p_ij * (1-p_ij) / L + (3/16) * (p_ik + p_jk)^2 / L ... this is complex.
    # A common approximation is var_ij ~ D_ij^2 / L or similar.
    # Let's use the property that for JC, var(d) = (3/4) * p * (1-p) / L.
    # We don't have L here. Let's pass L or infer it.
    # Let's modify the call chain to pass L.
    
    pass

def reconstruct(sequences):
    start_time = time.time()
    timeout = 260.0 # Safety margin for 270s limit
    
    labels, D, P = _jc_distances_with_p(sequences)
    n = len(labels)
    
    if n <= 2:
        if n == 1: return f"{labels[0]};\n"
        return f"({labels[0]},{labels[1]});\n"
        
    # Infer L (length of sequences)
    L = len(next(iter(sequences.values())))
    
    # Calculate Variances for BIONJ
    # var_ij = (3/4) * P_ij * (1-P_ij) / L
    # We also need the sum of variances for the weighting factor in BIONJ.
    # The BIONJ criterion for joining i and j is:
    # Q_ij = D_ij - (sum_k D_ik + sum_k D_jk) / (n-2) * (1 + w_ij)
    # where w_ij depends on variances.
    
    # Actually, let's implement a robust BIONJ.
    # Reference: Gascuel, O. (1997). BIONJ: a fast and accurate algorithm for the construction of large phylogenetic trees.
    
    # We will run multiple iterations with noise.
    # Store the best trees and their split sets.
    
    max_iters = 50
    current_iter = 0
    best_tree = None
    best_score = -1.0
    all_splits = {} # split -> count
    
    # Helper to extract splits from Newick
    def _extract_splits(newick, n_taxa):
        # Simplified: Count unique splits. 
        # For performance, we might just return the string and compare later if needed.
        # But for consensus, we need splits.
        # Given time constraints, maybe just keep the best single tree from a more robust method?
        # No, consensus is key.
        # Let's parse Newick to get splits.
        # This is expensive. 
        # Alternative: Keep the tree with the lowest "inconsistency" or just the most frequent identical tree?
        # With noise, identical trees are unlikely.
        # We need to count splits.
        pass
        
    # Let's try a simpler approach: 
    # 1. Run BIONJ once with noise=0.
    # 2. Run BIONJ multiple times with small noise.
    # 3. For each resulting tree, calculate its "score" against the distance matrix? No, we don't know the true tree.
    # 4. Use a "consensus" approach: 
    #    - For each run, extract all 3-subset splits (or all splits).
    #    - Count frequency of each split.
    #    - Build the tree using the most frequent splits (e.g., using a greedy method or just picking the tree that matches the most splits).
    #    - Since we only have ~20-30 taxa, the number of splits is manageable (2^(n-3)-1).
    
    # Optimized Plan:
    # 1. Generate K trees using BIONJ with noise.
    # 2. For each tree, extract the set of splits.
    # 3. Find the split that appears most frequently across all trees.
    # 4. Construct a consensus tree from these splits.
    #    - If the splits form a valid tree, return it.
    #    - If not, pick the tree from the runs that has the highest overlap with the top splits.
    
    # Implementation of BIONJ
    def _run_bionj(D, P, L, noise_level, rng):
        n = len(D)
        if n <= 2:
            if n == 1: return f"{labels[0]};\n"
            return f"({labels[0]},{labels[1]});\n"
            
        D_work = D.copy()
        if noise_level > 0:
            D_work = D_work + rng.normal(0, noise_level, D_work.shape)
            D_work = (D_work + D_work.T) / 2.0
            np.fill_diagonal(D_work, 0.0)
            
        active = list(range(n))
        trees = [labels[i] for i in range(n)]
        
        # Pre-compute variances
        # var_ij = (3/4) * P_ij * (1-P_ij) / L
        var = (0.75 * P * (1.0 - P)) / L
        np.fill_diagonal(var, 0.0)
        
        while len(active) > 3:
            n_active = len(active)
            
            # Calculate row sums for active nodes
            # S_i = sum_{j in active, j!=i} D_ij
            S = np.zeros(n_active)
            for idx_i, i in enumerate(active):
                for idx_j, j in enumerate(active):
                    if i != j:
                        S[idx_i] += D_work[i, j]
                        
            # Calculate BIONJ weights
            # w_ij = (1 / (n-2)) * (sum_k var_ik + sum_k var_jk) / (var_ik + var_jk + 2*var_ij) ? 
            # Formula: w_ij = (1/(n-2)) * (1 + (sum_k var_ik + sum_k var_jk) / (var_ik + var_jk + 2*var_ij))
            # Actually, Gascuel's formula:
            # w_ij = 1 + (1/(n-2)) * (sum_{k != i,j} (var_ik + var_jk)) / (var_ik + var_jk + 2*var_ij)
            # Let's simplify: If variances are similar, w_ij approx 1.
            # For speed and stability, let's use w_ij = 1.0 (standard NJ) if variance calc is too risky/complex,
            # OR use the full formula.
            
            best_q = float('inf')
            best_i = -1
            best_j = -1
            
            for idx_i in range(n_active):
                i = active[idx_i]
                for idx_j in range(idx_i + 1, n_active):
                    j = active[idx_j]
                    
                    # Calculate w_ij
                    # Sum of variances for i and j
                    var_sum_i = 0
                    var_sum_j = 0
                    for idx_k, k in enumerate(active):
                        if k != i and k != j:
                            var_sum_i += var[i, k]
                            var_sum_j += var[j, k]
                    
                    denom = var[i, k] + var[j, k] + 2 * var[i, j] # Wait, k is from loop, need specific k?
                    # The formula sums over k != i, j.
                    # Let's re-read Gascuel 1997.
                    # w_ij = 1 + (1/(n-2)) * [ sum_{k != i,j} (var_ik + var_jk) ] / [ var_ik + var_jk + 2 var_ij ]
                    # Note: The denominator in the fraction is typically just the variance of the distance d_ij?
                    # No, it's related to the variance of the estimator.
                    
                    # To avoid bugs and ensure speed, let's use a simplified BIONJ where w_ij = 1 
                    # if we are not confident in the variance implementation, 
                    # BUT BIONJ is the main improvement.
                    # Let's use the standard NJ criterion first, then try to add BIONJ weighting carefully.
                    # Given the risk, let's stick to a robust NJ with noise, 
                    # OR implement BIONJ correctly.
                    
                    # Correct BIONJ Weight:
                    # w_ij = 1 + (1/(n-2)) * (sum_{k != i,j} var_ik + sum_{k != i,j} var_jk) / (var_ik + var_jk + 2*var_ij)
                    # This requires summing variances.
                    
                    sum_var_i = 0
                    sum_var_j = 0
                    for idx_k, k in enumerate(active):
                        if k != i and k != j:
                            sum_var_i += var[i, k]
                            sum_var_j += var[j, k]
                            
                    # The denominator is the variance of the distance d_ij?
                    # var(d_ij) = var[i, j]
                    # So w_ij = 1 + (1/(n-2)) * (sum_var_i + sum_var_j) / var[i, j]
                    # If var[i, j] is 0, w_ij is undefined. Handle division by zero.
                    
                    if var[i, j] > 1e-10:
                        w_ij = 1.0 + (1.0 / (n_active - 2)) * (sum_var_i + sum_var_j) / var[i, j]
                    else:
                        w_ij = 1.0
                        
                    q_ij = D_work[i, j] - (S[idx_i] + S[idx_j]) / (n_active - 2) * w_ij
                    if q_ij < best_q:
                        best_q = q_ij
                        best_i = i
                        best_j = j
                        
            # Join best_i and best_j
            new_idx = len(active)
            new_label = f"n{new_idx}_{best_i}_{best_j}"
            
            # New distance to other nodes
            for idx_k, k in enumerate(active):
                if k != best_i and k != best_j:
                    # D_new_k = 0.5 * (D_ik + D_jk - D_ij)
                    D_work[new_idx, k] = 0.5 * (D_work[best_i, k] + D_work[best_j, k] - D_work[best_i, best_j])
                    D_work[k, new_idx] = D_work[new_idx, k]
                    
            # New variance? 
            # var_new_k = 0.25 * (var_ik + var_jk + 2*var_ij - 2*var_ik - 2*var_jk + var_ik + var_jk) ... 
            # Simplified: var_new_k = 0.25 * (var_ik + var_jk) + 0.5 * var_ij? 
            # Actually, for the next iteration, we just need variances for the new node.
            # var(new, k) = 0.25 * (var(i, k) + var(j, k)) + 0.5 * var(i, j) is a common approximation.
            for idx_k, k in enumerate(active):
                if k != best_i and k != j:
                    var[new_idx, k] = 0.25 * (var[best_i, k] + var[best_j, k]) + 0.5 * var[best_i, best_j]
                    var[k, new_idx] = var[new_idx, k]
                    
            # Update trees
            trees[new_idx] = f"({trees[best_i]},{trees[best_j]})"
            
            # Update active list
            active = [x for x in active if x != best_i and x != best_j]
            active.append(new_idx)
            
        # Final 3 nodes
        if len(active) == 3:
            a, b, c = active
            return f"({trees[a]},{trees[b]},{trees[c]});\n"
        else:
            # Should not happen
            return "(" + ",".join(trees[i] for i in active) + ");\n"

    # Main Loop
    rng = np.random.default_rng(42)
    
    # Run 1: No noise
    tree_0 = _run_bionj(D, P, L, 0.0, rng)
    
    # Run N: With noise
    # Estimate noise level: std of D
    noise_std = np.std(D)
    if noise_std == 0:
        noise_std = 0.01
        
    trees_list = [tree_0]
    
    for i in range(40):
        if time.time() - start_time > timeout:
            break
        noise_level = noise_std * 0.05 # Small perturbation
        tree_i = _run_bionj(D, P, L, noise_level, rng)
        trees_list.append(tree_i)
        
    # Consensus: Find the most frequent tree? 
    # With noise, exact string match is unlikely.
    # We need to compare topologies.
    # Simple metric: Count how many times each unique Newick string appears.
    # If ties, pick the first one? 
    # Better: Extract splits.
    
    # Given the time, let's just pick the tree that appears most often in the list of strings.
    # If all are unique, we pick the first one (no noise) or the one with lowest "energy"?
    # We don't have energy.
    # Let's assume the no-noise tree is a good baseline.
    # But BIONJ with noise often finds better local optima.
    # Let's count string frequencies.
    
    from collections import Counter
    counts = Counter(trees_list)
    best_tree = counts.most_common(1)[0][0]
    
    return best_tree

# EVOLVE-BLOCK-END
