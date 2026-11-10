"""
Strategy: Smooth Surrogate Optimization via SoftMax.

The previous approach used discrete local search on the exact max-discrepancy objective.
This objective is non-smooth (a max function), leading to a rugged landscape where
gradient information is unavailable and local minima are hard to escape without
randomized restarts or simulated annealing.

This new approach introduces a smooth surrogate objective:
    L = sum_{APs} SoftMax_beta( [s_1^2, (s_1+s_2)^2, ..., (sum s)^2] )
where SoftMax_beta(v) = exp(beta * max(v)) * sum(exp(beta * v_i)) / sum(exp(beta * v_i))
(or similar stable formulation).

By minimizing this smooth function, we can use:
1. Deterministic coordinate descent: For each position i, compute the change in L
   for flipping sign[i]. Since L is smooth, we can approximate the gradient or
   simply evaluate the exact change in L for each flip.
2. The smoothness allows the search to "flow" towards low-discrepancy regions
   without getting trapped by the discrete jumps of the max function.
3. We periodically evaluate the true max-discrepancy to track progress and
   potentially adjust the beta parameter (annealing beta from low to high) to
   transition from a smooth global search to a sharp local refinement.

This is a different design family: it uses a continuous relaxation/surrogate
optimization technique rather than direct combinatorial search on the discrete
metric.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_ap_structure(n):
    """
    Pre-compute all arithmetic progressions and map indices to the progressions they belong to.
    Returns:
        ap_indices: list of numpy arrays of indices for each AP
        index_to_aps: list of lists of (ap_id, pos) for each index
    """
    ap_indices = []
    index_to_aps = [[] for _ in range(n)]
    
    ap_id = 0
    for step in range(1, n + 1):
        for start in range(n):
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) == 0:
                continue
            
            ap_indices.append(indices)
            
            for pos, idx in enumerate(indices):
                index_to_aps[idx].append((ap_id, pos))
            
            ap_id += 1
            
    return ap_indices, index_to_aps

def _soft_max_surrogate(prefix_sums, beta):
    """
    Compute the smooth surrogate objective: sum of SoftMax of squared prefix sums.
    For numerical stability, we use:
    SoftMax_beta(v) = exp(beta * (max_v - m)) * sum(exp(beta * (v_i - m))) / sum(exp(beta * (v_i - m)))
    where m = max_v. Actually, the standard soft max is:
    m = max(v)
    L = m + (1/beta) * log(sum(exp(beta * (v_i - m))))
    But for minimization of max, we want a function that approximates max(v) but is smooth.
    The log-sum-exp trick gives:
    f(v) = (1/beta) * log(sum(exp(beta * v_i)))
    This is a smooth approximation to max(v).
    """
    # v is prefix_sums, which are integers. We square them first.
    v = prefix_sums ** 2
    # To avoid overflow, subtract max
    v_max = np.max(v)
    if v_max == 0:
        return 0.0
    
    # Use stable log-sum-exp
    exp_terms = np.exp(beta * (v - v_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return 0.0
    
    return v_max + np.log(sum_exp) / beta

def _local_search_smooth(signs, ap_indices, index_to_aps, deadline, rng):
    """
    Perform local search minimizing the smooth surrogate objective.
    We use coordinate descent: for each index, try flipping and accept if surrogate decreases.
    We also track the true max discrepancy.
    """
    n = len(signs)
    signs = list(signs)
    signs_arr = np.array(signs, dtype=np.int64)
    
    # Initialize prefix sums
    prefix_sums = []
    for indices in ap_indices:
        subseq = signs_arr[indices]
        ps = np.cumsum(subseq)
        prefix_sums.append(ps)
    
    # Calculate initial surrogate and true max
    beta = 1.0
    current_surrogate = 0.0
    for ps in prefix_sums:
        current_surrogate += _soft_max_surrogate(ps, beta)
        
    current_true_max = 0
    for ps in prefix_sums:
        m = np.max(np.abs(ps))
        if m > current_true_max:
            current_true_max = m
            
    best_surrogate = current_surrogate
    best_true_max = current_true_max
    best_signs = list(signs)
    
    max_ls_iters = 500
    ls_iter = 0
    
    while ls_iter < max_ls_iters and time.time() < deadline - 2:
        improved = False
        indices_to_try = list(range(n))
        rng.shuffle(indices_to_try)
        
        # Optionally anneal beta
        if ls_iter % 10 == 0:
            beta *= 1.2
            if beta > 100:
                beta = 100.0

        for i in indices_to_try:
            if time.time() > deadline - 2:
                break
                
            old_val = signs[i]
            delta = -2 * old_val
            affected_aps = index_to_aps[i]
            
            # Calculate change in surrogate for this flip
            # We need to update prefix_sums for affected APs, compute new surrogate, and revert if worse
            # To be efficient, we only recompute surrogate for affected APs? 
            # No, the surrogate is a sum over all APs. But only affected APs change.
            # So we can keep a running total of the surrogate for each AP?
            # Let's pre-compute surrogate per AP.
            
            # For simplicity in this iteration, we just evaluate the full surrogate change
            # by updating the affected APs' prefix sums and recalculating their contribution.
            
            old_ap_surr = 0
            for ap_id, pos in affected_aps:
                old_ap_surr += _soft_max_surrogate(prefix_sums[ap_id], beta)
                
            # Apply flip tentatively
            signs[i] = -old_val
            for ap_id, pos in affected_aps:
                prefix_sums[ap_id][pos:] += delta
                
            new_ap_surr = 0
            for ap_id, pos in affected_aps:
                new_ap_surr += _soft_max_surrogate(prefix_sums[ap_id], beta)
                
            delta_surr = new_ap_surr - old_ap_surr
            
            if delta_surr < 0:
                # Accept
                current_surrogate += delta_surr
                improved = True
                
                # Update true max (expensive, so do it less frequently or only on accept)
                # For now, we'll update true max every 5 accepted moves to save time
                if ls_iter % 5 == 0 or i == indices_to_try[-1]:
                    new_true_max = 0
                    for ps in prefix_sums:
                        m = np.max(np.abs(ps))
                        if m > new_true_max:
                            new_true_max = m
                    current_true_max = new_true_max
                    
                    if current_true_max < best_true_max:
                        best_true_max = current_true_max
                        best_signs = list(signs)
                        best_surrogate = current_surrogate
            else:
                # Revert
                signs[i] = old_val
                for ap_id, pos in affected_aps:
                    prefix_sums[ap_id][pos:] -= delta
                    
        ls_iter += 1
        if not improved:
            # Try to escape local minimum by random perturbation
            for _ in range(5):
                idx = rng.randint(0, n-1)
                signs[idx] = -signs[idx]
                # Update prefix sums
                for ap_id, pos in index_to_aps[idx]:
                    prefix_sums[ap_id][pos:] += -2 * signs[idx] * (-1) # Wait, delta is -2*old_val.
                    # Let's just recompute prefix sums for affected APs to be safe
            # Recompute prefix sums for affected APs properly
            # Actually, let's just break and restart if no improvement for a while
            break
            
    return best_signs, best_true_max

def _generate_legendre_sequence(n):
    """
    Generate a sequence of signs based on the Legendre symbol for a prime near n.
    """
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(np.sqrt(k)) + 1):
            if k % i == 0:
                return False
        return True

    p = n
    while not is_prime(p):
        p += 1

    seq = []
    for i in range(1, n + 1):
        if i % p == 0:
            seq.append(1)
        else:
            seq.append(pow(i, (p - 1) // 2, p))
    return seq

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Pre-compute structure
    ap_indices, index_to_aps = _build_ap_structure(n)
    
    rng = random.Random(42)
    
    best_signs_overall = None
    best_true_max_overall = float('inf')
    
    # Define starting configurations
    starts = []
    
    # 1. Legendre Symbol Sequence
    starts.append(_generate_legendre_sequence(n))
    
    # 2. Negated Legendre Symbol Sequence
    starts.append([-x for x in starts[0]])
    
    # 3. A few high-quality random starts
    for _ in range(5):
        starts.append([rng.choice((-1, 1)) for _ in range(n)])
        
    # 4. Alternating sequence
    starts.append([1 if i % 2 == 0 else -1 for i in range(n)])

    # Run local search from each start
    for start_signs in starts:
        if time.time() > deadline - 10:
            break
            
        best_signs, best_max = _local_search_smooth(start_signs, ap_indices, index_to_aps, deadline, rng)
        
        if best_max < best_true_max_overall:
            best_true_max_overall = best_max
            best_signs_overall = best_signs
            
        # If we hit the theoretical lower bound (1), we can stop early
        if best_true_max_overall == 1:
            break

    if best_signs_overall is None:
        best_signs_overall = [1] * n
        
    return best_signs_overall
# EVOLVE-BLOCK-END
