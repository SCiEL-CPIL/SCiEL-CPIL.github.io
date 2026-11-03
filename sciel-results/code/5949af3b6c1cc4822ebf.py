"""Strategy: Spectral Initialization + Simulated Annealing with Tabu Memory."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    Uses precomputed a_sym = a + a.T for efficiency.
    """
    x = p[i]
    y = p[j]
    
    # diff_a[k] = (a[i,k] + a[k,i]) - (a[j,k] + a[k,j])
    # diff_b[loc] = b[y, loc] - b[x, loc]
    diff_a = a_sym[i] - a_sym[j]
    diff_b = b[y] - b[x]
    
    # delta = sum_{k != i,j} diff_a[k] * diff_b[p[k]]
    # We compute sum_{k} diff_a[k] * diff_b[p[k]] and subtract the i and j terms
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _vector_alignment_init(a, b):
    """
    Initialize permutation using spectral embedding (vector alignment).
    This provides a much better starting point than rank matching.
    """
    n = len(a)
    # Center matrices
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    
    # SVD for spectral embedding
    # We want to align the top eigenvectors of a and b
    Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
    Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
    
    # Take top k components (usually a small fraction of n is enough)
    k = min(n, 50)
    Xa = Ua[:, :k]
    Xb = Vb[:, :k] # Note: Depending on convention, we might need to align rows/cols carefully.
                   # Standard QAP spectral method aligns the embeddings of facilities and locations.
                   # Let's use the standard approach: Project facilities onto location space.
    
    # A simpler and robust spectral init:
    # 1. Compute eigenvectors of A
    # 2. Compute eigenvectors of B
    # 3. Match them
    # Actually, let's use the standard "rank matching" but enhanced with a few iterations of
    # "soft assignment" or just use the SVD of the interaction.
    
    # Let's stick to a robust spectral method:
    # Embed facilities in R^k using top eigenvectors of A
    # Embed locations in R^k using top eigenvectors of B
    # Then solve the linear assignment problem between the embeddings.
    
    # Since scipy is available, we can use linear_sum_assignment for the final step.
    from scipy.optimize import linear_sum_assignment
    
    # Top eigenvectors of A (facilities)
    w_a, v_a = np.linalg.eigh(a_centered)
    # Take top k
    k = min(n, 32)
    idx_a = np.argsort(w_a)[::-1][:k]
    X_a = v_a[:, idx_a]
    
    # Top eigenvectors of B (locations)
    w_b, v_b = np.linalg.eigh(b_centered)
    idx_b = np.argsort(w_b)[::-1][:k]
    X_b = v_b[:, idx_b]
    
    # We want to assign facility i to location p[i] to maximize sum a[i,j] b[p[i],p[j]]
    # This is approximated by aligning X_a and X_b.
    # Cost matrix for linear assignment: -X_a @ X_b.T
    # We want to minimize cost, so we maximize similarity.
    
    # However, the dimensions must match for the assignment problem.
    # X_a is (n, k), X_b is (n, k).
    # The "distance" between facility i embedding and location l embedding is ||X_a[i] - X_b[l]||^2
    # = ||X_a[i]||^2 + ||X_b[l]||^2 - 2 X_a[i] @ X_b[l].T
    # Minimizing distance is equivalent to maximizing X_a[i] @ X_b[l].T
    
    # Create cost matrix C where C[i, l] = - (X_a[i] dot X_b[l])
    # This is an n x n matrix.
    C = - (X_a @ X_b.T)
    
    # Solve linear assignment
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    # p[i] is the location for facility i
    # row_ind is facilities, col_ind is locations
    for i, l in zip(row_ind, col_ind):
        p[i] = l
        
    return p

def _solve_tai256c(a, b, time_limit=145.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # 1. Initialization
    p = _vector_alignment_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 1000.0
    T_end = 1.0
    # We will run for a fixed number of iterations or until time limit
    # Since n=256, n^2/2 swaps per full pass is ~32k swaps.
    # Let's do random swaps instead of full passes to allow more flexibility.
    
    # Tabu list to avoid cycling
    tabu_length = 50
    tabu = {} # (i, j) -> time when it becomes available
    current_time_step = 0
    
    # Estimate number of steps we can do.
    # Each step involves O(n) delta calculation (vectorized).
    # Let's just loop until time limit.
    
    while time.time() - start_time < time_limit:
        current_time_step += 1
        
        # Temperature cooling (geometric or linear)
        # Let's use a simple cooling schedule
        # If we don't know total steps, use a time-based or step-based heuristic.
        # Let's assume we can do ~10k steps in 100s? No, vectorized O(n) is fast.
        # Let's just use a fixed number of steps per "phase" or dynamic T.
        
        # Dynamic T:
        # If we haven't improved in a while, lower T.
        # For simplicity, let's use a geometric cooling based on step count,
        # but reset if we find a new best? No, standard SA doesn't reset.
        
        # Let's use a simple linear cooling from T_start to T_end over the available time.
        # We need to estimate total steps.
        # Let's just use a fixed number of iterations for the whole run, say 20000.
        # If we finish early, we can restart or continue.
        
        # Actually, let's just pick a random pair and decide.
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        # Check tabu
        key = (min(i, j), max(i, j))
        if key in tabu and tabu[key] > current_time_step:
            continue
            
        delta = _swap_delta(a, b, p, i, j, a_sym)
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvement
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            # We need T. Let's use a time-based T.
            elapsed = time.time() - start_time
            # Map elapsed time to T.
            # Let's say T goes from 1000 to 1 over 100 seconds.
            progress = min(1.0, elapsed / 100.0)
            T = T_start * (T_end / T_start) ** progress
            
            if T > 0:
                prob = np.exp(-delta / T)
                accept = random.random() < prob
            else:
                accept = False
        
        if accept:
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            # Update tabu
            tabu[key] = current_time_step + tabu_length
            # Remove old tabu entries to keep dict small
            if len(tabu) > 1000:
                # Simple cleanup: remove entries older than tabu_length * 2
                for k in list(tabu.keys()):
                    if tabu[k] < current_time_step - tabu_length:
                        del tabu[k]
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p.tolist()

def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
