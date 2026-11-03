"""
Quantum-Inspired Annealing for QAPLIB wil100.
Strategy:
1. Represent the assignment as a complex-valued unitary matrix U.
2. Initialize U using a greedy heuristic or random initialization.
3. Update U via a quantum rotation operator: U_new = exp(-i * t * H) * U_old,
   where H is a Hamiltonian derived from the QAP cost gradient.
4. Extract a permutation from U (e.g., via Hungarian algorithm on the real part or magnitude).
5. Refine with classical local search to exploit the "tunneled" states.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i and k=j
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _local_search(a, b, p, time_limit, start_time):
    """
    First-improvement local search.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            no_improve = 0
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return best_p, best_cost

def _quantum_rotation(U, H, t):
    """
    Apply quantum rotation: U_new = exp(-i * t * H) * U_old.
    Since H is Hermitian, exp(-i * t * H) is unitary.
    We approximate the exponential using a Taylor series or matrix exponential.
    For performance, we use a simple first-order approximation: 
    U_new ≈ (1 - i * t * H) * U_old, then re-unitarize.
    """
    # First-order approximation
    U_new = (1 - 1j * t * H) @ U
    
    # Re-unitarize using SVD to ensure U remains unitary
    # U_new = W * Vh, where W and Vh are unitary
    try:
        U_svd, _, Vh_svd = np.linalg.svd(U_new)
        U_new = U_svd @ Vh_svd
    except np.linalg.LinAlgError:
        pass
        
    return U_new

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    # Initialize U as a random unitary matrix
    # Generate a random complex matrix and unitarize it
    Z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    U, _ = np.linalg.qr(Z)
    
    # Ensure determinant is 1 if necessary (SU(n) vs U(n)), but U(n) is fine for permutations
    # Actually, for permutation extraction, we just need a good unitary.
    
    best_p = None
    best_cost = np.inf
    
    # Quantum Annealing Parameters
    T_start = 10.0
    T_end = 0.1
    num_steps = 100
    
    # Precompute some things for speed if possible, but H depends on U
    
    for step in range(num_steps):
        elapsed = time.time() - start_time
        if elapsed > total_deadline - 10:
            break
        
        # Temperature schedule (geometric cooling)
        T = T_start * (T_end / T_start) ** (step / num_steps)
        
        # Extract current permutation from U
        # Use the real part of U as the cost matrix for LAP? Or magnitude?
        # Let's use the real part, as it represents the "alignment"
        cost_matrix = np.real(U)
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(-cost_matrix) # Minimize negative = Maximize alignment
        p = np.zeros(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        # Evaluate exact cost
        cost = _qap_cost(a, b, p)
        
        # Local Search Refinement
        ls_time = min(1.0, total_deadline - (time.time() - start_time) - 5)
        if ls_time > 0.1:
            p_ls, cost_ls = _local_search(a, b, p.copy(), ls_time, start_time)
        else:
            p_ls, cost_ls = p.copy(), cost
        
        # Update best
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
        
        # Compute Gradient/Hamiltonian
        # The gradient of the QAP cost with respect to the assignment matrix X (where X[i, p[i]] = 1)
        # is G[i, j] = sum_k (a[i, k] * X[k, j] + a[k, i] * X[i, k]) * (b[j, p[k]] + b[p[k], j]) ?
        # Actually, the gradient of QAP cost C = sum a[i,j] b[p[i], p[j]] with respect to X[i, j]
        # is G[i, j] = sum_k (a[i, k] b[j, p[k]] + a[k, i] b[p[k], j])
        
        # Let's compute the gradient matrix G
        # b_j_p = b[:, p] -> b[j, p[k]]
        # b_p_j = b[p, :] -> b[p[k], j]
        
        b_j_p = b[:, p_ls]
        b_p_j = b[p_ls, :]
        
        term1 = a @ b_j_p.T # a[i, k] b[j, p[k]]
        term2 = a.T @ b_p_j # a[k, i] b[p[k], j]
        
        G = term1 + term2
        
        # Construct a Hermitian Hamiltonian H from G
        # H = (G + G.T) / 2 + 1j * (G - G.T) / 2 ? No, G is real.
        # So H = G (if we treat it as real symmetric part) or just use G as the generator.
        # To make it Hermitian: H = G.T @ G ? No.
        # Let's use H = G (assuming G is symmetric enough or we symmetrize)
        H = (G + G.T) / 2.0
        
        # Normalize H to control the step size
        h_norm = np.linalg.norm(H)
        if h_norm > 0:
            H = H / h_norm
        
        # Apply Quantum Rotation
        # t is related to temperature T
        t = T
        U = _quantum_rotation(U, H, t)
        
    # Final polish with SA from best solution
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3 and best_p is not None:
        sa_time = total_deadline - elapsed - 2
        p_sa = best_p.copy()
        cost_sa = best_cost
        T = 100.0
        while time.time() - start_time < total_deadline - 1:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = _swap_delta(a, b, p_sa, i, j)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                p_sa[i], p_sa[j] = p_sa[j], p_sa[i]
                cost_sa += delta
                if cost_sa < best_cost:
                    best_cost = cost_sa
                    best_p = p_sa.copy()
            T = max(1.0, T * 0.99)
            
    if best_p is None:
        # Fallback
        best_p = np.arange(n)
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
