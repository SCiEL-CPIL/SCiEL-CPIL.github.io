"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product


def _build_cut_matrix(n, edges):
    """
    Build the MaxCut cost matrix C.
    C_ij = 1 if (i,j) is an edge, else 0.
    The expected cut value for assignment s (s_i = +/-1) is (1/4) * s^T C s.
    The unitary is exp(-i gamma C), so C is the adjacency matrix of the graph.
    """
    C = np.zeros((n, n))
    for i, j in edges:
        C[i, j] = 1.0
        C[j, i] = 1.0
    return C


def _exhaustive_maxcut(n, edges):
    """
    Compute the exact maximum cut value via exhaustive search.
    Returns the maximum number of cut edges.
    """
    if n <= 20:
        best = 0
        # Represent assignment as integer bitmask
        for mask in range(1 << n):
            cut_count = 0
            for i, j in edges:
                # Check if i and j have different bits
                bit_i = (mask >> i) & 1
                bit_j = (mask >> j) & 1
                if bit_i != bit_j:
                    cut_count += 1
            if cut_count > best:
                best = cut_count
        return float(best)
    else:
        # Fallback for larger n (not expected in this task)
        # Use a simple greedy heuristic or return a lower bound
        # For n>20, exhaustive is too slow. But task says n=6,8,10.
        # We'll use a quick random restart greedy just in case.
        best = 0
        for _ in range(100):
            assign = np.random.randint(0, 2, n)
            cut_count = 0
            for i, j in edges:
                if assign[i] != assign[j]:
                    cut_count += 1
            best = max(best, cut_count)
        return float(best)


def _expected_cut_value(n, edges, angles):
    """
    Compute the expected cut value for given QAOA angles.
    angles: (p, 2) array of (gamma, beta).
    
    The state is |psi> = U_p ... U_1 |+>^n
    U_k = exp(-i beta_k X_total) exp(-i gamma_k C)
    
    Expected cut = <psi| (1/4) C |psi>  (since C is the MaxCut operator in the Pauli Z basis? No.)
    Wait, the MaxCut Hamiltonian is H_C = -1/4 sum_{(i,j) in E} (Z_i Z_j - 1) ?
    Usually MaxCut is defined as maximizing sum_{(i,j)} (1 - Z_i Z_j)/2.
    The operator to measure is A = sum_{(i,j)} Z_i Z_j.
    The expected cut size is E[Cut] = (1/2) * |E| - (1/4) * <A>.
    Alternatively, E[Cut] = (1/4) * <sum_{(i,j)} (1 - Z_i Z_j)> = |E|/2 - (1/4)<A>.
    
    Let's compute <A> = <psi| A |psi> where A = sum_{(i,j) in E} Z_i Z_j.
    This is equivalent to computing the expectation value of the Pauli string Z_i Z_j for each edge.
    
    We can simulate the state vector directly since n is small (<=10).
    Dimension = 2^n. For n=10, 2^10 = 1024. This is very fast.
    """
    dim = 1 << n
    # Initial state |+>^n = (1/sqrt(2^n)) * ones
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute matrices for X_total and C
    # X_total = sum_i X_i. In the computational basis, X flips a bit.
    # It's easier to apply the unitaries step by step.
    
    # Matrix for exp(-i gamma C)
    # C is the adjacency matrix. The operator is C itself (or -C? depends on convention).
    # The task says exp(-i gamma C).
    # We can exponentiate the matrix expm(-1j * gamma * C_matrix)
    # But doing this for every evaluation step is expensive if we do it naively.
    # However, n is small. We can precompute the eigen-decomposition of C.
    # C is real symmetric. C = V diag(lambda) V^T.
    # exp(-i gamma C) = V diag(exp(-i gamma lambda)) V^T.
    
    C_mat = _build_cut_matrix(n, edges)
    
    # Eigen-decompose C
    eigenvalues, eigenvectors = np.linalg.eigh(C_mat)
    
    # X_total operator
    # X_i flips bit i.
    # X_total = sum_i X_i.
    # We can construct the matrix for X_total.
    X_total_mat = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for mask in range(dim):
            # Flip bit i
            new_mask = mask ^ (1 << i)
            X_total_mat[new_mask, mask] += 1.0
            
    # Eigen-decompose X_total for fast exponentiation?
    # X_total is also real symmetric.
    X_eigenvalues, X_eigenvectors = np.linalg.eigh(X_total_mat)
    
    # Now apply the layers
    for k in range(len(angles)):
        gamma, beta = angles[k]
        
        # Apply exp(-i gamma C)
        # V diag(exp(-i gamma lambda)) V^T
        phase_C = np.exp(-1j * gamma * eigenvalues)
        U_C = (eigenvectors * phase_C) @ eigenvectors.T.conj()
        state = U_C @ state
        
        # Apply exp(-i beta X_total)
        phase_X = np.exp(-1j * beta * X_eigenvalues)
        U_X = (X_eigenvectors * phase_X) @ X_eigenvectors.T.conj()
        state = U_X @ state
    
    # Compute <A> = sum_{(i,j) in E} <Z_i Z_j>
    # Z_i Z_j expectation:
    # For a state vector |psi>, <Z_i Z_j> = sum_{mask} psi^*[mask] psi[mask * (1<<i) * (1<<j)] ? No.
    # Z_i is diagonal with entries +1 if bit i is 0, -1 if bit i is 1.
    # Z_i Z_j is diagonal with entry (+1) if bit i == bit j, (-1) if bit i != bit j.
    # So <Z_i Z_j> = sum_{mask} |psi[mask]|^2 * (1 if bit_i == bit_j else -1)
    
    probs = np.abs(state) ** 2
    
    exp_A = 0.0
    for i, j in edges:
        exp_zz = 0.0
        for mask in range(dim):
            bit_i = (mask >> i) & 1
            bit_j = (mask >> j) & 1
            if bit_i == bit_j:
                exp_zz += probs[mask]
            else:
                exp_zz -= probs[mask]
        exp_A += exp_zz
    
    # Expected cut size
    # E[Cut] = |E|/2 - (1/4) * <A>
    # Wait, let's verify.
    # Cut size for assignment s is sum_{(i,j)} (1 - s_i s_j)/2 where s_i, s_j in {+1, -1}.
    # If s_i, s_j are Z eigenvalues (+1/-1).
    # (1 - Z_i Z_j)/2 is 1 if different, 0 if same.
    # So E[Cut] = (1/2) sum_{(i,j)} (1 - <Z_i Z_j>) = |E|/2 - (1/4) <A> where A = sum Z_i Z_j.
    # Yes.
    
    num_edges = len(edges)
    expected_cut = num_edges / 2.0 - 0.25 * exp_A
    
    return expected_cut


def _optimize_angles_for_graph(n, edges, p, deadline):
    """
    Optimize angles for a specific graph.
    Uses gradient-free optimization (Nelder-Mead) or a simple local search.
    Since the landscape is smooth and low-dimensional (2*p), Nelder-Mead is good.
    """
    from scipy.optimize import minimize
    
    # Initial guess
    # Common good starts: gamma ~ pi/4, beta ~ pi/8 or similar.
    # Or random.
    x0 = np.random.uniform(0, np.pi, (p, 2))
    
    def objective(angles_flat):
        if time.time() > deadline:
            return 1e10 # Penalty
        angles_2d = angles_flat.reshape(p, 2)
        val = _expected_cut_value(n, edges, angles_2d)
        return -val # Minimize negative value
        
    # Use Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
        if res.success or res.fun < 1e9:
            return res.x.reshape(p, 2)
    except Exception:
        pass
    
    # Fallback: simple random search if optimization fails or is too slow
    best_val = -1e9
    best_angles = x0
    for _ in range(100):
        if time.time() > deadline:
            break
        cand = np.random.uniform(0, np.pi, (p, 2))
        val = _expected_cut_value(n, edges, cand)
        if val > best_val:
            best_val = val
            best_angles = cand
            
    return best_angles


def angles(n, edges, p):
    """
    Interface: angles(n, edges, p) -> real array (p, 2)
    """
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    # Check if we have enough time
    if time.time() > deadline:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Optimize
    try:
        opt_angles = _optimize_angles_for_graph(n, edges, p, deadline)
        return opt_angles
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

# EVOLVE-BLOCK-END
