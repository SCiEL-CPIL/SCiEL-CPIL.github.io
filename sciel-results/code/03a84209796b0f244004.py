"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import product

# Precompute or cache? Since n is small (6,8,10) and p=2, we can do this quickly.
# But we need to be careful about time. 120s limit.
# Grid search + NM refinement should be fast enough for n<=10.

def _exact_maxcut(n, edges):
    """Compute the exact maximum cut value for a graph with n nodes and given edges."""
    # For n <= 10, we can brute force 2^n assignments.
    # edges is a list of (u, v) pairs.
    # MaxCut value = sum of 1 if u and v are in different partitions, else 0.
    # We want the maximum possible sum.
    m = len(edges)
    if m == 0:
        return 0.0
    
    best_val = 0.0
    # Iterate over all 2^n cut assignments
    # Represent cut as a bitmask of length n.
    for mask in range(1 << n):
        val = 0
        for u, v in edges:
            # Check if u and v are in different partitions
            # bit u: (mask >> u) & 1
            # bit v: (mask >> v) & 1
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                val += 1
        if val > best_val:
            best_val = val
    return float(best_val)


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut size for a QAOA circuit with given angles.
    Circuit: |+>^n -> exp(-i gamma_1 C) -> exp(-i beta_1 X) -> ... -> exp(-i gamma_p C) -> exp(-i beta_p X)
    C is the MaxCut Hamiltonian: C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2 ? 
    Wait, standard QAOA MaxCut Hamiltonian is usually defined as C = sum_{(u,v) in E} (1 - Z_u Z_v)/2 or just sum (1 - Z_u Z_v).
    The problem statement says "C is unweighted MaxCut".
    Usually, the cost function for MaxCut is C = sum_{(u,v) in E} (1 - Z_u Z_v)/2.
    The probability of an edge being cut is < (1 - Z_u Z_v) / 2 >.
    Expected cut = sum_{(u,v) in E} < (1 - Z_u Z_v) / 2 >.
    
    Let's verify the standard definition.
    If we use H_C = sum_{(u,v) in E} (1 - Z_u Z_v)/2, then the expected value is the expected number of cut edges.
    This is the standard quantity.
    
    We need to compute < (1 - Z_u Z_v) / 2 > = 0.5 * (1 - <Z_u Z_v>).
    So Expected Cut = 0.5 * (M - sum_{(u,v) in E} <Z_u Z_v>).
    
    We need to compute the expectation values of Z_u Z_v for each edge.
    This requires evolving the state vector or using tensor networks.
    For n=10, the state vector has dimension 2^10 = 1024. This is very small.
    We can represent the state as a complex array of size 2^n.
    
    Steps:
    1. Initialize state |+>^n.
    2. Apply layers.
    3. Compute expectation values.
    """
    dim = 1 << n
    # Initial state: uniform superposition
    # |0>...|0> to |1>...|1> all have amplitude 1/sqrt(2^n)
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute Z_u Z_v operators? Or just compute expectations directly.
    # Applying exp(-i gamma C) and exp(-i beta H_B) where H_B = sum X_i.
    
    # H_B = sum X_i. This is a local operator.
    # exp(-i beta H_B) = tensor product of exp(-i beta X_i) for each i.
    # exp(-i beta X) = cos(beta) I - i sin(beta) X.
    # So we can apply this efficiently.
    
    # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2.
    # exp(-i gamma C) is a global operator. For n=10, we can build the matrix.
    # However, building the full 1024x1024 matrix for every evaluation might be slow if done repeatedly.
    # But grid search + NM might call this function ~100-1000 times.
    # 1024x1024 matrix multiplication is fast.
    
    # Let's build the C matrix once for a given graph.
    # But the function signature is angles(n, edges, p). It doesn't take a precomputed graph.
    # So we must compute inside.
    
    # Optimization:
    # We can precompute the C matrix for the specific (n, edges) inside the function,
    # cache it if possible? No, we can't easily cache across calls if edges change.
    # But for a single call to angles(), we will call _expected_cut many times.
    # So we should build the C matrix once per angles() call, and pass it to a helper.
    
    # Let's restructure.
    pass


def _build_cost_matrix(n, edges):
    """Build the cost matrix C for the graph."""
    dim = 1 << n
    C = np.zeros((dim, dim), dtype=np.complex128)
    
    # C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # This is a diagonal matrix in the computational basis.
    # For a basis state |s>, the value of (1 - Z_u Z_v)/2 is:
    # If s_u == s_v, Z_u Z_v = 1, term is 0.
    # If s_u != s_v, Z_u Z_v = -1, term is 1.
    # So C is diagonal, with entries equal to the number of cut edges in the cut defined by |s>.
    
    for s in range(dim):
        val = 0
        for u, v in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                val += 1
        C[s, s] = val
        
    return C


def _expected_cut_with_matrix(n, edges, C, gammas, betas):
    """
    Compute expected cut given precomputed C matrix.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    
    # Precompute exp(-i beta X) for each layer?
    # H_B = sum X_i.
    # U_B(beta) = exp(-i beta sum X_i) = tensor product of U_x(beta) for each i.
    # U_x(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    
    # Applying U_B to state:
    # We can do this by iterating over bits? Or use matrix multiplication.
    # Since dim is small, we can build the full U_B matrix.
    # U_B is a 2^n x 2^n matrix.
    
    # Actually, applying a tensor product of 2x2 matrices is efficient if done carefully,
    # but for n=10, building the full matrix is also fine.
    # Let's build U_B for each beta.
    
    for i in range(p):
        # Apply exp(-i gamma_i C)
        # C is diagonal, so this is just multiplying state by exp(-i gamma_i C_diag)
        c_diag = np.diag(C)
        state = state * np.exp(-1j * gammas[i] * c_diag)
        
        # Apply exp(-i beta_i H_B)
        # Build U_B matrix
        # U_B = tensor product of U_x(beta_i)
        # This is expensive to build from scratch each time.
        # Alternative: Apply X rotations bit by bit?
        # No, H_B is sum of X_i, so the operators commute.
        # We can apply each X_i rotation sequentially.
        # Applying exp(-i beta X_k) to the state.
        # This affects only the k-th qubit.
        
        # Let's implement a function to apply a single qubit rotation X to the state.
        # State is indexed by binary string.
        # For qubit k, we pair states that differ only in bit k.
        
        beta = betas[i]
        c_beta = np.cos(beta)
        s_beta = -1j * np.sin(beta)
        
        # Apply rotation to qubit k
        # We need to iterate over all pairs of basis states differing in bit k.
        # This is O(2^n) per qubit, so O(n 2^n) per layer.
        # For n=10, this is 10 * 1024 = 10240 operations. Very fast.
        
        # However, doing this in a Python loop over 2^n/2 pairs might be slow.
        # Let's use vectorized NumPy operations.
        
        # Reshape state into a tensor of shape (2, 2, ..., 2) with n dimensions.
        # Then apply the rotation to the k-th axis.
        
        # state_tensor = state.reshape([2]*n)
        # Apply rotation on axis k.
        
        # The rotation matrix is [[c, s], [s, c]] where c=cos(beta), s=-i sin(beta).
        # Note: The standard X rotation is exp(-i beta X) = [[cos, -i sin], [-i sin, cos]].
        # So the matrix is:
        # [ [cos, -i sin],
        #   [-i sin, cos] ]
        
        # Let's use np.einsum or simple indexing.
        
        # Actually, building the full U_B matrix might be simpler and fast enough for n=10.
        # 1024x1024 matrix multiplication is ~1M operations. 
        # If we have ~1000 evaluations, that's 1B operations. Might be too slow in Python.
        
        # Let's stick to the tensor reshape method which is O(n 2^n) per layer.
        
        # Reshape to tensor
        shape = [2] * n
        tensor = state.reshape(shape)
        
        # Apply rotation on axis k? No, H_B = sum X_i, so we apply exp(-i beta X_i) for all i.
        # Since they commute, we can apply them one by one.
        # But applying one by one requires reshaping/unreshaping or careful indexing.
        
        # Alternative: Build the full U_B matrix once for a given beta?
        # No, beta changes.
        
        # Let's try building the full U_B matrix.
        # U_B = kron(U_x(beta), U_x(beta), ..., U_x(beta))
        
        # We can build it recursively or using np.kron.
        # For n=10, np.kron of 10 2x2 matrices results in a 1024x1024 matrix.
        # This is fast in C/NumPy.
        
        U_x = np.array([[np.cos(beta), -1j*np.sin(beta)],
                        [-1j*np.sin(beta), np.cos(beta)]], dtype=np.complex128)
        
        U_B = U_x
        for _ in range(n - 1):
            U_B = np.kron(U_B, U_x)
            
        state = U_B @ state
        
    # Compute expected cut
    # E[C] = sum_{s} |<s|psi>|^2 * C[s,s]
    probs = np.abs(state) ** 2
    c_diag = np.diag(C)
    expected_cut = np.dot(probs, c_diag)
    
    return float(np.real(expected_cut))


def angles(n, edges, p):
    # 1. Compute exact optimum
    opt_val = _exact_maxcut(n, edges)
    if opt_val == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    # 2. Build cost matrix
    C = _build_cost_matrix(n, edges)
    
    # 3. Define objective function for minimization (we want to maximize ratio)
    # Ratio = expected_cut / opt_val
    # We want to maximize this.
    # scipy.optimize.minimize minimizes, so we minimize -ratio.
    
    def objective(params):
        # params is a flat array of length 2*p: [gamma_1, beta_1, ..., gamma_p, beta_p]
        # Or [gamma_1, ..., gamma_p, beta_1, ..., beta_p]?
        # The return format is (p, 2) with rows (gamma, beta).
        # Let's use flat array [g1, b1, g2, b2, ...] for ease of slicing.
        
        gammas = params[0::2]
        betas = params[1::2]
        
        exp_cut = _expected_cut_with_matrix(n, edges, C, gammas, betas)
        ratio = exp_cut / opt_val
        return -ratio
    
    # 4. Grid search
    # Coarse grid over a reasonable range.
    # Typical QAOA angles are in [0, pi] or [0, pi/2].
    # Let's use [0, pi] for both gamma and beta.
    
    # Grid size: 5x5 per angle? That's 25^p. For p=2, 625 points.
    # Let's try 6x6 grid for each of the 4 angles? No, that's 6^4 = 1296 points.
    # Maybe 5x5 for gamma, 5x5 for beta.
    
    n_grid = 6
    grid_points = np.linspace(0.1, np.pi - 0.1, n_grid)
    
    best_params = None
    best_score = -np.inf
    
    # Iterate over grid
    for g1 in grid_points:
        for b1 in grid_points:
            for g2 in grid_points:
                for b2 in grid_points:
                    params = np.array([g1, b1, g2, b2])
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
    
    # 5. Nelder-Mead refinement
    # Start from the best grid point.
    result = minimize(objective, best_params, method='Nelder-Mead', 
                      options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_params = result.x
    
    # 6. Format output
    # Convert flat params [g1, b1, g2, b2] to (p, 2) array
    out = np.zeros((p, 2))
    for i in range(p):
        out[i, 0] = best_params[2*i]
        out[i, 1] = best_params[2*i + 1]
        
    return out

# EVOLVE-BLOCK-END
