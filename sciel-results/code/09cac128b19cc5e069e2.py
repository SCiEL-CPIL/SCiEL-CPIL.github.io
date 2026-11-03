"""Matrix multiplication 2,3,9 tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2_terms():
    """
    7-term Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns list of (u_dict, v_dict, w_dict) where:
      u_dict: {(i,j): coeff} for i,j in {0,1}
      v_dict: {(j,k): coeff} for j,k in {0,1}
      w_dict: {(i,k): coeff} for i,k in {0,1}
    """
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

def _construct_2x3x3_rank8():
    """
    Attempt to construct a rank-8 decomposition of 2x3 x 3x3 matrix multiplication.
    Uses a structured approach: split the 3 columns into (1,2) and use Strassen for 2x2x2 blocks.
    
    Actually, let's use a known approach: 
    2x3 x 3x3 can be viewed as:
    - First column: 2x1 x 1x3 (schoolbook: 6 terms)
    - Last two columns: 2x2 x 2x3 (need decomposition)
    
    For 2x2 x 2x3: split into 2x2x2 (Strassen: 7) + 2x2x1 (schoolbook: 4) = 11
    Total: 6 + 11 = 17. Not great.
    
    Alternative: Use the fact that 2x3x3 has rank at most 2*3*3=18 (schoolbook).
    Let's try a numerical search for rank 8.
    """
    n, m, p = 2, 3, 3
    
    # Try to find rank-8 decomposition numerically
    # Parameters: U (6x8), V (9x8), W (6x8)
    # Objective: minimize ||T_computed - T_expected||_F
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _objective(params):
        """Compute residual for given parameter vector."""
        # Reshape parameters
        U_flat = params[:6*8]
        V_flat = params[6*8:6*8+9*8]
        W_flat = params[6*8+9*8:6*8+9*8+6*8]
        
        U = U_flat.reshape(6, 8)
        V = V_flat.reshape(9, 8)
        W = W_flat.reshape(6, 8)
        
        # Compute T_computed[i*m+j, j*p+k, k*n+i]
        T_computed = np.zeros((6, 9, 6))
        for c in range(8):
            u_vec = U[:, c]  # length 6
            v_vec = V[:, c]  # length 9
            w_vec = W[:, c]  # length 6
            T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
        # Expected: T[i*m+j, j*p+k, k*n+i] = 1
        T_expected = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T_expected[i*3+j, j*3+k, k*2+i] = 1
        
        # Residual
        return (T_computed - T_expected).flatten()
    
    # Multiple random restarts
    best_params = None
    best_resid = np.inf
    
    for attempt in range(50):
        # Initialize near schoolbook
        np.random.seed(attempt * 12345)
        
        # Start from a random rank-8 initialization
        params = np.random.randn(6*8 + 9*8 + 6*8) * 0.5
        
        try:
            result = least_squares(_objective, params, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.cost < best_resid:
                best_resid = result.cost
                best_params = result.x
        except Exception:
            continue
        
        if best_resid < 1e-10:
            break
    
    if best_params is not None and best_resid < 1e-10:
        U = best_params[:6*8].reshape(6, 8)
        V = best_params[6*8:6*8+9*8].reshape(9, 8)
        W = best_params[6*8+9*8:6*8+9*8+6*8].reshape(6, 8)
        
        # Convert to lists of dicts
        U_terms = []
        V_terms = []
        W_terms = []
        for c in range(8):
            U_term = {}
            for i in range(2):
                for j in range(3):
                    if abs(U[i*3+j, c]) > 1e-10:
                        U_term[(i, j)] = float(U[i*3+j, c])
            V_term = {}
            for j in range(3):
                for k in range(3):
                    if abs(V[j*3+k, c]) > 1e-10:
                        V_term[(j, k)] = float(V[j*3+k, c])
            W_term = {}
            for i in range(2):
                for k in range(3):
                    if abs(W[i*3+k, c]) > 1e-10:
                        W_term[(i, k)] = float(W[i*3+k, c])
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
        
        return list(zip(U_terms, V_terms, W_terms))
    
    return None

def _verify_2x3x3(terms):
    """Verify rank-8 decomposition for 2x3x3."""
    n, m, p = 2, 3, 3
    T_computed = np.zeros((n*m, m*p, p*n))
    for (u_dict, v_dict, w_dict) in terms:
        for (i, j), uc in u_dict.items():
            for (jj, k), vc in v_dict.items():
                for (ii, kk), wc in w_dict.items():
                    if jj == j:  # must match middle index
                        T_computed[i*m+j, j*p+k, k*n+i] += uc * vc * wc
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    return np.max(np.abs(T_computed - T_expected)) < 1e-10

def _construct_2x3x9_rank24():
    """
    Construct rank-24 decomposition for 2x3 x 3x9 by applying
    rank-8 decomposition of 2x3x3 to each of 3 blocks of 3 columns.
    """
    n, m, p = 2, 3, 9
    base_terms = _construct_2x3x3_rank8()
    if base_terms is None:
        return None
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for q in range(3):
        for (u_dict, v_dict, w_dict) in base_terms:
            U_term = {}
            for (i, j), coeff in u_dict.items():
                U_term[(i, j)] = coeff
            
            V_term = {}
            for (j, k), coeff in v_dict.items():
                V_term[(j, q*3 + k)] = coeff
            
            W_term = {}
            for (i, k), coeff in w_dict.items():
                W_term[(i, q*3 + k)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-10

def _construct_2x3x9_strassen():
    """Strassen-based decomposition achieving rank 50."""
    n, m, p = 2, 3, 9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = dict(u_dict)
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
    for k in k_single:
        for i in range(2):
            for j in range(2):
                U_terms.append({(i, j): 1})
                V_terms.append({(j, k): 1})
                W_terms.append({(i, k): 1})
    
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
    
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _construct_2x3x9_rank36():
    """
    Alternative approach: split 3x9 into three 3x3 blocks and use
    a better decomposition for each block.
    
    For 2x3 x 3x3, we can try to find rank-12 decomposition (better than schoolbook 18).
    3 blocks * 12 = 36.
    """
    n, m, p = 2, 3, 9
    
    # For 2x3 x 3x3, try to find a rank-12 decomposition
    # Parameters: U (6x12), V (9x12), W (6x12)
    
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None
    
    def _objective(params):
        U_flat = params[:6*12]
        V_flat = params[6*12:6*12+9*12]
        W_flat = params[6*12+9*12:6*12+9*12+6*12]
        
        U = U_flat.reshape(6, 12)
        V = V_flat.reshape(9, 12)
        W = W_flat.reshape(6, 12)
        
        T_computed = np.zeros((6, 9, 6))
        for c in range(12):
            u_vec = U[:, c]
            v_vec = V[:, c]
            w_vec = W[:, c]
            T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
        T_expected = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T_expected[i*3+j, j*3+k, k*2+i] = 1
        
        return (T_computed - T_expected).flatten()
    
    best_params = None
    best_resid = np.inf
    
    for attempt in range(30):
        np.random.seed(attempt * 67890)
        params = np.random.randn(6*12 + 9*12 + 6*12) * 0.5
        
        try:
            result = least_squares(_objective, params, max_nfev=8000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if result.cost < best_resid:
                best_resid = result.cost
                best_params = result.x
        except Exception:
            continue
        
        if best_resid < 1e-10:
            break
    
    if best_params is not None and best_resid < 1e-10:
        U = best_params[:6*12].reshape(6, 12)
        V = best_params[6*12:6*12+9*12].reshape(9, 12)
        W = best_params[6*12+9*12:6*12+9*12+6*12].reshape(6, 12)
        
        U_terms = []
        V_terms = []
        W_terms = []
        for c in range(12):
            U_term = {}
            for i in range(2):
                for j in range(3):
                    if abs(U[i*3+j, c]) > 1e-10:
                        U_term[(i, j)] = float(U[i*3+j, c])
            V_term = {}
            for j in range(3):
                for k in range(3):
                    if abs(V[j*3+k, c]) > 1e-10:
                        V_term[(j, k)] = float(V[j*3+k, c])
            W_term = {}
            for i in range(2):
                for k in range(3):
                    if abs(W[i*3+k, c]) > 1e-10:
                        W_term[(i, k)] = float(W[i*3+k, c])
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
        
        # Tile for 3x9
        all_U = []
        all_V = []
        all_W = []
        for q in range(3):
            for (u_dict, v_dict, w_dict) in zip(U_terms, V_terms, W_terms):
                U_t = dict(u_dict)
                V_t = {}
                for (j, k), coeff in v_dict.items():
                    V_t[(j, q*3+k)] = coeff
                W_t = {}
                for (i, k), coeff in w_dict.items():
                    W_t[(i, q*3+k)] = coeff
                all_U.append(U_t)
                all_V.append(V_t)
                all_W.append(W_t)
        
        rank = len(all_U)
        U_mat = [[0] * rank for _ in range(n * m)]
        V_mat = [[0] * rank for _ in range(m * p)]
        W_mat = [[0] * rank for _ in range(p * n)]
        
        for col in range(rank):
            for (i, j), coeff in all_U[col].items():
                U_mat[i * m + j][col] = coeff
            for (j, k), coeff in all_V[col].items():
                V_mat[j * p + k][col] = coeff
            for (i, k), coeff in all_W[col].items():
                W_mat[k * n + i][col] = coeff
        
        return U_mat, V_mat, W_mat
    
    return None

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Tries multiple approaches in order of increasing rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin (150s limit)
    
    # Try rank-24 first (3x rank-8)
    if time.time() < deadline:
        try:
            U, V, W = _construct_2x3x9_rank24()
            if U is not None and _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try rank-36 (3x rank-12)
    if time.time() < deadline:
        try:
            U, V, W = _construct_2x3x9_rank36()
            if U is not None and _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Fall back to Strassen rank-50
    if time.time() < deadline:
        try:
            U, V, W = _construct_2x3x9_strassen()
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Ultimate fallback: rank-54 schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
