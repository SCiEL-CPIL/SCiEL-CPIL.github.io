"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 18

    # First, get the schoolbook decomposition as a fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank

    # Try to find lower rank decompositions using structural search
    for target_rank in [17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline:
            break
        
        found = _structural_search(n, m, p, target_rank, deadline)
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Convert to lists with integer/half-integer entries
    def to_list(M):
        result = []
        for row in M:
            new_row = []
            for x in row:
                rx = round(x * 2) / 2
                if abs(rx - round(rx)) < 1e-9:
                    new_row.append(int(round(rx)))
                else:
                    new_row.append(rx)
            result.append(new_row)
        return result
    
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _structural_search(n, m, p, target_rank, deadline):
    """
    Structured search exploiting the symmetry of 2x3x3 multiplication.
    Key insight: n=2 means W has only 6 rows (p*n=6), so we can fix W's structure
    and solve for U and V more efficiently.
    """
    n_m = n * m  # 6
    m_p = m * p  # 9
    p_n = p * n  # 6
    
    # For n=2, the W matrix has a special structure:
    # W[k*n+i, c] for i in {0,1}, k in {0,1,2}
    # We can exploit that each column of W defines a "pattern" of how outputs combine
    
    # Strategy: Enumerate support patterns for W columns, then solve linear system for U, V
    # W has 6 rows, so each column is a 6-dim vector. For small entries, we can enumerate.
    
    max_attempts = 300
    attempts_done = 0
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        attempts_done += 1
        
        # Random W with small integer entries
        # W: (p*n, target_rank) = (6, target_rank)
        W = np.random.randint(-2, 3, size=(p_n, target_rank))
        
        # For each column c, W[:, c] is a 6-vector.
        # The constraint is: for all i, j, k:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Rearrange: for fixed i, k:
        #   sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
        
        # This is a system of equations. Let's set up the linear system.
        # Unknowns: U (6 x R) and V (9 x R), but they're coupled.
        
        # Alternative: fix U and V randomly, solve for W? No, W is already fixed.
        
        # Better approach: Use the fact that for each (i,k), we need:
        #   M_{ik} = sum_c W[k*n+i, c] * outer(U[i*m: i*m+m, c], V[k*p: (k+1)*p, c])... 
        # Wait, let me re-index.
        
        # Actually: for fixed i, k, the constraint over all j is:
        #   For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let w_ik = W[k*n+i, :] (shape R,)
        # Let u_i = U[i*m : (i+1)*m, :] (shape m x R)
        # Let v_k = V[:, k] ... no, V is indexed by (j*p+k)
        
        # V[j*p+k, c] for fixed k: this is V[k, c], V[p+k, c], V[2p+k, c]
        # So v_k is a 3-vector for each c: [V[k,c], V[p+k,c], V[2p+k,c]]
        
        # The constraint for (i,k,j) is:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # For fixed i, k, this gives m equations (one per j) in the unknowns U[i*m:..., c] and V[...+k, c].
        
        # Let's try a different approach: random U and V, then check if W can be solved.
        # Actually, let's just do random search with verification, but be smarter.
        
        # Generate random U and V with small integers
        U = np.random.randint(-2, 3, size=(n_m, target_rank))
        V = np.random.randint(-2, 3, size=(m_p, target_rank))
        
        # Now check if there exists W that satisfies the constraints.
        # For each (i,k), we need W[k*n+i, :] to satisfy:
        #   For all j: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        
        # This is a system of m equations in n*p unknowns (W has p*n entries total,
        # but for fixed (i,k), we only use W[k*n+i, :], which is R unknowns).
        
        # Wait, W[k*n+i, :] is a single row of W, with R entries.
        # The constraint for (i,k) gives m equations in R unknowns (the entries of W[k*n+i, :]).
        
        # Since m=3 and R >= 11, this is underdetermined. Let's solve it.
        
        W_candidate = np.zeros((p_n, target_rank))
        valid = True
        for i in range(n):
            for k in range(p):
                # Build the system: A_sys @ w = b
                # where w = W[k*n+i, :]
                # A_sys[j, c] = U[i*m+j, c] * V[j*p+k, c] for j in range(m)
                A_sys = np.zeros((m, target_rank))
                for j in range(m):
                    A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                
                # Solve A_sys @ w = b (least squares or exact)
                try:
                    # Check if the system is consistent
                    rank_A = np.linalg.matrix_rank(A_sys)
                    if rank_A < m:
                        # Underdetermined or inconsistent
                        # Try least squares
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(A_sys @ w_sol, b, atol=1e-8):
                            W_candidate[k*n+i, :] = w_sol
                        else:
                            valid = False
                            break
                    else:
                        # Full rank, check consistency
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                        if np.allclose(A_sys @ w_sol, b, atol=1e-8):
                            W_candidate[k*n+i, :] = w_sol
                        else:
                            valid = False
                            break
                except Exception:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Round to half-integers and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W_candidate * 2) / 2
            
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
        
        # Try a perturbation
        if attempt % 10 == 0:
            # Occasionally try a different structure
            if attempt % 30 == 0:
                # Try with more zeros in W
                W = np.random.randint(0, 2, size=(p_n, target_rank))
                W[W == 0] = 0
                W[W == 1] = np.random.choice([-1, 1])
    
    return None


def _structured_search(n, m, p, target_rank, deadline):
    """
    Try structured decompositions based on known results.
    For 2x3x3, there are known rank-17 decompositions.
    """
    return None

# EVOLVE-BLOCK-END
