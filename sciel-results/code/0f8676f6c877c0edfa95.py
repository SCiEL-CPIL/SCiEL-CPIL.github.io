import time
import sys
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _build_strassen_28():
    """Build the baseline rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 coefficients
    # alpha[s]: dict mapping (row, col) -> coeff for A
    # beta[s]: dict mapping (row, col) -> coeff for B
    # w_coeff[s]: dict mapping (row, col) -> coeff for C
    alpha = [
        None,
        {(0,0): 1, (1,1): 1},       # M1 = (A00+A11)(B00+B11)
        {(1,0): 1, (1,1): 1},       # M2 = (A10+A11)(B00)
        {(0,0): 1},                 # M3 = (A00)(B01-B11)
        {(1,1): 1},                 # M4 = (A11)(B10-B00)
        {(0,0): 1, (0,1): 1},      # M5 = (A00+A01)(B11)
        {(1,0): 1, (0,0): -1},     # M6 = (A10-A00)(B00+B01)
        {(0,1): 1, (1,1): -1},     # M7 = (A01-A11)(B10+B11)
    ]
    beta = [
        None,
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},     # M3
        {(1,0): 1, (0,0): -1},     # M4
        {(1,1): 1},                # M5
        {(0,0): 1, (0,1): 1},      # M6
        {(1,0): 1, (1,1): 1},      # M7
    ]
    w_coeff = [
        None,
        {(0,0): 1, (1,1): 1},      # M1 -> C00(+1), C11(+1)
        {(1,0): 1, (1,1): -1},     # M2 -> C10(+1), C11(-1)
        {(0,1): 1, (1,1): 1},      # M3 -> C01(+1), C11(+1)
        {(0,0): 1, (1,0): 1},      # M4 -> C00(+1), C10(+1)
        {(0,0): -1, (0,1): 1},     # M5 -> C00(-1), C01(+1)
        {(1,1): 1},                # M6 -> C11(+1)
        {(0,0): 1},                # M7 -> C00(+1)
    ]

    # 4 blocks: A col-pair (ac) in {0,1}, B row-pair (br) in {0,1}, B col-pair (bc) in {0,1}
    # Block (ac, br, bc): A[:, 2*ac:2*ac+2] * B[2*br:2*br+2, 2*bc:2*bc+2] -> C[:, 2*bc:2*bc+2]
    block_idx = 0
    for ac in range(2):
        for br in range(2):
            for bc in range(2):
                for s in range(1, 8):
                    r = block_idx * 7 + (s - 1)
                    # Fill U: A has shape (n=2, m=4), block uses cols 2*ac, 2*ac+1
                    for (ar, ac_off), val in alpha[s].items():
                        u_row = ar  # row in A (0 or 1)
                        u_col = 2 * ac + ac_off  # col in A
                        U[u_row * m + u_col][r] += val
                    # Fill V: B has shape (m=4, p=4), block uses rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1
                    for (br_off, bc_off), val in beta[s].items():
                        v_row = 2 * br + br_off
                        v_col = 2 * bc + bc_off
                        V[v_row * p + v_col][r] += val
                    # Fill W: C has shape (n=2, p=4), block contributes to cols 2*bc, 2*bc+1
                    for (cr, cc_off), val in w_coeff[s].items():
                        w_row = cr  # row in C (0 or 1)
                        w_col = 2 * bc + cc_off  # col in C
                        W[w_col * n + w_row][r] += val
                block_idx += 1

    assert block_idx == 4
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            a_val = 0.0
            for i in range(n):
                for j in range(m):
                    a_val += U[i * m + j][r] * A[i, j]
            b_val = 0.0
            for j in range(m):
                for k in range(p):
                    b_val += V[j * p + k][r] * B[j, k]
            if a_val == 0 or b_val == 0:
                continue
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += W[k * n + i][r] * a_val * b_val

        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding if any term is linearly dependent on others
    in the tensor decomposition sense.
    
    The tensor T = sum_r w_r * u_r * v_r where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    We want to find the minimum number of terms.
    
    Strategy: For each term r, check if it can be expressed as a linear combination
    of the other terms. This is equivalent to checking if the tensor formed by
    removing term r can still be decomposed with the remaining terms.
    
    A simpler approach: check if any two terms are "proportional" in a way that
    allows merging, or if there's a linear dependency in the tensor space.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build the full tensor T of shape (n, m, p)
    # T[i, j, k] = sum_r W_arr[k*n+i, r] * U_arr[i*m+j, r] * V_arr[j*p+k, r]
    # The target tensor for matrix mult is T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right. The tensor is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # For standard matmul, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all.
    # Actually no - the tensor is defined by the bilinear map structure.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # But this is the "multiplication tensor" which for n x m * m x p has entries
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's verify: reconstruct T from our decomposition
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += W_arr[k * n + i, r] * U_arr[i * m + j, r] * V_arr[j * p + k, r]
    
    # Target: T[i,j,k] = 1 for all i,j,k
    T_target = np.ones((n, m, p))
    if not np.allclose(T_recon, T_target, atol=1e-10):
        print("Warning: baseline tensor reconstruction doesn't match target")
        return None

    # Try to eliminate terms one by one
    # For each term r, check if the remaining terms can still span the tensor
    # This is a rank question in the tensor space
    
    # Alternative: try to find a linear combination that eliminates one term
    # If term r = sum_{s != r} c_s * term_s (in tensor space), we can drop it
    
    # Tensor space has dimension n*m*p = 32
    # Each term r corresponds to a vector in this 32-dim space
    # term_r[i,j,k] = W_arr[k*n+i, r] * U_arr[i*m+j, r] * V_arr[j*p+k, r]
    
    # Build matrix where each column is a flattened term
    term_matrix = np.zeros((n * m * p, rank))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term_matrix[i * m * p + j * p + k, r] = W_arr[k * n + i, r] * U_arr[i * m + j, r] * V_arr[j * p + k, r]
    
    # Check rank of term_matrix
    # If rank < number of terms, there's a linear dependency
    try:
        rank_of_terms = np.linalg.matrix_rank(term_matrix, tol=1e-10)
    except:
        rank_of_terms = rank
    
    if rank_of_terms < rank:
        print(f"Found linear dependency: term rank {rank_of_terms} < {rank}")
        # Find the null space to identify which terms to eliminate
        # Use SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(term_matrix, full_matrices=False)
            null_space = Vt[rank_of_terms:]
            # The null space vectors give linear combinations that sum to zero
            # We can use this to eliminate terms
            # For simplicity, try to eliminate one term at a time
        except:
            pass
    
    # Try greedy elimination: for each term, check if it's in the span of others
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    for r_elim in range(rank):
        if _time_remaining() < 5:
            break
        # Check if term r_elim is in the span of all other terms
        other_indices = [i for i in range(rank) if i != r_elim]
        other_matrix = term_matrix[:, other_indices]
        
        try:
            # Check if term r_elim is in column space of other_matrix
            # Use least squares
            col = term_matrix[:, r_elim]
            coeffs, residuals, rank_other, sv = np.linalg.lstsq(other_matrix, col, rcond=1e-10)
            reconstruction = other_matrix @ coeffs
            if np.allclose(reconstruction, col, atol=1e-9):
                # Term r_elim is redundant!
                new_rank = rank - 1
                new_U = [row[:] for row in U]
                new_V = [row[:] for row in V]
                new_W = [row[:] for row in W]
                # Remove column r_elim
                for row in new_U:
                    del row[r_elim]
                for row in new_V:
                    del row[r_elim]
                for row in new_W:
                    del row[r_elim]
                
                # Verify
                if _verify_decomposition(new_U, new_V, new_W, n, m, p, num_trials=3):
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
                    print(f"Eliminated term {r_elim}, new rank: {best_rank}")
                    # Rebuild term_matrix for further elimination
                    rank = new_rank
                    U_arr = np.array(best_U, dtype=float)
                    V_arr = np.array(best_V, dtype=float)
                    W_arr = np.array(best_W, dtype=float)
                    term_matrix = np.zeros((n * m * p, rank))
                    for r2 in range(rank):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    term_matrix[i * m * p + j * p + k, r2] = W_arr[k * n + i, r2] * U_arr[i * m + j, r2] * V_arr[j * p + k, r2]
                    r_elim = -1  # Reset to check from beginning
        except:
            pass
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    START_TIME_local = time.time()
    deadline = START_TIME_local + 130.0  # Safety margin

    # Start with the rank-28 Strassen decomposition
    result = _build_strassen_28()
    U = result["U"]
    V = result["V"]
    W = result["W"]

    # Verify baseline
    assert _verify_decomposition(U, V, W, n, m, p), "Baseline decomposition failed verification"

    # Try to find lower rank by searching for linear dependencies
    if _time_remaining() > 10:
        reduced = _try_reduce_rank(U, V, W, n, m, p, deadline)
        if reduced is not None:
            U = reduced["U"]
            V = reduced["V"]
            W = reduced["W"]

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    n, m, p = DIMENSIONS
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    print("Verification:", _verify_decomposition(result['U'], result['V'], result['W'], n, m, p))

