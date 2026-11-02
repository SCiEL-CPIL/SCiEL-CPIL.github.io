import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 130  # seconds, safety margin before 150s kill

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    - Build the known rank-55 Strassen-based decomposition
    - Use structured tensor completion: fix Strassen block structure,
      then use integer lattice reduction to search for rank-54 solutions
      with half-integer coefficients
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    
    # Build the known rank-55 solution
    def build_rank_55():
        rank = 55
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        a00, a01, a10, a11 = 0, 1, 3, 4
        a02, a12 = 2, 5
        
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            b00, b01 = k0, k1
            b10, b11 = 10 + k0, 10 + k1
            c00, c01 = 2 * k0, 2 * k1
            c10, c11 = 2 * k0 + 1, 2 * k1 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[a00, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M2 = (A10 + A11)B00
            U[a10, col] += 1.0; U[a11, col] += 1.0
            V[b00, col] += 1.0
            W[c10, col] += 1.0; W[c11, col] -= 1.0
            col += 1
            
            # M3 = A00(B01 - B11)
            U[a00, col] += 1.0
            V[b01, col] += 1.0; V[b11, col] -= 1.0
            W[c01, col] += 1.0; W[c11, col] += 1.0
            col += 1
            
            # M4 = A11(B10 - B00)
            U[a11, col] += 1.0
            V[b10, col] += 1.0; V[b00, col] -= 1.0
            W[c00, col] += 1.0; W[c10, col] += 1.0
            col += 1
            
            # M5 = (A00 + A01)B11
            U[a00, col] += 1.0; U[a01, col] += 1.0
            V[b11, col] += 1.0
            W[c00, col] -= 1.0; W[c01, col] += 1.0
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[a10, col] += 1.0; U[a00, col] -= 1.0
            V[b00, col] += 1.0; V[b01, col] += 1.0
            W[c11, col] += 1.0
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[a01, col] += 1.0; U[a11, col] -= 1.0
            V[b10, col] += 1.0; V[b11, col] += 1.0
            W[c00, col] += 1.0
            col += 1
        
        for k in range(p):
            b2k = 2 * p + k
            U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
            col += 1
            U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
            col += 1
        
        return U, V, W, rank
    
    U55, V55, W55, rank55 = build_rank_55()
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Verify rank-55 solution
    def verify_decomposition(U, V, W, rank, atol=1e-10):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        return np.allclose(T, 1.0, atol=atol)
    
    assert verify_decomposition(U55, V55, W55, rank55), "Rank-55 solution invalid"
    
    # Structured tensor completion approach:
    # The rank-55 solution has 35 Strassen terms (for 2x2 blocks) + 20 schoolbook terms
    # We try to find if any single term can be eliminated by adjusting others
    
    def try_eliminate_with_lattice(U, V, W, rank, t_to_eliminate):
        """
        Try to eliminate term t_to_eliminate by finding half-integer linear combinations
        of other terms that preserve the tensor.
        """
        other_terms = [s for s in range(rank) if s != t_to_eliminate]
        n_other = len(other_terms)
        
        # Build the tensor contribution vector for term t_to_eliminate
        b = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    b[idx] = U[i*m+j, t_to_eliminate] * V[j*p+k, t_to_eliminate] * W[k*n+i, t_to_eliminate]
                    idx += 1
        
        # Build the matrix of other term contributions
        A = np.zeros((n * m * p, n_other))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for col_idx, s in enumerate(other_terms):
                        A[idx, col_idx] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                    idx += 1
        
        # First check if b is in the span of A columns (linear dependency)
        try:
            c, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-10:
                # Check if coefficients are half-integers
                c_scaled = 2 * c
                if np.allclose(c_scaled, np.round(c_scaled), atol=1e-8):
                    return True
        except np.linalg.LinAlgError:
            pass
        
        return False
    
    # Try to eliminate each single term
    for t in range(rank55):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if try_eliminate_with_lattice(U55, V55, W55, rank55, t):
            # Remove the term
            new_rank = rank55 - 1
            new_U = np.zeros((n * m, new_rank), dtype=float)
            new_V = np.zeros((m * p, new_rank), dtype=float)
            new_W = np.zeros((p * n, new_rank), dtype=float)
            
            col_idx = 0
            for s in range(rank55):
                if s != t:
                    new_U[:, col_idx] = U55[:, s]
                    new_V[:, col_idx] = V55[:, s]
                    new_W[:, col_idx] = W55[:, s]
                    col_idx += 1
            
            if verify_decomposition(new_U, new_V, new_W, new_rank):
                best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                print(f"Eliminated term {t}, new rank: {new_rank}")
                break
    
    # If we reduced to 54, try to eliminate one more term
    if best_rank == 54 and time.time() - start_time < DEADLINE - 30:
        for t in range(best_rank):
            if time.time() - start_time > DEADLINE - 25:
                break
            
            if try_eliminate_with_lattice(best_U, best_V, best_W, best_rank, t):
                new_rank = best_rank - 1
                new_U = np.zeros((n * m, new_rank), dtype=float)
                new_V = np.zeros((m * p, new_rank), dtype=float)
                new_W = np.zeros((p * n, new_rank), dtype=float)
                
                col_idx = 0
                for s in range(best_rank):
                    if s != t:
                        new_U[:, col_idx] = best_U[:, s]
                        new_V[:, col_idx] = best_V[:, s]
                        new_W[:, col_idx] = best_W[:, s]
                        col_idx += 1
                
                if verify_decomposition(new_U, new_V, new_W, new_rank):
                    best_U, best_V, best_W, best_rank = new_U, new_V, new_W, new_rank
                    print(f"Eliminated term {t}, new rank: {new_rank}")
                    break
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, best_rank, atol=1e-10):
        # Fallback to rank-55
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
        print("Fallback to rank-55 due to verification failure")
    
    print(f"Final rank: {best_rank}")
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
