"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 4 matrix using linear optimization and rational reconstruction.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8

    # Build the tensor T with trace convention: W[k*n + i, r]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Rational reconstruction using continued fractions
    def rational_reconstruct(x, max_denom=1000):
        """Reconstruct a rational number from a float using continued fractions."""
        if abs(x) < 1e-10:
            return Fraction(0, 1)
        
        # Use Fraction with limit_denominator
        frac = Fraction(x).limit_denominator(max_denom)
        
        # Verify the reconstruction
        if abs(float(frac) - x) < 1e-8:
            return frac
        else:
            # Try with larger denominator
            frac2 = Fraction(x).limit_denominator(max_denom * 10)
            if abs(float(frac2) - x) < 1e-8:
                return frac2
            return Fraction(0, 1)

    def rationalize_matrix(M, max_denom=1000):
        """Convert a matrix of floats to a matrix of Fractions."""
        return [[rational_reconstruct(x, max_denom) for x in row] for row in M]

    def verify_decomposition(U, V, W, T, tol=1e-8):
        """Verify that U, V, W exactly decompose T."""
        # Compute the reconstructed tensor
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(recon - T))
        return error < tol

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = 12, but realistically higher
    # For 2x3 x 3x4, let's try from 14 to 24
    for R in range(14, n * m * p + 1):
        # Multiple random initializations
        for seed in range(20):
            np.random.seed(seed)
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5

            # Iterative ALS (Alternating Least Squares)
            converged = False
            for it in range(300):
                # Fix U and V, solve for W
                # For each c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b
                # Coefficient matrix for W[c,:]: D[a,b,r] = U[a,r] * V[b,r]
                D_uv = np.einsum('ar,br->abr', U, V)  # (nm, mp, R)
                D_uv_flat = D_uv.reshape(nm * mp, R)
                
                for c in range(pn):
                    T_c = T[:, :, c].flatten()
                    W[c, :], _, _, _ = np.linalg.lstsq(D_uv_flat, T_c, rcond=None)

                # Fix V and W, solve for U
                D_vw = np.einsum('br,cr->bcr', V, W)  # (mp, pn, R)
                D_vw_flat = D_vw.reshape(mp * pn, R)
                
                for a in range(nm):
                    T_a = T[a, :, :].flatten()
                    U[a, :], _, _, _ = np.linalg.lstsq(D_vw_flat, T_a, rcond=None)

                # Fix U and W, solve for V
                D_uw = np.einsum('ar,cr->acr', U, W)  # (nm, pn, R)
                D_uw_flat = D_uw.reshape(nm * pn, R)
                
                for b in range(mp):
                    T_b = T[:, b, :].flatten()
                    V[b, :], _, _, _ = np.linalg.lstsq(D_uw_flat, T_b, rcond=None)

                # Check convergence
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.max(np.abs(recon - T))
                if error < 1e-10:
                    converged = True
                    break

            if not converged:
                continue

            # Rationalize the solution
            U_frac = rationalize_matrix(U, max_denom=100)
            V_frac = rationalize_matrix(V, max_denom=100)
            W_frac = rationalize_matrix(W, max_denom=100)

            # Convert back to float for verification
            U_check = np.array([[float(x) for x in row] for row in U_frac], dtype=np.float64)
            V_check = np.array([[float(x) for x in row] for row in V_frac], dtype=np.float64)
            W_check = np.array([[float(x) for x in row] for row in W_frac], dtype=np.float64)

            # Verify the exact decomposition
            if verify_decomposition(U_check, V_check, W_check, T, tol=1e-10):
                return {
                    "U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]
                }

    # Fallback to schoolbook decomposition (rank = n*m*p = 24)
    R = n * m * p
    U = [[0] * R for _ in range(nm)]
    V = [[0] * R for _ in range(mp)]
    W = [[0] * R for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
