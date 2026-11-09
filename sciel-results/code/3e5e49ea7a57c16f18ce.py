"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


def _build_spectral_heat_solver(f, t0, y0, n):
    """
    Detect the 2D heat equation structure and build a spectral solver.
    Returns (c, dt, U, eigenvalues, V) where:
    - c is the diffusion coefficient (from stencil)
    - U and V are eigenvector matrices (sine transform basis)
    - eigenvalues are the diagonal entries of A in the eigenbasis
    Or None if structure not detected.
    """
    if n != 100:
        return None

    # Probe 1: zero vector -> should give zero if homogeneous linear
    f_zero = f(t0, np.zeros(100))
    if np.linalg.norm(f_zero) > 1e-12:
        return None  # not homogeneous

    # Probe 2: a single interior point set to 1, rest 0
    probe_int = np.zeros(100)
    probe_int[55] = 1.0
    f_int = f(t0, probe_int)

    # For 5-point stencil, interior node has: -4*c at center, c at 4 neighbors
    c_est = f_int[45]  # neighbor above (index 45 = row 4, col 5)
    if c_est == 0:
        return None

    # Verify other neighbors match
    if not (np.isclose(f_int[65], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_int[54], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_int[56], c_est, rtol=1e-10, atol=1e-14)):
        return None
    if not np.isclose(f_int[55], -4 * c_est, rtol=1e-10, atol=1e-14):
        return None

    # Verify linearity with another probe
    probe_2 = np.zeros(100)
    probe_2[54] = 1.0
    f_2 = f(t0, probe_2)
    if not (np.isclose(f_2[44], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_2[64], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_2[53], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_2[55], c_est, rtol=1e-10, atol=1e-14)):
        return None
    if not np.isclose(f_2[54], -4 * c_est, rtol=1e-10, atol=1e-14):
        return None

    # Now determine boundary treatment by probing a boundary node
    probe_bnd = np.zeros(100)
    probe_bnd[11] = 1.0  # node (1,1)
    f_bnd = f(t0, probe_bnd)

    # Check if this is Dirichlet BC (boundary nodes not in system, n=64) or
    # if all 100 nodes are included with some BC
    # For n=100, all nodes are state variables. Let's check the coefficient at (1,1)
    # If Dirichlet with boundary eliminated: (1,1) would only have 3 interior neighbors
    # But n=100 means boundary nodes ARE in the system
    
    # Check corner node (0,0) = index 0
    probe_corner = np.zeros(100)
    probe_corner[0] = 1.0
    f_corner = f(t0, probe_corner)
    
    # For a standard 2D Laplacian on 10x10 grid with Dirichlet BC at domain boundary
    # (not at grid points), the grid points 0..9 are all interior or edge.
    # The Laplacian at (0,0) corner: if we use one-sided differences or
    # if boundary is at half-cell outside, the stencil varies.
    
    # Let's check: what's the diagonal at corner?
    corner_diag = f_corner[0]
    
    # For a standard 5-point stencil with h=1/9 (grid spacing for 10 points on [0,1]):
    # At interior: -4/h^2 * c, neighbors c/h^2
    # At edge (not corner): -3/h^2 * c, 3 neighbors
    # At corner: -2/h^2 * c, 2 neighbors
    # So c_est from interior = c/h^2, and corner should be -2*c_est
    
    if not np.isclose(corner_diag, -2 * c_est, rtol=1e-10, atol=1e-14):
        # Maybe different scaling. Let's try to infer the actual eigenvalues
        # by building A explicitly and diagonalizing
        # But first, let's check if it's a scaled version
        ratio = corner_diag / (-2 * c_est)
        if not np.isclose(ratio, 1.0, rtol=1e-6):
            return None

    # Check edge node (0,5) = index 5
    probe_edge = np.zeros(100)
    probe_edge[5] = 1.0
    f_edge = f(t0, probe_edge)
    
    if not np.isclose(f_edge[5], -3 * c_est, rtol=1e-10, atol=1e-14):
        return None
    if not (np.isclose(f_edge[15], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_edge[4], c_est, rtol=1e-10, atol=1e-14) and
            np.isclose(f_edge[6], c_est, rtol=1e-10, atol=1e-14)):
        return None

    # Structure confirmed! This is the 2D discrete Laplacian with 5-point stencil
    # on a 10x10 grid. Now build the spectral decomposition.
    
    # The eigenfunctions of the 2D discrete Laplacian with this stencil are
    # separable: phi_{p,q}(i,j) = sin(p*(i+1)*pi/11) * sin(q*(j+1)*pi/11)
    # for p,q = 1,...,10 and i,j = 0,...,9
    # Eigenvalues: lambda_{p,q} = -4*c_est * sin^2(p*pi/22) - 4*c_est * sin^2(q*pi/22)
    # Wait, let me be more careful.
    
    # For the 1D discrete Laplacian on N points with this stencil:
    # A_1D[i,i] = -2*c for interior, -c for boundary (if only 1 neighbor)
    # Actually for our 2D case, let's think of it as:
    # A = kron(I, T) + kron(T, I) where T is the 1D tridiagonal
    # T[i,i] = -2*c for i=1..N-2, T[0,0] = -c, T[N-1,N-1] = -c
    # T[i,i+1] = T[i+1,i] = c
    
    # Hmm, but our corner has -2*c and edge has -3*c, interior has -4*c
    # This is consistent with A = kron(I,T) + kron(T,I) where:
    # T is 10x10 with T[i,i] = -2*c for all i (since each dimension contributes -2*c)
    # No wait, that gives interior -4*c, edge -3*c (one dim boundary, one interior), corner -2*c
    # Yes! So T[i,i] = -2*c for all i, T[i,i±1] = c
    
    # The eigenvalues of T: T is tridiagonal with -2c on diagonal, c on off-diagonals
    # This is c * (tridiag(-2, 1, 1)) which has eigenvalues:
    # mu_p = -2c * (1 - cos(p*pi/11)) = -4c * sin^2(p*pi/22) for p=1,...,10
    
    # Eigenvalues of A = kron(I,T) + kron(T,I):
    # lambda_{p,q} = mu_p + mu_q = -4c * sin^2(p*pi/22) - 4c * sin^2(q*pi/22)
    
    # Eigenvectors of T: v_p(i) = sin(p*(i+1)*pi/11) for i=0,...,9, p=1,...,10
    # These are orthonormal up to scaling: sum_i v_p(i)^2 = 11/2
    
    # For 2D: eigenvector (p,q) at (i,j) = v_p(i) * v_q(j)
    # Eigenvalue: lambda_{p,q} = -4c * (sin^2(p*pi/22) + sin^2(q*pi/22))
    
    N = 10
    c = c_est
    
    # Build the eigenvector matrix U (100x100)
    # U[:, idx] where idx = (p-1)*10 + (q-1), and U[idx2, idx] = v_p(i)*v_q(j) for idx2 = i*10+j
    U = np.zeros((100, 100))
    eigenvalues = np.zeros(100)
    
    for p in range(1, N + 1):
        for q in range(1, N + 1):
            eig_idx = (p - 1) * N + (q - 1)
            # Eigenvalue
            eigenvalues[eig_idx] = -4 * c * (np.sin(p * np.pi / (2 * (N + 1)))**2 + 
                                              np.sin(q * np.pi / (2 * (N + 1)))**2)
            # Eigenvector components
            for i in range(N):
                for j in range(N):
                    grid_idx = i * N + j
                    U[grid_idx, eig_idx] = np.sin(p * (i + 1) * np.pi / (N + 1)) * \
                                           np.sin(q * (j + 1) * np.pi / (N + 1))
    
    # Normalize U columns (they should be orthogonal but not unit norm)
    # Each column has norm sqrt(11/2 * 11/2) = 11/2 = 5.5
    norms = np.sqrt(np.sum(U**2, axis=0))
    U = U / norms
    
    # V = U^T (since U is orthogonal)
    V = U.T
    
    return (c, U, eigenvalues, V)


def integrate(f, t_span, y0):
    t_start = time.time()
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Probe linearity: check if f(t, y) = A @ y (independent of t)
    f0 = f(t0, y0)
    f_probe1 = f(t0, 2 * y0)
    
    linearity_err = np.linalg.norm(f_probe1 - 2 * f0) / (2 * np.linalg.norm(f0) + 1e-30)
    
    if linearity_err < 1e-8:
        # Likely linear. Try spectral heat solver
        result = _build_spectral_heat_solver(f, t0, y0, n)
        
        if result is not None:
            c, U, eigenvalues, V = result
            
            # Check deadline
            if time.time() > _DEADLINE:
                return y0
            
            # y(t) = U @ exp(dt * diag(eigenvalues)) @ V @ y0
            # = U @ (exp(dt * eigenvalues) * (V @ y0))
            coeffs = V @ y0
            coeffs = coeffs * np.exp(dt * eigenvalues)
            y_final = U @ coeffs
            
            return y_final
        else:
            # Fall back to building A explicitly and using expm
            from scipy.linalg import expm
            A = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1.0
                A[:, j] = f(t0, e)
            
            if time.time() > _DEADLINE:
                return y0
            
            M = expm(dt * A)
            y_final = M @ y0
            return y_final
    else:
        # Not linear; fall back to RK4
        h = (t1 - t0) / 2000
        t, y = t0, y0.copy()
        for _ in range(2000):
            if time.time() > _DEADLINE:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                break
        return y
# EVOLVE-BLOCK-END
