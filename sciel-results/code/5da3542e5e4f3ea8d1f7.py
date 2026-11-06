"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a 32x32 unit cell using iterative local search with a fast
    plane-wave proxy model. Optimizes binary pixels via bit-flip moves
    to maximize the TM band gap between bands 4 and 5.
    """
    start_time = time.time()
    deadline = 180  # Safety margin before 230s kill

    size = problem["pixels"]
    N = size
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    # Initial design: ring pattern (dielectric ring with air center)
    x = np.arange(N) - N/2.0 + 0.5
    y = np.arange(N) - N/2.0 + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    
    inner_r = 6.0
    outer_r = 12.0
    cell_arr = ((r >= inner_r) & (r <= outer_r)).astype(np.uint8)
    cell_arr = apply_d4_symmetry(cell_arr)
    
    def compute_proxy_gap(cell):
        """Fast proxy: estimate gap using diagonal approximation of PW matrix."""
        eps = eps_low + (eps_high - eps_low) * cell.astype(float)
        
        # Compute Fourier coefficients of epsilon
        Nf = 5  # Number of G-vectors in each direction
        G_vectors = [(m, n) for m in range(-Nf, Nf+1) for n in range(-Nf, Nf+1)]
        dim = len(G_vectors)
        
        # Precompute phase factors
        xs = np.arange(N) - N/2.0 + 0.5
        phase_cache = {}
        for m in range(-Nf, Nf+1):
            phase_cache[m] = np.exp(-2j * np.pi * m * xs / N)
        
        eps_f = np.zeros((dim, dim), dtype=complex)
        idx_map = {}
        for i, (m, n) in enumerate(G_vectors):
            idx_map[(m, n)] = i
        
        # Compute Fourier coefficients efficiently
        for i, (m, n) in enumerate(G_vectors):
            px = phase_cache[m]
            py = phase_cache[n]
            # eps_f[i, j] = (1/N^2) sum eps(x,y) exp(-2pi i (m*x + n*y)/N)
            # This is the 2D DFT of eps
            coeff = np.sum(eps * np.outer(px, py)) / (N*N)
            eps_f[i, i] = coeff  # Diagonal approximation for speed
        
        # K-points in irreducible BZ
        kpts = [(0, 0), (np.pi/4, 0), (np.pi/2, 0), (np.pi/2, np.pi/4), (np.pi/2, np.pi/2),
                (0, np.pi/4), (np.pi/4, np.pi/4), (0, np.pi/2), (np.pi/4, np.pi/2)]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for (kx, ky) in kpts:
            eigvals = np.zeros(dim)
            for i, (m, n) in enumerate(G_vectors):
                kx_i = kx + 2*np.pi*m
                ky_i = ky + 2*np.pi*n
                eps_diag = np.real(eps_f[i, i])
                if eps_diag > 0.01:
                    eigvals[i] = (kx_i**2 + ky_i**2) / eps_diag
                else:
                    eigvals[i] = (kx_i**2 + ky_i**2)
            
            eigvals = np.sort(np.real(eigvals))
            if len(eigvals) >= 5:
                max_omega4 = max(max_omega4, eigvals[3])
                min_omega5 = min(min_omega5, eigvals[4])
        
        if min_omega5 <= max_omega4 or max_omega4 <= 0:
            return -1.0
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return 0.0
        return 2.0 * (min_omega5 - max_omega4) / midgap
    
    best_cell = cell_arr.copy()
    best_score = compute_proxy_gap(best_cell)
    
    # Local search with multiple restarts
    rng = np.random.RandomState(42)
    n_restarts = 3
    max_iters_per_restart = 2000
    
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        # For first restart, use current best; for others, random perturbation
        if restart == 0:
            current = best_cell.copy()
        else:
            current = best_cell.copy()
            # Perturb: flip some random pixels
            for _ in range(20):
                i, j = rng.randint(0, N), rng.randint(0, N)
                current[i, j] = 1 - current[i, j]
            current = apply_d4_symmetry(current)
        
        cur_score = compute_proxy_gap(current)
        
        for it in range(max_iters_per_restart):
            if time.time() - start_time > deadline:
                break
            
            trial = current.copy()
            i, j = rng.randint(0, N), rng.randint(0, N)
            trial[i, j] = 1 - trial[i, j]
            trial = apply_d4_symmetry(trial)
            
            score = compute_proxy_gap(trial)
            
            if score > cur_score:
                current = trial
                cur_score = score
            elif rng.random() < 0.1:
                # Accept worse moves occasionally (simulated annealing lite)
                current = trial
                cur_score = score
        
        if cur_score > best_score:
            best_cell = current
            best_score = cur_score
    
    cell = [[int(best_cell[i, j]) for j in range(N)] for i in range(N)]
    return cell

def apply_d4_symmetry(cell):
    """Apply D4 symmetry by averaging over all 8 operations."""
    sym = cell.astype(float).copy()
    sym += np.rot90(cell, 1).astype(float)
    sym += np.rot90(cell, 2).astype(float)
    sym += np.rot90(cell, 3).astype(float)
    sym += np.fliplr(cell).astype(float)
    sym += np.flipud(cell).astype(float)
    sym += np.fliplr(np.flipud(cell)).astype(float)
    sym += cell.T.astype(float)
    sym_avg = sym / 8.0
    return (sym_avg > 0.5).astype(np.uint8)
# EVOLVE-BLOCK-END
