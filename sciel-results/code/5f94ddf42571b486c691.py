import numpy as np
import time

def design(problem):
    start_time = time.time()
    deadline = start_time + 215.0

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # --- Symmetry setup ---
    # Pixel (i,j) must equal (size-1-i, size-1-j) [center symmetry]
    # and (i,j) must equal (j,i) [transpose]
    # So the orbit of (i,j) under C4v about center:
    #   {(i,j), (size-1-i, size-1-j), (j,i), (size-1-j, size-1-i)}

    def orbit(i, j):
        s = size
        return [(i, j), (s-1-i, s-1-j), (j, i), (s-1-j, s-1-i)]

    # Build unique orbits
    orbit_map = {}
    orbits = []
    for i in range(size):
        for j in range(size):
            key = (min(i,j), max(i,j)) if i <= j else (min(j,i), max(j,i))
            # Actually we need to be more careful; use the full orbit set as key
            pass

    # Better: iterate and assign orbit IDs
    orbit_id = np.full((size, size), -1, dtype=int)
    orbits = []
    for i in range(size):
        for j in range(size):
            if orbit_id[i, j] == -1:
                o = orbit(i, j)
                oid = len(orbits)
                orbits.append(o)
                for (u, v) in o:
                    orbit_id[u, v] = oid
    num_orbits = len(orbits)

    # --- Fast TM band solver ---
    # 289 PWs: G = (gx, gy) with gx, gy in [-8, 8]
    # Use circular cutoff: gx^2 + gy^2 <= 64
    N_pw = 8
    G_list = []
    for gx in range(-N_pw, N_pw+1):
        for gy in range(-N_pw, N_pw+1):
            if gx*gx + gy*gy <= N_pw*N_pw:
                G_list.append((gx, gy))
    G_arr = np.array(G_list, dtype=float)
    num_pw = len(G_arr)

    # k-points: 30 points covering the 1/8 Wigner-Seitz cell
    # Triangle: (0,0) -> (0.5,0) -> (0.5,0.5)
    # Sample along edges and interior
    k_points = []
    # Edge 1: Gamma to M: (t, 0), t in [0, 0.5]
    for t in np.linspace(0, 0.5, 6):
        k_points.append((t, 0))
    # Edge 2: M to X: (0.5, t), t in [0, 0.5]
    for t in np.linspace(0.5, 0.0, 6):
        k_points.append((0.5, t))
    # Edge 3: X to Gamma: (t, t), t in [0.5, 0]
    for t in np.linspace(0.5, 0.0, 6):
        k_points.append((t, t))
    # Interior points
    for (kx, ky) in [(0.2, 0.1), (0.3, 0.2), (0.25, 0.25), (0.4, 0.1),
                     (0.15, 0.05), (0.35, 0.35), (0.45, 0.05), (0.1, 0.2),
                     (0.3, 0.15), (0.4, 0.25)]:
        k_points.append((kx, ky))

    # Deduplicate
    k_points = list(set(k_points))
    # Trim to ~30
    if len(k_points) > 30:
        k_points = k_points[:30]

    # Precompute |k+G|^2 for each k and each G
    kG_sq = np.zeros((len(k_points), num_pw), dtype=float)
    for ki, (kx, ky) in enumerate(k_points):
        k_plus_G = G_arr + np.array([kx, ky])
        kG_sq[ki] = np.sum(k_plus_G**2, axis=1)

    def build_eps_inv(eps_grid):
        """Compute Fourier coefficients of 1/eps from the pixel grid."""
        inv = 1.0 / eps_grid
        # FFT
        fft_inv = np.fft.fft2(inv) / (size * size)
        # Build coefficient array for differences in [-16, 16]
        # coeff[dx, dy] for dx, dy in [-2*N, 2*N]
        c = np.zeros((2*N_pw*2+1, 2*N_pw*2+1), dtype=complex)
        offset = 2*N_pw
        for dx in range(-2*N_pw, 2*N_pw+1):
            for dy in range(-2*N_pw, 2*N_pw+1):
                ix = dx % size
                iy = dy % size
                c[offset+dx, offset+dy] = fft_inv[ix, iy]
        return c

    def evaluate(eps_grid):
        """Compute gap-midgap ratio for TM bands 4 and 5."""
        c = build_eps_inv(eps_grid)
        offset = 2*N_pw

        # Precompute the (1/eps) convolution matrix once
        # M_conv[i,j] = c[G_i - G_j]
        M_conv = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                dx = int(G_arr[i, 0] - G_arr[j, 0])
                dy = int(G_arr[i, 1] - G_arr[j, 1])
                M_conv[i, j] = c[offset+dx, offset+dy]

        max_omega4 = 0.0
        min_omega5 = np.inf

        for ki in range(len(k_points)):
            # H[i,j] = M_conv[i,j] * kG_sq[ki,i]
            H = M_conv * kG_sq[ki, :].reshape(-1, 1)
            # Solve eigenvalue problem
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                eigvals = np.linalg.eigvals(H).real
            eigvals = np.sort(np.real(eigvals))

            # Filter out spurious negative eigenvalues
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) < 5:
                continue

            omega4 = np.sqrt(eigvals[3])
            omega5 = np.sqrt(eigvals[4])

            if omega4 > max_omega4:
                max_omega4 = omega4
            if omega5 < min_omega5:
                min_omega5 = omega5

        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0

        gap = min_omega5 - max_omega4
        midgap = 0.5 * (min_omega5 + max_omega4)
        if midgap <= 0:
            return -1.0
        return 2.0 * gap / midgap

    # --- Initial cell: dielectric rod (common starting point for photonic crystals) ---
    cell = np.zeros((size, size), dtype=int)
    center = (size - 1) / 2.0
    radius = 7
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            if dx*dx + dy*dy < radius*radius:
                cell[i, j] = 1

    # Enforce symmetry
    for oid, o in enumerate(orbits):
        vals = [cell[u, v] for (u, v) in o]
        val = int(np.mean(vals))
        for (u, v) in o:
            cell[u, v] = val

    eps_grid = np.where(cell == 1, eps_high, eps_low)
    best_score = evaluate(eps_grid)
    best_cell = cell.copy()

    # --- Local search with multi-flip moves ---
    T = 0.5
    T_min = 0.005
    cooling = 0.998

    while time.time() < deadline:
        # Pick a random orbit to flip
        oid = np.random.randint(num_orbits)
        o = orbits[oid]

        # Flip all pixels in the orbit
        for (u, v) in o:
            cell[u, v] = 1 - cell[u, v]

        eps_grid = np.where(cell == 1, eps_high, eps_low)
        new_score = evaluate(eps_grid)

        if new_score > best_score:
            best_score = new_score
            best_cell = cell.copy()
        else:
            delta = new_score - best_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() > prob:
                    # Revert
                    for (u, v) in o:
                        cell[u, v] = 1 - cell[u, v]

        T *= cooling
        if T < T_min:
            T = T_min
            if np.random.random() < 0.05:
                T = 0.3  # Reheat

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    return result
