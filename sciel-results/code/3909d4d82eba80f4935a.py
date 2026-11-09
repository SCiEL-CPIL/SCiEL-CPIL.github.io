import numpy as np
import time

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    deadline = time.time() + 110

    def enforce_symmetry(cell):
        cell = np.array(cell, dtype=float)
        n = cell.shape[0]
        c1 = cell
        c2 = cell[::-1, ::-1]
        c3 = cell[:, ::-1]
        c4 = cell[::-1, :]
        c5 = cell.T
        c6 = cell.T[::-1, ::-1]
        c7 = cell[:, ::-1].T
        c8 = cell[::-1, :].T
        avg = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) / 8.0
        return (avg >= 0.5).astype(int)

    def create_solver(cell, n_pw):
        n = cell.shape[0]
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        F_full = np.fft.fft2(inv_eps) / (n * n)
        
        half = n_pw // 2
        if n_pw % 2 == 0:
            g_idx = np.array(list(range(0, half)) + list(range(-half, 0)))
        else:
            g_idx = np.array(list(range(0, half + 1)) + list(range(-half, 0)))
        N = len(g_idx)
        
        # Precompute F_diff
        F_diff = np.zeros((N, N, N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                for i2 in range(N):
                    for j2 in range(N):
                        F_diff[i, j, i2, j2] = F_full[(g_idx[i] - g_idx[i2]) % n, (g_idx[j] - g_idx[j2]) % n]
        
        Gx = 2 * np.pi * g_idx
        Gy = 2 * np.pi * g_idx
        
        def solve_k(kx, ky):
            k2 = (kx + Gx[:, None])**2 + (ky + Gy[None, :])**2
            A = (F_diff * k2[None, None, :, :]).reshape(N*N, N*N)
            eigvals = np.linalg.eigvalsh(A)
            return eigvals[:n_pw]
        
        return solve_k

    # k-points in irreducible BZ
    k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    for t in np.linspace(0.1, 0.4, 4):
        k_points.append((t, 0.0))
        k_points.append((0.5, t))
        k_points.append((t, t))
    for t1 in [0.1, 0.2, 0.3]:
        for t2 in [0.05, 0.1, 0.15]:
            if t1 + t2 <= 0.5:
                k_points.append((t1, t2))
    k_points = k_points[:25]

    def evaluate_gap(cell, n_pw=11):
        if time.time() > deadline - 20:
            return -1e10
        solver = create_solver(cell, n_pw)
        max_omega4 = 0.0
        min_omega5 = np.inf
        for kx, ky in k_points:
            eigvals = solver(kx, ky)
            if len(eigvals) >= 5:
                max_omega4 = max(max_omega4, np.sqrt(eigvals[3]))
                min_omega5 = min(min_omega5, np.sqrt(eigvals[4]))
        if min_omega5 == np.inf:
            return -1e10
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap <= 0:
            return -1e10
        return 2 * gap / midgap

    def make_checkerboard():
        cell = np.zeros((size, size), dtype=int)
        block = size // 4
        for i in range(size):
            for j in range(size):
                if (i // block + j // block) % 2 == 0:
                    cell[i, j] = 1
        return cell

    def make_anti_rod(r):
        cell = np.ones((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < r*r:
                    cell[i, j] = 0
        return cell

    def make_rod(r):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < r*r:
                    cell[i, j] = 1
        return cell

    # Try seeds
    seeds = [make_checkerboard()]
    for r in [0.15, 0.2, 0.25]:
        seeds.append(make_anti_rod(r))
    for r in [0.15, 0.2, 0.25]:
        seeds.append(make_rod(r))

    best_cell = None
    best_score = -1e10
    for cell in seeds:
        if time.time() > deadline - 50:
            break
        cell = enforce_symmetry(cell)
        score = evaluate_gap(cell, n_pw=11)
        if score > best_score:
            best_score = score
            best_cell = cell.copy()

    if best_cell is None:
        best_cell = enforce_symmetry(make_rod(0.2))

    # Local search
    if time.time() < deadline - 30:
        n = size
        visited = set()
        orbits = []
        for i in range(n):
            for j in range(n):
                if (i, j) in visited:
                    continue
                orbit = set()
                def add(i2, j2):
                    if (i2, j2) not in orbit:
                        orbit.add((i2, j2))
                        add(n-1-i2, n-1-j2)
                        add(i2, n-1-j2)
                        add(n-1-i2, j2)
                        add(j2, i2)
                add(i, j)
                visited.update(orbit)
                orbits.append(orbit)

        for _ in range(3):
            if time.time() > deadline - 25:
                break
            improved = False
            for orbit in orbits:
                if time.time() > deadline - 20:
                    break
                new_cell = best_cell.copy()
                for (i, j) in orbit:
                    new_cell[i, j] = 1 - new_cell[i, j]
                new_cell = enforce_symmetry(new_cell)
                score = evaluate_gap(new_cell, n_pw=11)
                if score > best_score + 1e-6:
                    best_score = score
                    best_cell = new_cell
                    improved = True
            if not improved:
                break

    return best_cell.tolist()
# EVOLVE-BLOCK-END
