"""Design a symmetric 32x32-pixel photonic crystal unit cell for maximum TM 4-5 band gap."""
import time
import math
import random

def design(problem):
    """
    Design a 32x32 pixel unit cell with 4-fold symmetry that maximizes
    the TM band gap between bands 4 and 5.
    
    Uses a parametric family of four-rod lattices with variable rod
    positions, radii, and shapes, optimized via simulated annealing
    with a fast surrogate score based on Fourier coefficients.
    """
    size = problem["pixels"]  # 32
    eps_high = problem["eps_high"]  # 12
    eps_low = problem["eps_low"]  # 1
    
    start_time = time.time()
    time_limit = 190.0  # Safety margin before 200s limit
    
    def make_cell(cx, cy, rx, ry, shape):
        """
        Create a symmetric 32x32 cell with 4 rods.
        cx, cy: center of one rod (in [0, 0.5) x [0, 0.5) quadrant)
        rx, ry: half-widths (elliptical)
        shape: 0=circle, 1=square, 2=diamond
        The other 3 rods are placed by 4-fold symmetry.
        """
        cell = [[0] * size for _ in range(size)]
        
        # The four rod centers in the full cell
        centers = [
            (cx, cy),
            (1 - cx, cy),
            (cx, 1 - cy),
            (1 - cx, 1 - cy)
        ]
        
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                is_diel = False
                for (px, py) in centers:
                    dx = x - px
                    dy = y - py
                    # Normalize by half-widths for ellipse
                    nx = abs(dx) / rx if rx > 0 else 999
                    ny = abs(dy) / ry if ry > 0 else 999
                    if shape == 0:  # circle
                        if nx*nx + ny*ny < 1.0:
                            is_diel = True
                            break
                    elif shape == 1:  # square (axis-aligned)
                        if nx < 1.0 and ny < 1.0:
                            is_diel = True
                            break
                    elif shape == 2:  # diamond (rotated square)
                        if nx + ny < 1.0:
                            is_diel = True
                            break
                cell[i][j] = 1 if is_diel else 0
        
        # Ensure exact symmetry: cell == cell[::-1] row-wise and col-wise, and transpose
        # Row-reversal symmetry: cell[i][j] == cell[size-1-i][j]
        # Col-reversal symmetry: cell[i][j] == cell[i][size-1-j]
        # Transpose symmetry: cell[i][j] == cell[j][i]
        # We enforce by averaging (majority) over the symmetry orbit
        for i in range(size):
            for j in range(size):
                # Orbit under D4: (i,j), (size-1-i,j), (i,size-1-j), (size-1-i,size-1-j),
                #                 (j,i), (size-1-j,i), (j,size-1-i), (size-1-j,size-1-i)
                orbit = [
                    (i, j),
                    (size-1-i, j),
                    (i, size-1-j),
                    (size-1-i, size-1-j),
                    (j, i),
                    (size-1-j, i),
                    (j, size-1-i),
                    (size-1-j, size-1-i)
                ]
                # Deduplicate
                seen = set()
                unique_orbit = []
                for (oi, oj) in orbit:
                    if (oi, oj) not in seen:
                        seen.add((oi, oj))
                        unique_orbit.append((oi, oj))
                
                count = 0
                for (oi, oj) in unique_orbit:
                    count += cell[oi][oj]
                val = 1 if count >= len(unique_orbit) // 2 else 0
                for (oi, oj) in unique_orbit:
                    cell[oi][oj] = val
        
        return cell
    
    def fourier_score(cell):
        """
        Fast surrogate: compute key Fourier coefficients of 1/epsilon
        that couple bands 4 and 5.
        
        The 4-5 TM gap is driven by Fourier components that connect
        G-vectors differing by ~ (1,1) in reciprocal space.
        We compute |eps_inv(-1,-1)|, |eps_inv(-2,0)|, |eps_inv(0,-2)|,
        |eps_inv(-1,1)|, |eps_inv(-1,-1)| and combine them.
        """
        # Compute 1/epsilon for each pixel
        N = size
        eps_inv = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                eps = eps_high if cell[i][j] == 1 else eps_low
                eps_inv[i][j] = 1.0 / eps
        
        # Compute DFT for specific (m,n) pairs
        def dft_coeff(m, n):
            s_re = 0.0
            s_im = 0.0
            for i in range(N):
                for j in range(N):
                    angle = 2.0 * math.pi * (m * i + n * j) / N
                    c = eps_inv[i][j]
                    s_re += c * math.cos(angle)
                    s_im -= c * math.sin(angle)
            factor = 1.0 / (N * N)
            return factor * math.sqrt(s_re*s_re + s_im*s_im)
        
        # Key Fourier coefficients for 4-5 TM gap
        c_11 = dft_coeff(1, 1)
        c_20 = dft_coeff(2, 0)
        c_02 = dft_coeff(0, 2)
        c_1m1 = dft_coeff(1, -1)
        c_2m2 = dft_coeff(2, -2)
        c_30 = dft_coeff(3, 0)
        c_03 = dft_coeff(0, 3)
        
        # Combine: emphasize the (1,1) component which is critical
        # Also include higher harmonics
        score = (c_11 ** 2 + 0.5 * c_20 ** 2 + 0.5 * c_02 ** 2 +
                 0.3 * c_1m1 ** 2 + 0.2 * c_2m2 ** 2 +
                 0.1 * c_30 ** 2 + 0.1 * c_03 ** 2)
        
        return score
    
    # Initial parameters: 4 rods at (0.25, 0.25) with radius ~0.15
    cx = 0.25
    cy = 0.25
    rx = 0.15
    ry = 0.15
    shape = 0  # circle
    
    current_cell = make_cell(cx, cy, rx, ry, shape)
    current_score = fourier_score(current_cell)
    best_cell = [row[:] for row in current_cell]
    best_score = current_score
    
    # Simulated annealing over parameters
    T = 1.0
    cooling = 0.995
    random.seed(42)
    
    iter_count = 0
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Perturb parameters
        new_cx = cx + random.gauss(0, 0.02)
        new_cy = cy + random.gauss(0, 0.02)
        new_rx = max(0.05, min(0.45, rx + random.gauss(0, 0.03)))
        new_ry = max(0.05, min(0.45, ry + random.gauss(0, 0.03)))
        new_shape = shape
        if random.random() < 0.1:
            new_shape = random.randint(0, 2)
        
        # Keep cx, cy in reasonable range
        new_cx = max(0.05, min(0.45, new_cx))
        new_cy = max(0.05, min(0.45, new_cy))
        
        new_cell = make_cell(new_cx, new_cy, new_rx, new_ry, new_shape)
        new_score = fourier_score(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-6)):
            cx, cy, rx, ry, shape = new_cx, new_cy, new_rx, new_ry, new_shape
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Periodic restart: occasionally try a random initial point
        if iter_count % 500 == 0 and time.time() - start_time < time_limit - 30:
            rcx = random.uniform(0.15, 0.35)
            rcy = random.uniform(0.15, 0.35)
            rrx = random.uniform(0.10, 0.30)
            rry = random.uniform(0.10, 0.30)
            rshape = random.randint(0, 2)
            rcell = make_cell(rcx, rcy, rrx, rry, rshape)
            rscore = fourier_score(rcell)
            if rscore > best_score:
                best_score = rscore
                best_cell = [row[:] for row in rcell]
                cx, cy, rx, ry, shape = rcx, rcy, rrx, rry, rshape
                current_cell = rcell
                current_score = rscore
                T = 0.5  # Reheat
    
    # Final check
    if time.time() - start_time < time_limit - 5:
        # Try a few more specific configurations
        test_configs = [
            (0.25, 0.25, 0.15, 0.15, 0),   # original
            (0.25, 0.25, 0.20, 0.20, 0),   # larger circles
            (0.25, 0.25, 0.15, 0.15, 1),   # squares
            (0.25, 0.25, 0.15, 0.15, 2),   # diamonds
            (0.20, 0.20, 0.15, 0.15, 0),   # shifted
            (0.30, 0.30, 0.15, 0.15, 0),   # shifted other way
            (0.25, 0.25, 0.25, 0.10, 0),   # elongated x
            (0.25, 0.25, 0.10, 0.25, 0),   # elongated y
            (0.25, 0.25, 0.20, 0.20, 1),   # larger squares
            (0.22, 0.22, 0.18, 0.18, 0),   # compact
        ]
        for (tcx, tcy, trx, try_, tshape) in test_configs:
            tcell = make_cell(tcx, tcy, trx, try_, tshape)
            tscore = fourier_score(tcell)
            if tscore > best_score:
                best_score = tscore
                best_cell = [row[:] for row in tcell]
    
    # Ensure the returned cell has exact symmetry
    result = [row[:] for row in best_cell]
    N = size
    for i in range(N):
        for j in range(N):
            orbit = [
                (i, j), (N-1-i, j), (i, N-1-j), (N-1-i, N-1-j),
                (j, i), (N-1-j, i), (j, N-1-i), (N-1-j, N-1-i)
            ]
            seen = set()
            unique = []
            for (oi, oj) in orbit:
                if (oi, oj) not in seen:
                    seen.add((oi, oj))
                    unique.append((oi, oj))
            count = sum(result[oi][oj] for (oi, oj) in unique)
            val = 1 if count >= len(unique) // 2 else 0
            for (oi, oj) in unique:
                result[oi][oj] = val
    
    return result
