import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = start_time + 215.0  # Safety margin
    
    N_PW = 8
    G_range = np.arange(-N_PW, N_PW + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    n_pw = len(G_vectors)
    
    def compute_inv_eps_fourier(cell, G_max=16):
        inv_eps_low = 1.0 / eps_low
        inv_eps_high = 1.0 / eps_high
        pix = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(pix, pix, indexing='ij')
        cell_arr = np.array(cell)
        inv_eps_pix = np.where(cell_arr == 1, inv_eps_high, inv_eps_low)
        
        G_vals = np.arange(-G_max, G_max + 1)
        Gx_ext, Gy_ext = np.meshgrid(G_vals, G_vals, indexing='ij')
        G_ext = np.column_stack((Gx_ext.ravel(), Gy_ext.ravel()))
        
        inv_eps_G_ext = np.zeros(len(G_ext), dtype=np.complex128)
        for idx in range(len(G_ext)):
            gx, gy = G_ext[idx]
            phase = np.exp(-1j * 2 * np.pi * (gx * X + gy * Y))
            inv_eps_G_ext[idx] = np.sum(inv_eps_pix * phase) / (size * size)
        
        G_to_idx = {}
        for idx, (gx, gy) in enumerate(G_ext):
            G_to_idx[(int(gx), int(gy))] = idx
        
        return G_to_idx, inv_eps_G_ext
    
    def generate_k_points():
        k_points = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.25), (0.5, 0.25)]
        for t in np.linspace(0.05, 0.45, 5):
            k_points.append((t, 0.0))
        for t in np.linspace(0.05, 0.45, 5):
            k_points.append((t, t))
        for t in np.linspace(0.05, 0.45, 5):
            k_points.append((0.5, t))
        k_points.extend([(0.3, 0.1), (0.4, 0.2), (0.35, 0.15), (0.2, 0.1),
                         (0.45, 0.3), (0.3, 0.25), (0.4, 0.35), (0.25, 0.1),
                         (0.45, 0.1), (0.35, 0.3)])
        return k_points[:30]
    
    k_points = generate_k_points()
    
    def evaluate_cell(cell):
        G_to_idx, inv_eps_G_ext = compute_inv_eps_fourier(cell)
        
        B = np.zeros((n_pw, n_pw), dtype=np.complex128)
        for i in range(n_pw):
            Gi = G_vectors[i]
            for j in range(n_pw):
                Gj = G_vectors[j]
                diff = (int(Gi[0] - Gj[0]), int(Gi[1] - Gj[1]))
                if diff in G_to_idx:
                    B[i, j] = inv_eps_G_ext[G_to_idx[diff]]
        
        min_omega5 = np.inf
        max_omega4 = 0
        
        for kx, ky in k_points:
            D_diag = (kx + G_vectors[:, 0])**2 + (ky + G_vectors[:, 1])**2
            D_mat = np.diag(D_diag)
            
            try:
                from scipy.linalg import eig
                vals, _ = eig(D_mat, B)
                vals = np.real(vals)
                vals = vals[vals > 0]
                vals = np.sort(vals)
                
                if len(vals) >= 5:
                    omega4 = vals[3]
                    omega5 = vals[4]
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
        
        if min_omega5 == np.inf or max_omega4 <= 0:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        if midgap <= 0:
            return -1.0
        return gap / midgap
    
    def make_cell(radius, inverted=False):
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                inside = x*x + y*y < radius*radius
                if inverted:
                    row.append(1 if inside else 0)
                else:
                    row.append(1 if not inside else 0)
            cell.append(row)
        return cell
    
    # Evaluate a few candidates
    candidates = []
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(make_cell(r, inverted=False))
    for r in [0.15, 0.2, 0.25, 0.3]:
        candidates.append(make_cell(r, inverted=True))
    
    best_cell = candidates[0]
    best_score = -1.0
    
    for cell in candidates:
        if time.time() > deadline - 30:
            break
        try:
            score = evaluate_cell(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            continue
    
    return best_cell
