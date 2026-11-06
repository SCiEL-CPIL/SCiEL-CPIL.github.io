import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    def _raw_score(centers):
        if not np.all(np.isfinite(centers)):
            return 0.0
        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms > 1.0):
            return 0.0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        min_edge_dist = np.min(1.0 - norms)
        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(0.0, r)

    def _objective(centers_flat, beta):
        centers = centers_flat.reshape(N, 2)
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        soft_min_val = -log_sum_exp / beta
        return -soft_min_val

    def _gradient(centers_flat, beta):
        centers = centers_flat.reshape(N, 2)
        grad = np.zeros_like(centers)
        norms = np.sqrt(np.sum(centers**2, axis=1)) + 1e-12
        edge_constraints = 1.0 - norms
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-12
        iu = np.triu_indices(N, 1)
        pair_constraints = dists[iu] / 2.0
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        edge_w = weights[:N]
        grad += edge_w[:, np.newaxis] * (centers / norms[:, np.newaxis])
        i_idx = iu[0]
        j_idx = iu[1]
        w_pair = weights[N:]
        diff_pairs = centers[i_idx] - centers[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
        return grad.reshape(-1)

    def _refine(centers, betas, maxiter_each=100):
        current = centers.copy()
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
            try:
                result = minimize(
                    _objective,
                    current.flatten(),
                    args=(beta,),
                    jac=lambda x, b: _gradient(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': maxiter_each, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30}
                )
                if result.success or result.fun < _objective(current.flatten(), beta):
                    current = result.x.reshape(N, 2)
            except Exception:
                pass
        return current

    def _make_rings(radii, counts, offsets=None):
        centers = [[0.0, 0.0]]
        start_idx = 1
        for r, c in zip(radii, counts):
            base_offset = offsets[start_idx-1] if offsets else 0.0
            for k in range(c):
                theta = 2 * np.pi * k / c + base_offset
                centers.append([r * np.cos(theta), r * np.sin(theta)])
            start_idx += 1
        return np.array(centers[:N])

    def _make_hex(spacing):
        points = []
        for j in range(-15, 16):
            for i in range(-15, 16):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2 - 1e-9)**2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]

    # Generate diverse initializations
    inits = []
    
    # Ring configurations
    r1 = [0.0, 0.22, 0.44, 0.66, 0.86]
    c1 = [1, 6, 12, 18, 13]
    inits.append(_make_rings(r1, c1))
    
    for seed in range(12):
        np.random.seed(seed * 1000 + 42)
        pert_r = [0.0] + [r + np.random.uniform(-0.02, 0.02) for r in r1[1:]]
        offsets = [0.0] + [np.random.uniform(-0.15, 0.15) for _ in range(4)]
        inits.append(_make_rings(pert_r, c1, offsets))
    
    # Hexagonal lattices
    for sp in [0.23, 0.24, 0.25, 0.26]:
        h = _make_hex(sp)
        if h is not None:
            inits.append(h)
    
    # Random with jitter
    np.random.seed(123)
    for _ in range(8):
        cfg = np.random.uniform(-0.93, 0.93, size=(N, 2))
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.93
        if np.any(mask):
            cfg[mask] *= 0.93 / norms[mask, np.newaxis]
        inits.append(cfg)

    # Phase 1: Refine all initializations with multi-stage alpha
    best_score = 0.0
    best_centers = inits[0]
    top_solutions = []
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        refined = _refine(x0, [50, 150, 300, 600], maxiter_each=80)
        score = _raw_score(refined)
        if score > best_score:
            best_score = score
            best_centers = refined.copy()
        top_solutions.append((score, refined.copy()))
        top_solutions.sort(key=lambda x: x[0], reverse=True)
        top_solutions = top_solutions[:5]
    
    # Phase 2: Simulated Annealing from top solutions
    def _sa(centers, steps=1500, T_init=0.012, T_final=0.0002):
        current = centers.copy()
        current_score = _raw_score(current)
        best = current.copy()
        best_score = current_score
        for i in range(steps):
            if time.time() > deadline - 15.0:
                break
            T = T_init * (T_final / T_init) ** (i / steps)
            n_move = np.random.randint(1, 4)
            idxs = np.random.choice(N, n_move, replace=False)
            new_c = current.copy()
            step = max(T * 0.4, 0.004)
            new_c[idxs] += np.random.normal(0, step, size=(n_move, 2))
            norms = np.sqrt(np.sum(new_c**2, axis=1))
            out = norms > 0.998
            if np.any(out):
                new_c[out] *= 0.998 / norms[out, np.newaxis]
            new_s = _raw_score(new_c)
            delta = new_s - current_score
            if delta > 0:
                current = new_c
                current_score = new_s
                if current_score > best_score:
                    best_score = current_score
                    best = new_c.copy()
            else:
                clamped = max(delta, -10.0 * T)
                if np.random.rand() < np.exp(clamped / T):
                    current = new_c
                    current_score = new_s
        return best, best_score

    for sa_round in range(min(3, len(top_solutions))):
        if time.time() > deadline - 20.0:
            break
        start_sol = top_solutions[sa_round][1]
        sa_c, sa_s = _sa(start_sol, steps=1500, T_init=0.012, T_final=0.0002)
        if sa_s > best_score:
            best_score = sa_s
            best_centers = sa_c.copy()
            top_solutions.append((sa_s, sa_c.copy()))
            top_solutions.sort(key=lambda x: x[0], reverse=True)
            top_solutions = top_solutions[:5]
        if time.time() < deadline - 10.0:
            refined_sa = _refine(best_centers, [200, 400, 700], maxiter_each=100)
            s = _raw_score(refined_sa)
            if s > best_score:
                best_score = s
                best_centers = refined_sa.copy()
                top_solutions.append((s, refined_sa.copy()))
                top_solutions.sort(key=lambda x: x[0], reverse=True)
                top_solutions = top_solutions[:5]

    # Final polish
    if time.time() < deadline - 5.0:
        final = _refine(best_centers, [400, 700, 1000], maxiter_each=150)
        fs = _raw_score(final)
        if fs > best_score:
            best_score = fs
            best_centers = final

    if time.time() < deadline - 2.0:
        final2 = _refine(best_centers, [800, 1200], maxiter_each=100)
        fs2 = _raw_score(final2)
        if fs2 > best_score:
            best_score = fs2
            best_centers = final2

    if not np.all(np.isfinite(best_centers)):
        return inits[0]
    return best_centers

if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
