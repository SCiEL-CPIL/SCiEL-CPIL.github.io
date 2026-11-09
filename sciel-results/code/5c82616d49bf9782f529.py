"""Sutton-Chen cluster global minima via Basin Hopping with Soft-Potential Relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import scipy.optimize as opt

def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # Efficient pairwise distance check
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True

def _generate_fcc(n, lattice=0.8):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _generate_hcp(n, lattice=0.8):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _generate_icosahedral(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v.copy())
    # Add edge midpoints for shell 2
    edge_mids = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.7
                edge_mids.append(mid)
    pts.extend(edge_mids)
    pts = np.array(pts)
    if len(pts) >= n:
        center = pts.mean(axis=0)
        d = np.linalg.norm(pts - center, axis=1)
        order = np.argsort(d)
        return pts[order[:n]]
    # Pad if needed
    rng = np.random.default_rng(42)
    while len(pts) < n:
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * 2.5
        pts = np.vstack([pts, p])
    center = pts.mean(axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _soft_relax(pos, kind, n_exp, m_exp, c, maxiter=100):
    """
    Relax using a softened potential to allow escape from local minima.
    We modify the exponents to be smaller, creating a smoother landscape.
    """
    # Soften exponents: e.g., 12->6, 6->3. This reduces the repulsive steepness.
    soft_n = max(2, n_exp // 2)
    soft_m = max(2, m_exp // 2)
    soft_c = c * (soft_n / n_exp) # Rough scaling, not critical as we only care about topology
    
    def soft_obj(x):
        x = x.reshape(-1, 3)
        if not np.all(np.isfinite(x)):
            return 1e10
        # Check min distance to avoid singularities
        for i in range(len(x)-1):
            d = np.linalg.norm(x[i+1:] - x[i], axis=1)
            if d.size and d.min() < 0.35:
                return 1e10
        
        n = len(x)
        energy = 0.0
        # Vectorized pair calculation for speed
        # This is O(N^2) which is fine for N<100
        for i in range(n):
            r = np.linalg.norm(x[i+1:] - x[i], axis=1)
            if r.size == 0: continue
            # Avoid division by zero
            r_safe = np.maximum(r, 1e-6)
            rep = np.sum(r_safe ** (-soft_n))
            att = np.sum(r_safe ** (-soft_m))
            energy += 0.5 * (rep - soft_c * np.sqrt(att))
        return energy

    def soft_grad(x):
        x = x.reshape(-1, 3)
        n = len(x)
        grad = np.zeros_like(x)
        for i in range(n):
            r_vec = x[i+1:] - x[i]
            r = np.linalg.norm(r_vec, axis=1)
            if r.size == 0: continue
            r_safe = np.maximum(r, 1e-6)
            
            # d/dr (r^-n) = -n r^-(n+1)
            # d/dr (r^-m) = -m r^-(m+1)
            
            # Repulsive term gradient
            # E_rep = sum r^-n
            # dE_rep/dx_i = -sum (n * r^-(n+2)) * r_vec
            grad_rep = -soft_n * np.sum((r_safe ** (-(soft_n + 1)))[:, None] * r_vec, axis=0)
            
            # Attractive term gradient
            # E_att = -c sqrt(sum r^-m)
            # Let S = sum r^-m
            # dE_att/dx_i = -c * (1/(2*sqrt(S))) * dS/dx_i
            # dS/dx_i = sum (-m * r^-(m+2)) * r_vec
            S = np.sum(r_safe ** (-soft_m))
            if S > 1e-10:
                dS = -soft_m * np.sum((r_safe ** (-(soft_m + 1)))[:, None] * r_vec, axis=0)
                grad_att = -soft_c * (1.0 / (2.0 * np.sqrt(S))) * dS
            else:
                grad_att = np.zeros(3)
                
            grad[i] += 0.5 * (grad_rep + grad_att)
            # Gradient for the other atoms in the pair
            # dE/dx_j = - dE/dx_i (by symmetry of the pair term, but we sum over i)
            # Actually, let's just accumulate for j
            # The term for pair (i,j) contributes to grad[i] and grad[j]
            # My loop structure above only updates grad[i]. I need to update grad[j] too.
            # Let's rewrite for clarity and correctness in vectorized way if possible, or just loop.
            # Given N is small (<100), a double loop is safer and fast enough for soft relax.
            
        # Re-calculate with explicit double loop for gradient correctness
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i+1, n):
                r_vec = x[j] - x[i]
                r = np.linalg.norm(r_vec)
                if r < 1e-6: continue
                r_safe = r
                
                # Repulsive
                coeff_rep = 0.5 * (-soft_n * r_safe ** (-(soft_n + 2)))
                # Attractive
                # We need the total sum S for this atom i? No, S is per-atom in SC potential.
                # E_i = 0.5 * (sum_j r_ij^-n - c sqrt(sum_j r_ij^-m))
                # This is NOT a simple pair potential sum. It's a many-body term.
                # The gradient of the many-body term is complex.
                # To avoid implementing the full many-body gradient manually (error-prone),
                # we will use numerical gradient for the soft potential if analytic is too hard,
                # OR we stick to the provided `relax` for hard potential and use a different strategy.
                
                # Actually, the provided `relax` uses L-BFGS-B which approximates Hessian.
                # For soft potential, we can just use `scipy.optimize.minimize` with BFGS or L-BFGS-B 
                # on the soft objective. We don't need analytic gradient for the soft part, 
                # just a good objective. Numerical gradient is slow but N is small.
                # However, calling `relax` (hard) is fast. 
                # The key insight from "inspiration" is that soft potential helps escape.
                # Let's implement a simple soft objective and use BFGS on it.
                
        # This function is getting messy. Let's simplify.
        # We will use `scipy.optimize.minimize` with method='L-BFGS-B' on the soft objective.
        # We will NOT provide a gradient, letting scipy use finite differences.
        # For N=100, 300 variables, FD gradient is 300 evaluations.
        # One soft relax step might take ~0.1s. 
        # If we do 100 steps, that's 10s. Feasible.
        
        # But wait, the `relax` function from the suite is highly optimized (C/Fortran likely?).
        # Let's check if we can just call `relax` with modified parameters? No, signature is fixed.
        
        # Alternative: Use the provided `sutton_chen_energy_gradient` but modify the input? No.
        
        # Let's stick to the plan: Implement a soft objective and use scipy minimize.
        # To make it fast, we will use a vectorized soft objective.
        
        pass # Placeholder, actual implementation below

    # Let's re-implement _soft_relax properly using vectorization in the objective.
    
    def soft_obj_vec(x):
        x = x.reshape(-1, 3)
        n = len(x)
        # Compute all pair distances
        # dists[i, j] = ||x_i - x_j||
        # Memory: N^2. For N=100, 10000 floats. Fine.
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero on diagonal
        
        # Check min distance
        min_dist = dists.min()
        if min_dist < 0.35:
            return 1e10
        
        # Sum over j for each i
        # rep[i] = sum_j dists[i, j]^-n
        rep = np.sum(dists ** (-soft_n), axis=1)
        att = np.sum(dists ** (-soft_m), axis=1)
        
        # E = sum_i 0.5 * (rep[i] - c * sqrt(att[i]))
        # Note: The diagonal is 1.0, so dists[i,i]^-n = 1. We should exclude diagonal.
        # Let's mask diagonal
        mask = ~np.eye(n, dtype=bool)
        rep = np.sum(np.where(mask, dists ** (-soft_n), 0), axis=1)
        att = np.sum(np.where(mask, dists ** (-soft_m), 0), axis=1)
        
        energy = np.sum(0.5 * (rep - soft_c * np.sqrt(att)))
        return energy

    try:
        res = opt.minimize(soft_obj_vec, pos.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-6})
        if res.success or res.fun < 1e9:
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return pos

def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Extract parameters for soft potential
    # We need to map kind to n_exp, m_exp, c
    # (12-6, 12, 6, 144.41), (9-6, 9, 6, 39.432), (10-8, 10, 8, 34.408)
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41

    # Generate diverse seeds
    seeds = [
        _generate_fcc(n, lattice=0.8),
        _generate_hcp(n, lattice=0.8),
        _generate_icosahedral(n),
        _generate_fcc(n, lattice=0.75),
        _generate_hcp(n, lattice=0.75)
    ]
    
    best_e = np.inf
    best_pos = None
    rng = np.random.default_rng(42)

    # Strategy: For each seed, run a short Basin Hopping cycle.
    # 1. Relax with Hard Potential (standard)
    # 2. Perturb
    # 3. Relax with Soft Potential (to escape local minima)
    # 4. Relax with Hard Potential (to refine)
    # 5. Accept if better.
    
    bh_steps = 10 # Number of BH iterations per seed
    pert_amp = 0.5 # Initial perturbation amplitude

    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        if not _validate_positions(seed):
            continue
            
        current_pos = seed.copy()
        current_e = np.inf
        
        # Initial hard relax
        try:
            current_e, current_pos = relax(current_pos, potential, kind)
            if not _validate_positions(current_pos):
                continue
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        except Exception:
            continue

        # Basin Hopping Loop
        for step in range(bh_steps):
            if time.time() > deadline - 5:
                break
                
            # Perturb
            noise = rng.normal(0, pert_amp, size=(n, 3))
            perturbed = current_pos + noise
            
            if not _validate_positions(perturbed):
                # Try smaller noise
                noise = rng.normal(0, pert_amp * 0.5, size=(n, 3))
                perturbed = current_pos + noise
                if not _validate_positions(perturbed):
                    continue

            # Soft Relax
            try:
                soft_pos = _soft_relax(perturbed, kind, n_exp, m_exp, c)
            except Exception:
                soft_pos = perturbed

            # Hard Relax
            try:
                new_e, new_pos = relax(soft_pos, potential, kind)
            except Exception:
                continue

            if not _validate_positions(new_pos):
                continue

            # Metropolis acceptance or simple hill climbing?
            # For global minima, simple hill climbing on basin energies is robust.
            # We accept if new_e < current_e.
            if new_e < current_e:
                current_e = new_e
                current_pos = new_pos.copy()
                if current_e < best_e:
                    best_e = current_e
                    best_pos = current_pos.copy()
            else:
                # Occasionally accept worse to escape? 
                # For deterministic scoring, let's stick to hill climbing for stability,
                # but maybe allow one step of annealing?
                # Let's try: if rng.random() < 0.1: accept
                pass # Keep it simple: Hill Climbing on Basin Energies

            # Anneal perturbation amplitude
            pert_amp *= 0.9

    # Final polish
    if best_pos is not None and time.time() < deadline - 2:
        try:
            final_e, final_pos = relax(best_pos, potential, kind)
            if _validate_positions(final_pos) and final_e < best_e:
                best_e = final_e
                best_pos = final_pos
        except Exception:
            pass

    if best_pos is None:
        # Fallback
        best_pos = _generate_fcc(n, 0.8)
        try:
            best_e, best_pos = relax(best_pos, potential, kind)
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
