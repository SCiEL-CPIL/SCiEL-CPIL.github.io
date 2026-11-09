"""Morse cluster global minima: Multi-start Basin Hopping with structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 310.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0


def _validate_and_fix(coords):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        # Simple iterative repulsion
        for _ in range(200):
            d = cdist(coords, coords)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            if min_d >= 0.5:
                break
            # Find closest pair
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            if i == j:
                continue
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.array([0.01, 0.01, 0.01])
                dist = np.linalg.norm(diff)
            diff /= dist
            coords[i] += 0.02 * diff
            coords[j] -= 0.02 * diff
    return coords


def _perturb(coords, rng, scale):
    """Random displacement of a random subset of atoms."""
    n = len(coords)
    # Perturb 10-30% of atoms, varying scale
    k = max(1, int(n * (0.1 + 0.2 * rng.random())))
    idx = rng.choice(n, size=k, replace=False)
    noise = rng.normal(0.0, scale, size=(k, 3))
    new = coords.copy()
    new[idx] += noise
    return new


def _basin_hopping_chain(rho, x0, rng, max_iter=300, temp_start=0.15, temp_end=0.005):
    """
    Basin hopping with cooling temperature.
    Returns (best_x, best_e).
    """
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    cur_x = best_x.copy()
    cur_e = best_e
    
    # Initial relaxation
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        if e_rel < 1e8:
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
        else:
            # Try to fix
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel < 1e8:
                cur_x = x_rel
                cur_e = e_rel
                if cur_e < best_e:
                    best_e = cur_e
                    best_x = x_rel.copy()
    except Exception:
        pass

    for i in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Cooling schedule
        t = i / max_iter
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Perturb
        # Scale varies to allow exploring different basin sizes
        scale = 0.08 + 0.25 * rng.random()
        x_new = _perturb(cur_x, rng, scale)
        
        # Relax
        try:
            e_rel, x_rel = relax(x_new, 'morse', rho)
        except Exception:
            continue
        
        if e_rel > 1e8:
            # Try to fix and retry once
            try:
                x_fixed = _validate_and_fix(x_rel)
                e_rel, x_rel = relax(x_fixed, 'morse', rho)
            except Exception:
                continue
            if e_rel > 1e8:
                continue

        dE = e_rel - cur_e
        # Metropolis acceptance
        if dE < 0 or rng.random() < np.exp(-dE / temp):
            cur_x = x_rel
            cur_e = e_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_rel.copy()
                
    return best_x, best_e


def _generate_seeds(n, rng):
    """Generate a diverse set of initial configurations."""
    seeds = []
    
    # 1. FCC-like packing
    a = 1.0
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    fcc_seed = pts[order[:n]].copy()
    fcc_seed += rng.normal(0, 0.05, fcc_seed.shape)
    seeds.append(fcc_seed)

    # 2. Icosahedral shell
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    if n <= 12:
        ico_seed = ico[:n].copy()
    else:
        inner_n = n - 12
        inner_r = 0.6
        inner_pts = []
        for _ in range(inner_n):
            while True:
                p = rng.uniform(-inner_r, inner_r, 3)
                if np.linalg.norm(p) < inner_r:
                    break
            inner_pts.append(p)
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.05, ico_seed.shape)
    seeds.append(ico_seed)

    # 3. Random Compact
    radius = n ** (1/3) * 0.8
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 4. Spherical Shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.5
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.05, shell_seed.shape)
    seeds.append(shell_seed)

    # 5. Two-Shell Structure
    if n > 10:
        outer_n = max(10, int(n * 0.6))
        inner_n = n - outer_n
        outer_r = 1.2
        inner_r = 0.5
        
        outer_pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            outer_pts.append([rr * np.cos(theta), rr * np.sin(theta), z])
        outer_pts = np.array(outer_pts) * outer_r
        
        inner_pts = rng.uniform(-inner_r, inner_r, size=(inner_n, 3))
        two_shell = np.vstack([outer_pts, inner_pts])
        two_shell += rng.normal(0, 0.05, two_shell.shape)
        seeds.append(two_shell)

    # 6. HCP-like packing
    a = 1.0
    hcp_pts = []
    for layer in range(-3, 4):
        z = layer * 0.8165  # sqrt(2/3)
        for i in range(-4, 5):
            for j in range(-4, 5):
                if layer % 2 == 0:
                    x = i * a + 0.5 * j * a
                    y = j * a * 0.866
                else:
                    x = i * a + 0.25 * j * a
                    y = j * a * 0.866 + 0.2887
                hcp_pts.append([x, y, z])
    hcp_pts = np.array(hcp_pts)
    hcp_pts -= hcp_pts.mean(axis=0)
    dists = np.linalg.norm(hcp_pts, axis=1)
    order = np.argsort(dists)
    hcp_seed = hcp_pts[order[:n]].copy()
    hcp_seed += rng.normal(0, 0.05, hcp_seed.shape)
    seeds.append(hcp_seed)

    # 7. BCC-like packing
    a = 1.0
    bcc_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                bcc_pts.append([i*a, j*a, k*a])
                bcc_pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    bcc_pts = np.array(bcc_pts)
    bcc_pts -= bcc_pts.mean(axis=0)
    dists = np.linalg.norm(bcc_pts, axis=1)
    order = np.argsort(dists)
    bcc_seed = bcc_pts[order[:n]].copy()
    bcc_seed += rng.normal(0, 0.05, bcc_seed.shape)
    seeds.append(bcc_seed)

    # 8. Random with varying density
    for scale in [0.6, 1.0, 1.4]:
        radius = n ** (1/3) * 0.8 * scale
        random_seed = rng.uniform(-radius, radius, size=(n, 3))
        seeds.append(random_seed)

    return seeds


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    # Relax all seeds first to get a baseline
    best_x = None
    best_e = np.inf
    
    for seed in seeds:
        try:
            e, x = relax(seed, 'morse', rho)
            if e < 1e8:
                if e < best_e:
                    best_e = e
                    best_x = x.copy()
            else:
                x_fixed = _validate_and_fix(x)
                e, x = relax(x_fixed, 'morse', rho)
                if e < 1e8:
                    if e < best_e:
                        best_e = e
                        best_x = x.copy()
        except Exception:
            continue

    if best_x is None:
        return seeds[0]

    # Parallel Basin Hopping using ThreadPoolExecutor
    # We use 4 threads as allowed by the prompt
    num_chains = 4
    chain_starts = seeds[:num_chains]
    
    # If we have fewer seeds than chains, add random ones
    while len(chain_starts) < num_chains:
        r = n ** (1/3) * 0.8
        chain_starts.append(rng.uniform(-r, r, size=(n, 3)))
    
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chains) as executor:
        future_to_seed = {}
        for i in range(num_chains):
            x0 = chain_starts[i].copy()
            # Create a unique RNG for each thread to avoid contention
            thread_rng = np.random.default_rng(rng.integers(0, 2**32))
            future = executor.submit(_basin_hopping_chain, rho, x0, thread_rng, max_iter=300, temp_start=0.15, temp_end=0.005)
            future_to_seed[future] = i
        
        for future in concurrent.futures.as_completed(future_to_seed, timeout=_remaining()):
            try:
                x_bh, e_bh = future.result(timeout=1.0)
                results.append((x_bh, e_bh))
            except Exception:
                continue

    # Combine results
    for x_bh, e_bh in results:
        if e_bh < best_e:
            best_e = e_bh
            best_x = x_bh.copy()

    # Second round of basin hopping from the best found so far
    if _remaining() > 10.0:
        # Create multiple perturbations of the best solution
        num_extra = min(4, int(_remaining() // 10))
        extra_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_extra) as executor:
            futures = []
            for i in range(num_extra):
                x0 = _perturb(best_x, rng, 0.15 + 0.1 * i)
                thread_rng = np.random.default_rng(rng.integers(0, 2**32))
                future = executor.submit(_basin_hopping_chain, rho, x0, thread_rng, max_iter=150, temp_start=0.1, temp_end=0.01)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures, timeout=_remaining()):
                try:
                    x_bh, e_bh = future.result(timeout=1.0)
                    extra_results.append((x_bh, e_bh))
                except Exception:
                    continue

        for x_bh, e_bh in extra_results:
            if e_bh < best_e:
                best_e = e_bh
                best_x = x_bh.copy()

    # Final Polish
    if _remaining() > 5.0:
        for _ in range(10):
            if _remaining() < 2.0:
                break
            x_pert = _perturb(best_x, rng, 0.03 + 0.03 * rng.random())
            try:
                e, x = relax(x_pert, 'morse', rho)
                if e < 1e8 and e < best_e:
                    best_e = e
                    best_x = x.copy()
            except Exception:
                break
        
        # One final relax
        try:
            _, best_x = relax(best_x, 'morse', rho)
        except Exception:
            pass
        
        # Validate
        if _morse_energy(rho, best_x) < 1e8:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
