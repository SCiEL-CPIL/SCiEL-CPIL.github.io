<<<<<<< SEARCH
def construct(instance):
    t0 = time.time()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)

    # Build the ordered list of starting configurations
    starts = []
    for gen in (_fcc_cluster, _hcp_cluster, _icosahedral_cluster, _compressed_icosahedral):
        starts.append(gen(size, spacing))
    # random compact with a few seeds
    for s in (size, size + 101, size + 202):
        starts.append(_random_compact(size, spacing, s))
    # perturbations of the structured seeds
    for base in starts[:4]:
        for sigma in (0.08, 0.15):
            starts.append(_perturb(base, sigma, size + int(sigma * 100)))

    best_energy = None
    best_pos = None
    for k, start in enumerate(starts):
        if not _deadline_ok(t0):
            break
        if best_pos is None:
            best_pos = _ensure_min_dist(start)  # always have a valid fallback
        e, p = _nested_relax(start, potential, parameter, t0)
        if p is not None and np.all(np.isfinite(p)):
            if best_energy is None or e < best_energy:
                best_energy = e
                best_pos = p
    # final validity guarantee
    return _ensure_min_dist(best_pos)
=======
def construct(instance):
    t0 = time.time()
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    parameter = instance.get("rho", kind)
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)

    # Build the ordered list of starting configurations
    starts = []
    for gen in (_fcc_cluster, _hcp_cluster, _icosahedral_cluster, _compressed_icosahedral):
        starts.append(gen(size, spacing))
    # random compact with a few seeds
    for s in (size, size + 101, size + 202):
        starts.append(_random_compact(size, spacing, s))
    # perturbations of the structured seeds
    for base in starts[:4]:
        for sigma in (0.08, 0.15):
            starts.append(_perturb(base, sigma, size + int(sigma * 100)))

    best_energy = None
    best_pos = None
    for k, start in enumerate(starts):
        if not _deadline_ok(t0):
            break
        if best_pos is None:
            best_pos = _ensure_min_dist(start)  # always have a valid fallback
        # Use annealed relax which combines coarse-to-fine with nested relax
        e, p = _annealed_relax(start, kind, t0)
        if p is not None and np.all(np.isfinite(p)):
            if best_energy is None or e < best_energy:
                best_energy = e
                best_pos = p
    # final validity guarantee
    return _ensure_min_dist(best_pos)
>>>>>>> REPLACE
