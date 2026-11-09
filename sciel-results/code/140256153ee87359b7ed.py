"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin under 90s limit

    def _gl8():
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return nodes, weights

    def _map_nodes(nparams):
        # Map 8 unconstrained params to strictly increasing nodes in (0,1)
        # Use cumulative logit transform
        sig = 1.0 / (1.0 + np.exp(-nparams))
        nodes = np.zeros(8)
        nodes[0] = sig[0] * 0.999
        for i in range(1, 8):
            remaining = 1.0 - nodes[i - 1]
            nodes[i] = nodes[i - 1] + sig[i] * remaining * 0.999
        # Ensure strictly increasing and within (0, 1)
        for i in range(1, 8):
            if nodes[i] <= nodes[i - 1]:
                nodes[i] = nodes[i - 1] + 1e-8
        if nodes[0] <= 0:
            nodes[0] = 1e-8
        if nodes[-1] >= 1:
            nodes[-1] = 1.0 - 1e-8
        # Ensure strict monotonicity
        for i in range(1, 8):
            if nodes[i] <= nodes[i - 1]:
                nodes[i] = nodes[i - 1] + 1e-8
        if nodes[-1] >= 1.0:
            # Scale down
            nodes = nodes * (1.0 - 1e-6) / nodes[-1]
            for i in range(1, 8):
                if nodes[i] <= nodes[i - 1]:
                    nodes[i] = nodes[i - 1] + 1e-10
        return nodes

    def _map_weights(wparams):
        exp_w = np.exp(wparams - np.max(wparams))
        weights = exp_w / exp_w.sum()
        return weights

    def E_func(z, Omega_m, w0, w1):
        Omega_L = 1.0 - Omega_m
        one_plus_z = 1.0 + z
        return np.sqrt(Omega_m * one_plus_z**3 + Omega_L * one_plus_z**(3.0 * (1.0 + w0 + w1 * z)))

    def true_integral(z, Omega_m, w0, w1):
        integrand = lambda u: 1.0 / E_func(z * u, Omega_m, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
        return res

    # Expanded representative cosmologies for training
    cosmo_params = [
        (0.3, -1.0, 0.0),
        (0.3, -1.2, 0.0),
        (0.3, -0.8, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.1, 0.3),
        (0.3, -0.9, -0.3),
        (0.25, -1.0, 0.2),
        (0.35, -1.0, -0.3),
        (0.2, -0.9, 0.2),
        (0.3, -1.3, 0.0),
        (0.3, -0.7, 0.0),
        (0.3, -1.0, 0.7),
        (0.3, -1.0, -0.7),
    ]

    # Test z values - include a range
    z_test = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))

    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)

    if time.time() > deadline - 5:
        nodes, weights = _gl8()
        return {"nodes": nodes, "weights": weights}

    def evaluate_params(params):
        nparams = params[:8]
        wparams = params[8:]
        nodes = _map_nodes(nparams)
        weights = _map_weights(wparams)

        max_err = 0.0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                true_val = true_vals[i, j]
                if true_val == 0:
                    err = 0.0
                else:
                    err = abs(quad_val - true_val) / true_val
                if err > max_err:
                    max_err = err
        return max_err

    bounds = [(-6.0, 6.0)] * 16

    # Differential evolution with multiple trials
    best_params = None
    best_val = np.inf

    # First DE run
    try:
        result1 = differential_evolution(
            evaluate_params,
            bounds,
            maxiter=150,
            maxfev=1500,
            tol=1e-10,
            seed=42,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        if result1.fun < best_val:
            best_val = result1.fun
            best_params = result1.x
    except Exception:
        pass

    if time.time() > deadline - 15:
        pass
    else:
        # Second DE run with different seed
        try:
            result2 = differential_evolution(
                evaluate_params,
                bounds,
                maxiter=100,
                maxfev=1000,
                tol=1e-10,
                seed=123,
                polish=False,
                mutation=(0.8, 1.0),
                recombination=0.8
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_params = result2.x
        except Exception:
            pass

    if best_params is None:
        nodes, weights = _gl8()
        return {"nodes": nodes, "weights": weights}

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 8:
        try:
            def local_obj(params):
                nparams = params[:8]
                wparams = params[8:]
                nodes = _map_nodes(nparams)
                weights = _map_weights(wparams)
                max_err = 0.0
                for i, (Om, w0, w1) in enumerate(cosmo_params):
                    for j, z in enumerate(z_test):
                        z_u = z * nodes
                        E_vals = E_func(z_u, Om, w0, w1)
                        integrand_vals = 1.0 / E_vals
                        quad_val = np.dot(weights, integrand_vals)
                        true_val = true_vals[i, j]
                        if true_val == 0:
                            err = 0.0
                        else:
                            err = abs(quad_val - true_val) / true_val
                        if err > max_err:
                            max_err = err
                return max_err

            local_result = minimize(local_obj, best_params, method='Nelder-Mead',
                                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if local_result.fun < best_val:
                best_val = local_result.fun
                best_params = local_result.x
        except Exception:
            pass

    # Additional local refinement with different initial perturbation
    if time.time() < deadline - 5:
        try:
            perturbed = best_params + 0.1 * np.random.default_rng(7).normal(size=16)
            perturbed = np.clip(perturbed, -6.0, 6.0)

            def local_obj2(params):
                nparams = params[:8]
                wparams = params[8:]
                nodes = _map_nodes(nparams)
                weights = _map_weights(wparams)
                max_err = 0.0
                for i, (Om, w0, w1) in enumerate(cosmo_params):
                    for j, z in enumerate(z_test):
                        z_u = z * nodes
                        E_vals = E_func(z_u, Om, w0, w1)
                        integrand_vals = 1.0 / E_vals
                        quad_val = np.dot(weights, integrand_vals)
                        true_val = true_vals[i, j]
                        if true_val == 0:
                            err = 0.0
                        else:
                            err = abs(quad_val - true_val) / true_val
                        if err > max_err:
                            max_err = err
                return max_err

            local_result2 = minimize(local_obj2, perturbed, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            if local_result2.fun < best_val:
                best_val = local_result2.fun
                best_params = local_result2.x
        except Exception:
            pass

    # Final extraction
    nparams = best_params[:8]
    wparams = best_params[8:]
    nodes = _map_nodes(nparams)
    weights = _map_weights(wparams)

    # Validate
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1 or np.any(weights <= 0):
        nodes, weights = _gl8()
    else:
        weights = weights / weights.sum()

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
