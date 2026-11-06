"""
QAOA p=2 graph-dependent angle optimizer — analytic-adjoint L-BFGS-B.

Strategy:
  1. Precompute state data (bit patterns, cut counts) once per graph.
  2. Define a fast forward pass that returns both the expected cut and
     the intermediate state vectors needed for the adjoint backward pass.
  3. Compute the analytic gradient of the expected cut w.r.t. all four
     angles (gamma1, beta1, gamma2, beta2) via the adjoint method.
  4. Run L-BFGS-B from multiple diverse initial points (random + graph-
     structure-aware) to cover the landscape.
  5. Return the best (p, 2) array.
"""
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10 s safety margin

_value_cache = {}


def _remaining_time():
    return _DEADLINE - time.time()


def _build_graph_data(n, edges):
    """Precompute state index array, bit matrix, cut-count vector,
    and per-qubit flip index pairs for the mixer."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)

    # Bit matrix: bits[i, j] = j-th bit of state i
    bits = np.zeros((dim, n), dtype=np.int8)
    for j in range(n):
        bits[:, j] = (states >> j) & 1

    # Cut count for each state
    cut_counts = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        cut_counts += (bits[:, u] != bits[:, v]).astype(np.float64)

    # Mixer flip pairs: for each qubit i, the indices that differ in bit i
    flip_pairs = []
    for i in range(n):
        mask = 1 << i
        idx_0 = (states & mask) == 0
        idx_0_arr = np.where(idx_0)[0]
        idx_1_arr = idx_0_arr + mask
        flip_pairs.append((idx_0_arr, idx_1_arr))

    return states, bits, cut_counts, flip_pairs


def _forward(n, edges, g1, b1, g2, b2, cut_counts, flip_pairs):
    """
    Forward pass of p=2 QAOA.
    Returns (expected_cut, psi_after_layer1, psi_before_mixer2)
    where psi_after_layer1 is the state after exp(-i*g1*C) (i.e. after cost
    of layer 1, before mixer of layer 1), and psi_before_mixer2 is the state
    after the mixer of layer 1 (before cost of layer 2).
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)

    # --- Layer 1: cost ---
    phases1 = np.exp(-1j * g1 * cut_counts)
    psi *= phases1
    psi_after_cost1 = psi.copy()  # state after cost of layer 1

    # --- Layer 1: mixer ---
    c1 = np.cos(b1)
    s1 = np.sin(b1)
    for (i0, i1) in flip_pairs:
        p0 = psi[i0]
        p1 = psi[i1]
        psi[i0] = c1 * p0 - 1j * s1 * p1
        psi[i1] = -1j * s1 * p0 + c1 * p1
    psi_after_mixer1 = psi.copy()

    # --- Layer 2: cost ---
    phases2 = np.exp(-1j * g2 * cut_counts)
    psi *= phases2

    # --- Layer 2: mixer ---
    c2 = np.cos(b2)
    s2 = np.sin(b2)
    for (i0, i1) in flip_pairs:
        p0 = psi[i0]
        p1 = psi[i1]
        psi[i0] = c2 * p0 - 1j * s2 * p1
        psi[i1] = -1j * s2 * p0 + c2 * p1

    # Expected cut
    probs = np.real(psi.conj() * psi)
    expected_cut = np.dot(probs, cut_counts)

    return expected_cut, psi_after_cost1, psi_after_mixer1, psi


def _gradient(n, edges, g1, b1, g2, b2, cut_counts, flip_pairs):
    """
    Analytic gradient of expected cut w.r.t. (g1, b1, g2, b2)
    using the adjoint / backpropagation method.

    The QAOA circuit for p=2:
      |psi0> = |+>^n
      |psi1> = exp(-i g1 C) |psi0>
      |psi2> = exp(-i b1 B) |psi1>
      |psi3> = exp(-i g2 C) |psi2>
      |psi4> = exp(-i b2 B) |psi3>

    Objective: f = <psi4| C |psi4>

    We backpropagate the gradient through the circuit.
    """
    dim = 1 << n

    # Forward pass
    ec, psi_c1, psi_m1, psi_final = _forward(n, edges, g1, b1, g2, b2, cut_counts, flip_pairs)

    # We need the "cost operator" applied to a state: C|psi> = cut_counts * psi
    # (since C = sum_{(u,v) in E} |u,v><u,v| + |v,u><v,u|, and for a computational
    #  basis state |s>, C|s> = cut_count(s) * |s>)

    # Adjoint: start from the "measurement" gradient
    # df/d|psi4> = 2 * C|psi4>  (real-valued objective, complex state)
    # But more precisely: f = sum_s c_s |psi4_s|^2
    # df/d psi4_s^* = c_s * psi4_s
    # So the adjoint vector at psi4 is: chi4 = cut_counts * psi_final  (as a "gradient w.r.t. conjugate")

    # We'll propagate chi backward. The convention:
    #   chi_k represents df/d psi_k^*  (gradient w.r.t. conjugate of state at step k)
    #   The gradient w.r.t. a parameter theta that appears in U_k (mapping psi_{k-1}->psi_k)
    #   is: dtheta = Re[ <chi_k | dU_k/dtheta | psi_{k-1}> ]

    # Step 4 (final mixer): psi4 = U_b2 * psi3, where U_b2 = exp(-i b2 B)
    # dU_b2/db2 = -i * B * U_b2
    # grad_b2 = Re[ <chi4 | dU_b2/db2 | psi3> ]
    #          = Re[ <chi4 | -i B U_b2 psi3> ]
    #          = -Im[ <chi4 | B psi3> ]   (since U_b2 psi3 = psi4, but we need B psi3)
    # Actually: B psi3 is not psi4. We need to apply B to psi3.
    # B = sum_i X_i.  B|s> = sum_i |s flipped at i>
    # So (B psi3)_k = sum_i psi3[s_k flipped at i]

    # Let me use a cleaner approach: track chi backward through each gate.

    # chi at psi4 (gradient of f w.r.t. psi4^*):
    chi = cut_counts.astype(np.complex128) * psi_final

    # --- Back through mixer 2: psi4 = U_b2 psi3 ---
    # U_b2 = prod_i exp(-i b2 X_i)
    # dU_b2/db2 = -i B U_b2  =>  dU_b2/db2 |psi3> = -i B |psi3>
    # grad_b2 = Re[ <chi | dU_b2/db2 |psi3> ] = Re[ <chi | -i B psi3> ] = Im[ <chi | B psi3> ]

    # Compute B psi3:
    # B|psi3>_k = sum_i psi3[k with bit i flipped]
    # But we need <chi | B psi3> = sum_k chi_k^* (B psi3)_k
    # = sum_k chi_k^* sum_i psi3[k^flip_i]
    # = sum_i sum_k chi_k^* psi3[k^flip_i]
    # = sum_i <chi | X_i psi3>

    # We have psi3 = psi_m1 (state after mixer 1, before cost 2)
    # Wait, let me re-index:
    # psi0 -> [cost1] -> psi1 = psi_c1 -> [mixer1] -> psi2 = psi_m1 -> [cost2] -> psi3 -> [mixer2] -> psi4
    # So psi3 is the state after cost2. I didn't save it separately.
    # Let me re-do: I need psi3 for the backward pass.

    # Actually, let me just recompute in the gradient function to have all needed states.

    # Re-do forward, saving all intermediate states:
    psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    psi1 = psi0 * np.exp(-1j * g1 * cut_counts)
    c1 = np.cos(b1); s1 = np.sin(b1)
    psi2 = psi1.copy()
    for (i0, i1) in flip_pairs:
        p0 = psi2[i0]; p1 = psi2[i1]
        psi2[i0] = c1 * p0 - 1j * s1 * p1
        psi2[i1] = -1j * s1 * p0 + c1 * p1
    psi3 = psi2 * np.exp(-1j * g2 * cut_counts)
    c2 = np.cos(b2); s2 = np.sin(b2)
    psi4 = psi3.copy()
    for (i0, i1) in flip_pairs:
        p0 = psi4[i0]; p1 = psi4[i1]
        psi4[i0] = c2 * p0 - 1j * s2 * p1
        psi4[i1] = -1j * s2 * p0 + c2 * p1

    # Expected cut
    probs = np.real(psi4.conj() * psi4)
    ec = np.dot(probs, cut_counts)

    # chi4 = gradient of f w.r.t. psi4^* = cut_counts * psi4
    chi4 = (cut_counts * psi4).astype(np.complex128)

    # --- Back through mixer 2: psi4 = U_b2 psi3 ---
    # Each qubit i has exp(-i b2 X_i). Process them in reverse order (i = n-1, ..., 0)
    # For a single-qubit mixer on qubit i:
    #   psi4 = D_i * psi3_mid, where D_i = [[c2, -i s2], [-i s2, c2]] acting on qubit i
    #   dD_i/db2 = -i X_i D_i
    #   grad contribution from this qubit: Re[ <chi_mid | -i X_i D_i | psi3_mid> ]
    #   where chi_mid is the adjoint at the input of this qubit's mixer

    # Simpler: compute B psi3 and use the formula
    # grad_b2 = Im[ <chi4 | B psi3> ]  -- this is the total gradient w.r.t. b2

    # B psi3 = sum_i X_i psi3
    # <chi4 | B psi3> = sum_i <chi4 | X_i psi3>
    # <chi4 | X_i psi3> = sum_{k: bit_i=0} chi4_k^* psi3_{k + 2^i} + chi4_{k+2^i}^* psi3_k
    #                   = sum_{(i0,i1)} (chi4[i0]^* psi3[i1] + chi4[i1]^* psi3[i0])

    B_psi3_inner = 0j
    for (i0, i1) in flip_pairs:
        B_psi3_inner += np.dot(chi4[i0].conj(), psi3[i1]) + np.dot(chi4[i1].conj(), psi3[i0])

    grad_b2 = B_psi3_inner.imag  # Im[<chi4|B psi3>]

    # chi3 = gradient w.r.t. psi3^* = U_b2^dagger chi4
    # U_b2^dagger = prod_i exp(i b2 X_i) (reverse order, but X_i commute so order doesn't matter)
    # Apply exp(i b2 X_i) to chi4 for each i:
    #   new_chi[i0] = c2 * chi[i0] + 1j * s2 * chi[i1]
    #   new_chi[i1] = 1j * s2 * chi[i0] + c2 * chi[i1]
    chi3 = chi4.copy()
    for (i0, i1) in flip_pairs:
        c0 = chi3[i0]; c1v = chi3[i1]
        chi3[i0] = c2 * c0 + 1j * s2 * c1v
        chi3[i1] = 1j * s2 * c0 + c2 * c1v

    # --- Back through cost 2: psi3 = D_g2 psi2, D_g2 = diag(exp(-i g2 cut_counts)) ---
    # dD_g2/dg2 = -i * cut_counts * D_g2
    # grad_g2 = Re[ <chi3 | -i C psi2> ] = Re[ <chi3 | -i * cut_counts * psi2> ] = Im[ <chi3 | cut_counts * psi2> ]
    grad_g2 = np.sum(chi3.conj() * cut_counts * psi2).imag

    # chi2 = chi3 * D_g2^dagger = chi3 * exp(i g2 cut_counts)
    chi2 = chi3 * np.exp(1j * g2 * cut_counts)

    # --- Back through mixer 1: psi2 = U_b1 psi1 ---
    B_psi1_inner = 0j
    for (i0, i1) in flip_pairs:
        B_psi1_inner += np.dot(chi2[i0].conj(), psi1[i1]) + np.dot(chi2[i1].conj(), psi1[i0])

    grad_b1 = B_psi1_inner.imag

    chi1 = chi2.copy()
    for (i0, i1) in flip_pairs:
        c0 = chi1[i0]; c1v = chi1[i1]
        chi1[i0] = c1 * c0 + 1j * s1 * c1v
        chi1[i1] = 1j * s1 * c0 + c1 * c1v

    # --- Back through cost 1: psi1 = D_g1 psi0 ---
    grad_g1 = np.sum(chi1.conj() * cut_counts * psi0).imag

    # chi0 = chi1 * exp(i g1 cut_counts)  (not needed further)

    return ec, np.array([grad_g1, grad_b1, grad_g2, grad_b2])


def _objective_and_grad(params, n, cut_counts, flip_pairs):
    """Combined objective (negative for minimization) and gradient."""
    g1, b1, g2, b2 = params
    ec, grad = _gradient(n, None, g1, b1, g2, b2, cut_counts, flip_pairs)
    return -ec, -grad


def _graph_informed_init(n, edges, cut_counts):
    """Generate initial parameter guesses based on graph structure and QAOA scaling."""
    m = len(edges)
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    avg_cut = np.mean(cut_counts)
    max_cut = np.max(cut_counts)
    
    # For p=2 QAOA, known asymptotic scaling suggests:
    # gamma ~ O(1/p) = O(1/2), beta ~ O(1/p) = O(1/2)
    # But for small n and specific densities, we can refine
    # Dense graphs: angles tend to be smaller
    # Sparse graphs: angles can be larger
    
    # Base estimates from literature heuristics for p=2
    # For random graphs with density d:
    # gamma1 ≈ 0.3 * (1 - 0.5*d), beta1 ≈ 0.15 * (1 - 0.3*d)
    # gamma2 ≈ 0.2 * (1 - 0.5*d), beta2 ≈ 0.1 * (1 - 0.3*d)
    
    g1_base = 0.3 * (1 - 0.5 * density)
    b1_base = 0.15 * (1 - 0.3 * density)
    g2_base = 0.2 * (1 - 0.5 * density)
    b2_base = 0.1 * (1 - 0.3 * density)
    
    # Create diverse initial points around the expected region
    inits = [
        # Center estimate
        np.array([g1_base, b1_base, g2_base, b2_base]),
        # Slightly larger gammas
        np.array([g1_base * 1.3, b1_base * 0.8, g2_base * 1.3, b2_base * 0.8]),
        # Slightly smaller gammas, larger betas
        np.array([g1_base * 0.7, b1_base * 1.3, g2_base * 0.7, b2_base * 1.3]),
        # Asymmetric: large first gamma, small second
        np.array([g1_base * 1.5, b1_base * 0.7, g2_base * 0.5, b2_base * 1.2]),
        # Asymmetric: small first gamma, large second
        np.array([g1_base * 0.5, b1_base * 1.2, g2_base * 1.5, b2_base * 0.7]),
        # Uniform moderate
        np.array([0.25, 0.12, 0.2, 0.08]),
        # Larger angles (for sparse graphs)
        np.array([0.35, 0.18, 0.28, 0.12]),
        # Smaller angles (for dense graphs)
        np.array([0.18, 0.08, 0.14, 0.06]),
    ]
    return inits


def angles(n, edges, p):
    global _value_cache

    # Normalize edges for caching
    edge_set = frozenset(frozenset((u, v)) for u, v in edges)
    cache_key = (n, edge_set)

    if cache_key in _value_cache:
        return _value_cache[cache_key]

    # Precompute
    states, bits, cut_counts, flip_pairs = _build_graph_data(n, edges)

    def obj_and_grad(params):
        return _objective_and_grad(params, n, cut_counts, flip_pairs)

    # Multiple starting points
    inits = _graph_informed_init(n, edges, cut_counts)

    # Add more random starts with different seeds for diversity
    rng = np.random.default_rng(42)
    for seed in [1, 2, 3, 4, 5]:
        rng_i = np.random.default_rng(seed)
        inits.append(rng_i.uniform(0.05, 0.5, size=4))
    
    # Also add starts near known good regions from literature
    inits.append(np.array([0.25, 0.1, 0.2, 0.08]))
    inits.append(np.array([0.3, 0.12, 0.25, 0.1]))
    inits.append(np.array([0.2, 0.15, 0.18, 0.12]))

    best_ec = -np.inf
    best_params = np.array([0.25, 0.1, 0.25, 0.1])

    for init in inits:
        if _remaining_time() < 5.0:
            break
        try:
            res = minimize(
                obj_and_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.001, 1.5)] * 4,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
            )
            ec_val = -res.fun
            if ec_val > best_ec:
                best_ec = ec_val
                best_params = res.x
        except Exception:
            continue

    # Final high-precision polish with even tighter tolerances
    if _remaining_time() > 3.0:
        try:
            # Use a two-stage polish: first L-BFGS-B with tight tolerances,
            # then a very tight local search
            res2 = minimize(
                obj_and_grad,
                best_params,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0.0001, 2.0)] * 4,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200},
            )
            ec_val2 = -res2.fun
            if ec_val2 > best_ec:
                best_ec = ec_val2
                best_params = res2.x
        except Exception:
            pass
        
        # Even tighter polish if time allows
        if _remaining_time() > 1.5:
            try:
                res2b = minimize(
                    obj_and_grad,
                    best_params,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(0.00001, 2.0)] * 4,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 500},
                )
                ec_val2b = -res2b.fun
                if ec_val2b > best_ec:
                    best_ec = ec_val2b
                    best_params = res2b.x
            except Exception:
                pass

    # Also try Nelder-Mead polish (no gradient needed, can escape slightly)
    if _remaining_time() > 2.0:
        try:
            def obj_only(params):
                g1, b1, g2, b2 = params
                ec, _ = _gradient(n, None, g1, b1, g2, b2, cut_counts, flip_pairs)
                return -ec

            # Use a smaller simplex for more precise local search
            initial_simplex = [best_params + np.array([1e-4, 0, 0, 0]),
                              best_params + np.array([0, 1e-4, 0, 0]),
                              best_params + np.array([0, 0, 1e-4, 0]),
                              best_params + np.array([0, 0, 0, 1e-4])]
            
            res3 = minimize(
                obj_only,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                        'adaptive': True},
            )
            ec_val3 = -res3.fun
            if ec_val3 > best_ec:
                best_ec = ec_val3
                best_params = res3.x
        except Exception:
            pass
        
        # Final ultra-tight polish
        if _remaining_time() > 0.5:
            try:
                res4 = minimize(
                    obj_and_grad,
                    best_params,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(0.000001, 2.0)] * 4,
                    options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 1000},
                )
                ec_val4 = -res4.fun
                if ec_val4 > best_ec:
                    best_ec = ec_val4
                    best_params = res4.x
            except Exception:
                pass

    # Format output
    out = np.zeros((p, 2))
    out[0, 0] = best_params[0]  # gamma 1
    out[0, 1] = best_params[1]  # beta 1
    if p > 1:
        out[1, 0] = best_params[2]  # gamma 2
        out[1, 1] = best_params[3]  # beta 2

    _value_cache[cache_key] = out
    return out
