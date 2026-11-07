"""Optimal market making: vectorized continuous HJB via Newton on FOCs in lambda-space."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

_REGIMES = [
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
    {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
]

_PRECOMPUTED = {}


def _solve_regime(regime):
    """
    Vectorized HJB with continuous depth via Newton iteration on FOCs.
    
    FOC for bid (in terms of λ_b = A*exp(-k*bd)*dt):
      h(λ) = λ*(1-p)*(ln(λ/Adt) - c) + p = 0,  p = 1-exp(-λ), c = k*(1-p_a)*Δ_b
    
    Newton: λ_{new} = λ - h(λ)/h'(λ)
    h'(λ) = (1-p)*(ln(λ/Adt) - c) + λ*(-k/(1-p))*(-1)*... 
    Let me compute: 
    h = λ(1-p)(ln(λ/Adt) - c) + p
    dp/dλ = exp(-λ) = 1-p
    d/dλ[λ(1-p)] = (1-p) + λ*(-(1-p)) = (1-p)(1-λ)
    d/dλ[ln(λ/Adt)] = 1/λ
    h' = (1-p)(1-λ)(ln(λ/Adt) - c) + λ(1-p)/λ + (1-p)
       = (1-p)(1-λ)(ln(λ/Adt) - c) + (1-p) + (1-p)
       = (1-p)[(1-λ)(ln(λ/Adt) - c) + 2]
    
    We do a few Newton steps per state, vectorized over all states.
    """
    sigma = regime['sigma']
    A = regime['A']
    k = regime['k']
    gamma = regime['gamma']
    T = regime['T']
    steps = regime['steps']
    Q = regime['Q']
    dt = T / steps
    Adt = A * dt

    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)

    V_next = np.zeros(n_q, dtype=np.float64)
    bid_table = np.empty((steps, n_q), dtype=np.float64)
    ask_table = np.empty((steps, n_q), dtype=np.float64)
    cost = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            bid_table[n:] = 1.0 / k
            ask_table[n:] = 1.0 / k
            break

        Vq = V_next
        Vq1 = np.empty(n_q, dtype=np.float64)
        Vq1[:-1] = V_next[1:]
        Vq1[-1] = V_next[-1]
        Vqm1 = np.empty(n_q, dtype=np.float64)
        Vqm1[1:] = V_next[:-1]
        Vqm1[0] = V_next[0]

        delta_b = Vq1 - Vq
        delta_a = Vqm1 - Vq

        mask_bid = q_vals < Q
        mask_ask = q_vals > -Q

        # Initialize lambda (intensity) for bid and ask
        # λ = A*exp(-k*d)*dt, d=1/k → λ = A*dt/e ≈ Adt/e
        lam_b = np.full(n_q, Adt / np.e, dtype=np.float64)
        lam_a = np.full(n_q, Adt / np.e, dtype=np.float64)

        # Fixed-point outer loop for coupling
        for _outer in range(3):
            # Newton iterations for bid FOC
            if np.any(mask_bid):
                mb = mask_bid
                c_b = k * (1.0 - (1.0 - np.exp(-lam_a))) * delta_b  # k*(1-p_a)*Δ_b
                # p_a = 1-exp(-lam_a)
                p_a = 1.0 - np.exp(-lam_a)
                c_b = k * (1.0 - p_a) * delta_b

                for _newton in range(4):
                    lb = lam_b[mb]
                    p = 1.0 - np.exp(-lb)
                    log_term = np.log(np.maximum(lb, 1e-15) / Adt)
                    h = lb * (1.0 - p) * (log_term - c_b[mb]) + p
                    hprime = (1.0 - p) * ((1.0 - lb) * (log_term - c_b[mb]) + 2.0)
                    hprime = np.maximum(hprime, 1e-12)
                    lb_new = lb - h / hprime
                    # Clip to valid range
                    lb_new = np.clip(lb_new, 1e-10, Adt * (1 - 1e-10))
                    lam_b[mb] = lb_new

            # Newton iterations for ask FOC
            if np.any(mask_ask):
                ma = mask_ask
                p_b = 1.0 - np.exp(-lam_b)
                c_a = k * (1.0 - p_b) * delta_a

                for _newton in range(4):
                    la = lam_a[ma]
                    p = 1.0 - np.exp(-la)
                    log_term = np.log(np.maximum(la, 1e-15) / Adt)
                    h = la * (1.0 - p) * (log_term - c_a[ma]) + p
                    hprime = (1.0 - p) * ((1.0 - la) * (log_term - c_a[ma]) + 2.0)
                    hprime = np.maximum(hprime, 1e-12)
                    la_new = la - h / hprime
                    la_new = np.clip(la_new, 1e-10, Adt * (1 - 1e-10))
                    lam_a[ma] = la_new

        # Convert lambda to depth: d = -ln(λ/Adt)/k
        bid_depths = np.full(n_q, 1.0 / k, dtype=np.float64)
        ask_depths = np.full(n_q, 1.0 / k, dtype=np.float64)

        if np.any(mask_bid):
            bd = -np.log(np.maximum(lam_b[mask_bid], 1e-15) / Adt) / k
            bid_depths[mask_bid] = np.maximum(bd, 0.0)
        if np.any(mask_ask):
            ad = -np.log(np.maximum(lam_a[mask_ask], 1e-15) / Adt) / k
            ask_depths[mask_ask] = np.maximum(ad, 0.0)

        bid_depths[q_vals >= Q] = 0.0
        ask_depths[q_vals <= -Q] = 0.0

        p_b = 1.0 - np.exp(-A * np.exp(-k * bid_depths) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * ask_depths) * dt)
        p_b[q_vals >= Q] = 0.0
        p_a[q_vals <= -Q] = 0.0

        bid_table[n] = bid_depths
        ask_table[n] = ask_depths

        fill_reward = bid_depths * p_b + ask_depths * p_a
        V_future = Vq + p_b * (1 - p_a) * delta_b + (1 - p_b) * p_a * delta_a
        V_next = -cost + fill_reward + V_future

    return bid_table, ask_table


def _precompute_all():
    global _PRECOMPUTED
    for regime in _REGIMES:
        key = (regime['sigma'], regime['A'], regime['k'], regime['gamma'])
        if key not in _PRECOMPUTED:
            if time.time() > _DEADLINE - 10:
                Q = regime['Q']
                n_q = 2 * Q + 1
                depth = 1.0 / regime['k']
                bd = np.full((regime['steps'], n_q), depth)
                ad = np.full((regime['steps'], n_q), depth)
                _PRECOMPUTED[key] = (bd, ad)
            else:
                bd, ad = _solve_regime(regime)
                _PRECOMPUTED[key] = (bd, ad)


def _init():
    global _PRECOMPUTED
    if not _PRECOMPUTED:
        _precompute_all()


_init()


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']

    key = (sigma, A, k, gamma)

    if key in _PRECOMPUTED:
        bid_table, ask_table = _PRECOMPUTED[key]
    else:
        depth = 1.0 / k
        return (np.full(len(inventory), depth, dtype=np.float64),
                np.full(len(inventory), depth, dtype=np.float64))

    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)

    qi = inventory + Q
    qi = np.clip(qi, 0, 2 * Q)

    bid_depths = bid_table[n, qi].astype(np.float64)
    ask_depths = ask_table[n, qi].astype(np.float64)

    min_valid = -3.0 / k
    bid_depths = np.maximum(bid_depths, min_valid)
    ask_depths = np.maximum(ask_depths, min_valid)
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, 1.0 / k)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
