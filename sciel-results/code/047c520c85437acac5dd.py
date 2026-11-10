"""Newton-Raphson continuous-depth market-making policy via backward DP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 325.0


def _newton_bid(Vq, Vp, Vm, A, k, dt, penalty):
    """
    Solve for optimal bid depth db given inventory state.
    Vq = V(q), Vp = V(q+1), Vm = V(q-1)
    The objective for bid side (independent approximation):
      f(db) = p(db)*db + p(db)*(Vp - Vq) - penalty + Vq
    where p(db) = 1 - exp(-A*exp(-k*db)*dt)
    
    Actually the full joint objective is:
      J(db, da) = p_b(db)*db + p_a(da)*da 
                  + p_b(db)*(V(q-1)-V(q)) + p_a(da)*(V(q+1)-V(q))
                  - p_b(db)*p_a(da)*(V(q+1)-2*V(q)+V(q-1))
                  - penalty + V(q)
    
    For the independent case (ignoring cross term):
      bid: maximize p(db)*db + p(db)*(Vm - Vq)
      ask: maximize p(da)*da + p(da)*(Vp - Vq)
    
    Let's use the full joint approach with Newton on the coupled system.
    
    Actually, let's do it properly. The value function update is:
      V(q) = max_{db,da >= 0} [ p_b(db)*db + p_a(da)*da 
              + p_b(db)*(V(q-1)-V(q_next)) ... ]
    
    Wait, I need to be more careful. Let me re-derive.
    
    At step n, with inventory q:
    - We choose bid depth db and ask depth da
    - Bid fills with prob pb = 1-exp(-A exp(-k db) dt), ask with pa = 1-exp(-A exp(-k da) dt)
    - If bid fills: earn db, inventory goes to q-1
    - If ask fills: earn da, inventory goes to q+1
    - Both fill: earn db+da, inventory goes to q (net zero)
    - Neither: earn 0, inventory stays q
    - Penalty: (gamma/2)*sigma^2*q^2*dt
    
    V_n(q) = max_{db,da} [ pb*db + pa*da 
              + pb*(1-pa)*V_{n+1}(q-1) + pa*(1-pb)*V_{n+1}(q+1) 
              + pb*pa*V_{n+1}(q) + (1-pb)*(1-pa)*V_{n+1}(q)
              - penalty + ... ]
    
    Wait, let me simplify. The expected continuation value:
    E[V_{n+1}] = (1-pb)(1-pa) V_{n+1}(q) + pb(1-pa) V_{n+1}(q-1) + (1-pb)pa V_{n+1}(q+1) + pb*pa V_{n+1}(q)
    
    = V_{n+1}(q) + pb(1-pa)(V_{n+1}(q-1) - V_{n+1}(q)) + pa(1-pb)(V_{n+1}(q+1) - V_{n+1}(q))
    
    Let me define:
    dM = V_{n+1}(q-1) - V_{n+1}(q)   (delta minus)
    dP = V_{n+1}(q+1) - V_{n+1}(q)   (delta plus)
    
    E[V_{n+1}] = V_{n+1}(q) + pb*dM + pa*dP - pb*pa*(dM + dP)
    
    Wait: pb(1-pa)*dM + pa(1-pb)*dP = pb*dM + pa*dP - pb*pa*dM - pb*pa*dP
    So E[V_{n+1}] = V_{n+1}(q) + pb*dM + pa*dP - pb*pa*(dM + dP)
    
    Objective:
    J(db, da) = pb*db + pa*da + pb*dM + pa*dP - pb*pa*(dM+dP) - penalty + V_{n+1}(q)
    
    = pb*(db + dM) + pa*(da + dP) - pb*pa*(dM + dP) - penalty + V_{n+1}(q)
    
    For interior states (not at boundaries), we optimize over db, da >= 0.
    
    Taking partial derivatives:
    dJ/ddb = (dpb/db)*(db + dM) + pb - (dpb/db)*pa*(dM + dP) = 0
    dJ/dda = (dpa/da)*(da + dP) + pa - (dpa/da)*pb*(dM + dP) = 0
    
    where dpb/db = A*k*exp(-k*db)*exp(-A*exp(-k*db)*dt) = k * pb * (1-pb) / (A*dt) ... 
    
    Actually: pb = 1 - exp(-A*exp(-k*db)*dt)
    dpb/db = A * k * exp(-k*db) * exp(-A*exp(-k*db)*dt) = k * A * exp(-k*db) * (1 - pb)
    
    Let me define:
    r_b = A * exp(-k*db) * dt  (so pb = 1 - exp(-r_b))
    dpb/db = k * (1 - pb) * r_b / dt * ... 
    
    Hmm, let me just compute numerically. Let me use a different approach.
    
    Let x = exp(-k*db), so db = -ln(x)/k, x in (0, 1]
    pb = 1 - exp(-A*x*dt)
    dpb/db = k*(1-pb)*A*x*dt/dt = k*(1-pb)*A*x
    
    Wait: dpb/dx = A*dt*exp(-A*x*dt) = (1-pb)*A*dt
    dx/db = -k*x
    dpb/db = (1-pb)*A*dt*(-k*x) = -k*A*x*(1-pb)
    
    So dJ/ddb = -k*A*x*(1-pb)*(db + dM) + (1-pb)*A*dt*(1-pa)*(dM+dP) ... 
    
    This is getting complicated. Let me just use scipy or a simple numerical Newton.
    """
    pass  # placeholder


def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Backward DP with Newton-Raphson for continuous depth optimization.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    n_states = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    penalties = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    opt_bid = np.full((steps, n_states), np.inf)
    opt_ask = np.full((steps, n_states), np.inf)

    V_next = np.zeros(n_states)

    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            # Fill remaining with analytical approximation
            for nn in range(n, -1, -1):
                t_curr = nn * dt
                remaining = T - t_curr
                base = 1.0 / k
                skew = gamma * sigma**2 * remaining / (4.0 * k)
                for qi in range(n_states):
                    q = qi - Q
                    bid = base + skew * q
                    ask = base - skew * q
                    bid = max(bid, 1e-4)
                    ask = max(ask, 1e-4)
                    if qi < 2 * Q:  # not at Q
                        opt_bid[nn, qi] = bid
                    if qi > 0:  # not at -Q
                        opt_ask[nn, qi] = ask
            _POLICY_CACHE[key] = (opt_bid, opt_ask)
            return _POLICY_CACHE[key]

        V_curr = np.zeros(n_states)
        opt_bid_n = np.full(n_states, np.inf)
        opt_ask_n = np.full(n_states, np.inf)

        for qi in range(n_states):
            q = qi - Q
            
            if qi == 0:
                # q = -Q: only bid possible
                Vq = V_next[qi]
                Vp = V_next[qi + 1]
                dP = Vp - Vq
                
                # Optimize bid only
                # J(db) = pb*db + pb*dP - penalty + Vq
                # dJ/db = dpb/db*(db + dP) = 0 => db + dP = 0 or dpb/db = 0
                # If dP < 0, optimal db = -dP (if that gives reasonable pb)
                # Otherwise use Newton
                if dP < 0:
                    db_opt = min(-dP, 10.0 / k)
                else:
                    db_opt = 1e-4
                
                # Refine with Newton
                db_opt = _newton_solve_bid(db_opt, dP, A, k, dt)
                opt_bid_n[qi] = db_opt
                pb = 1.0 - np.exp(-A * np.exp(-k * db_opt) * dt)
                V_curr[qi] = pb * (db_opt + dP) - penalties[qi] + Vq

            elif qi == n_states - 1:
                # q = Q: only ask possible
                Vq = V_next[qi]
                Vm = V_next[qi - 1]
                dM = Vm - Vq
                
                if dM < 0:
                    da_opt = min(-dM, 10.0 / k)
                else:
                    da_opt = 1e-4
                
                da_opt = _newton_solve_ask(da_opt, dM, A, k, dt)
                opt_ask_n[qi] = da_opt
                pa = 1.0 - np.exp(-A * np.exp(-k * da_opt) * dt)
                V_curr[qi] = pa * (da_opt + dM) - penalties[qi] + Vq

            else:
                # Interior: both bid and ask
                Vq = V_next[qi]
                Vp = V_next[qi + 1]
                Vm = V_next[qi - 1]
                dP = Vp - Vq
                dM = Vm - Vq
                cross = dM + dP  # = Vp - 2*Vq + Vm... wait no
                # cross term coefficient: pb*pa*(dM + dP) but with negative sign in objective
                # Actually: -pb*pa*(dM + dP) where dM = Vm-Vq, dP = Vp-Vq
                # So the cross coefficient is -(dM + dP)
                
                # Solve coupled Newton for (db, da)
                db_init = 1.0 / k
                da_init = 1.0 / k
                
                db_opt, da_opt = _newton_solve_coupled(db_init, da_init, dM, dP, A, k, dt)
                
                opt_bid_n[qi] = db_opt
                opt_ask_n[qi] = da_opt
                
                pb = 1.0 - np.exp(-A * np.exp(-k * db_opt) * dt)
                pa = 1.0 - np.exp(-A * np.exp(-k * da_opt) * dt)
                
                V_curr[qi] = pb * (db_opt + dM) + pa * (da_opt + dP) - pb * pa * (dM + dP) - penalties[qi] + Vq

        opt_bid[n] = opt_bid_n
        opt_ask[n] = opt_ask_n
        V_next = V_curr

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def _newton_solve_bid(db, dM, A, k, dt, max_iter=20, tol=1e-10):
    """Newton for single-side bid optimization.
    J(db) = pb*(db + dM) where pb = 1 - exp(-A*exp(-k*db)*dt)
    dJ/db = dpb/db * (db + dM) = 0
    So optimal: db + dM = 0 (if dM < 0) or db -> 0 (if dM >= 0)
    """
    if dM >= 0:
        return 1e-4
    # db* = -dM, but need to check if this is a max
    db_star = min(-dM, 10.0 / k)
    db_star = max(db_star, 1e-4)
    return db_star


def _newton_solve_ask(da, dP, A, k, dt, max_iter=20, tol=1e-10):
    """Newton for single-side ask optimization."""
    if dP >= 0:
        return 1e-4
    da_star = min(-dP, 10.0 / k)
    da_star = max(da_star, 1e-4)
    return da_star


def _newton_solve_coupled(db, da, dM, dP, A, k, dt, max_iter=30, tol=1e-10):
    """
    Solve the coupled Newton system for (db, da).
    
    J(db, da) = pb*(db + dM) + pa*(da + dP) - pb*pa*(dM + dP)
    
    dJ/db = dpb/db * (db + dM) - dpb/db * pa * (dM + dP) = dpb/db * (db + dM - pa*(dM+dP))
    dJ/da = dpa/da * (da + dP) - dpa/da * pb * (dM + dP) = dpa/da * (da + dP - pb*(dM+dP))
    
    Setting to zero:
    db + dM - pa*(dM+dP) = 0   ... (1)
    da + dP - pb*(dM+dP) = 0   ... (2)
    
    where pb = 1-exp(-A exp(-k db) dt), pa = 1-exp(-A exp(-k da) dt)
    
    This is a 2x2 nonlinear system. Use Newton-Raphson.
    """
    cross = dM + dP
    
    for _ in range(max_iter):
        pb = 1.0 - np.exp(-A * np.exp(-k * db) * dt)
        pa = 1.0 - np.exp(-A * np.exp(-k * da) * dt)
        
        # Residuals
        r1 = db + dM - pa * cross
        r2 = da + dP - pb * cross
        
        if abs(r1) < tol and abs(r2) < tol:
            break
        
        # Jacobian
        # dr1/db = 1, dr1/da = -dpb/da * cross = 0 (pa doesn't depend on db... wait)
        # r1 = db + dM - pa*(dM+dP), pa depends on da only
        # dr1/db = 1, dr1/da = -dpa/da * cross
        # r2 = da + dP - pb*(dM+dP), pb depends on db only
        # dr2/db = -dpb/db * cross, dr2/da = 1
        
        dpb_db = k * A * np.exp(-k * db) * (1.0 - pb)
        dpa_da = k * A * np.exp(-k * da) * (1.0 - pa)
        
        J11 = 1.0
        J12 = -dpa_da * cross
        J21 = -dpb_db * cross
        J22 = 1.0
        
        det = J11 * J22 - J12 * J21
        if abs(det) < 1e-15:
            break
        
        # Newton step
        step_db = (r1 * J22 - r2 * J12) / det
        step_da = (r2 * J11 - r1 * J21) / det
        
        # Damping to ensure positivity
        alpha = 1.0
        for _ in range(20):
            new_db = db - alpha * step_db
            new_da = da - alpha * step_da
            if new_db > 1e-6 and new_da > 1e-6:
                db, da = new_db, new_da
                break
            alpha *= 0.5
        else:
            break
    
    db = max(db, 1e-4)
    da = max(da, 1e-4)
    db = min(db, 10.0 / k)
    da = min(da, 10.0 / k)
    return db, da


def _analytical_quote(t, inventory, mid, params):
    """Avellaneda-Stoikov style analytical fallback."""
    sigma = params["sigma"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    dt = T / steps
    remaining = T - t
    base = 1.0 / k
    skew_coeff = gamma * sigma**2 * remaining / 4.0
    c = skew_coeff / k
    bid_depths = base + c * inventory
    ask_depths = base - c * inventory
    bid_depths = np.clip(bid_depths, 1e-4, 10.0 / k)
    ask_depths = np.clip(ask_depths, 1e-4, 10.0 / k)
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with analytical fallback."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    n = int(round(t / (T / steps)))
    n = min(max(n, 0), steps - 1)

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        if time.time() < _DEADLINE - 10.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except Exception:
                pass

    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx].copy()
        ask_depths = opt_ask[n, q_idx].copy()
        bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 1e-4), bid_depths, np.inf)
        ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 1e-4), ask_depths, np.inf)
        return (np.asarray(bid_depths, dtype=np.float64),
                np.asarray(ask_depths, dtype=np.float64))
    else:
        return _analytical_quote(t, inventory, mid, params)
# EVOLVE-BLOCK-END
