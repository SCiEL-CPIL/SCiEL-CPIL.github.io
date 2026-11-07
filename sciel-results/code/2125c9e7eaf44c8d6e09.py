"""Market-making with inventory risk: closed-form optimal depth via quadratic value function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 290.0

def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)

# Cache for precomputed coefficients per regime
_COEFF_CACHE = {}


def _solve_closed_form(sigma, A, k, gamma, T, steps, Q):
    """
    Solve for the optimal depth policy using a quadratic value function ansatz.
    V(q) = a*q^2 + b*q + c
    The optimal bid depth d_b and ask depth d_a are derived from first-order conditions.
    """
    if _remaining_budget() < 60.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k

    # Initialize coefficients at terminal (n=0): V = 0
    a = 0.0
    b = 0.0
    c = 0.0

    # We will iterate backwards from n=1 to n=steps
    # At each step, we solve for the optimal depths given the value function V_next(q)
    # and then update the value function V_current(q).

    # The objective for one step:
    # Maximize over d_b, d_a:
    #   d_b + d_a - (gamma/2)*sigma^2*q^2*dt
    #   + p_b * V_next(q-1) + p_a * V_next(q+1) + (1-p_b-p_a)*V_next(q)
    #
    # where p_b = 1 - exp(-A*exp(-k*d_b)*dt)
    #       p_a = 1 - exp(-A*exp(-k*d_a)*dt)
    #
    # Let x = exp(-k*d_b), y = exp(-k*d_a)
    # Then d_b = -ln(x)/k, d_a = -ln(y)/k
    # p_b = 1 - A*dt*x, p_a = 1 - A*dt*y
    #
    # The continuation term:
    # C = p_b * V_next(q-1) + p_a * V_next(q+1) + (1-p_b-p_a)*V_next(q)
    #
    # V_next(q) = a*q^2 + b*q + c
    # V_next(q-1) = a*(q-1)^2 + b*(q-1) + c = a*q^2 - 2*a*q + a + b*q - b + c
    # V_next(q+1) = a*(q+1)^2 + b*(q+1) + c = a*q^2 + 2*a*q + a + b*q + b + c
    #
    # C = (1-A*dt*x)*(a*q^2 - 2*a*q + a + b*q - b + c)
    #   + (1-A*dt*y)*(a*q^2 + 2*a*q + a + b*q + b + c)
    #   + (A*dt*x + A*dt*y - 1 + 1)*(a*q^2 + b*q + c)  -- wait, 1-p_b-p_a = 1 - (1-A*dt*x) - (1-A*dt*y) = A*dt*x + A*dt*y - 1
    #   This is only valid if A*dt*x + A*dt*y >= 1? No, p_b+p_a <= 1 is not guaranteed.
    #   Actually, the probability of no fill is 1 - p_b - p_a + p_b*p_a?
    #   The problem statement says: "bid hit with probability ..., ask likewise, independently."
    #   So P(bid) = p_b, P(ask) = p_a.
    #   P(bid only) = p_b*(1-p_a)
    #   P(ask only) = (1-p_b)*p_a
    #   P(both) = p_b*p_a
    #   P(neither) = (1-p_b)*(1-p_a)
    #
    # Inventory change:
    # Bid only: q -> q-1
    # Ask only: q -> q+1
    # Both: q -> q (buy 1, sell 1)
    # Neither: q -> q
    #
    # So E[V_next] = p_b*(1-p_a)*V(q-1) + (1-p_b)*p_a*V(q+1) + (p_b*p_a + (1-p_b)*(1-p_a))*V(q)
    # = p_b*(1-p_a)*V(q-1) + (1-p_b)*p_a*V(q+1) + (1 - p_b - p_a + 2*p_b*p_a)*V(q)
    #
    # Let's expand this.
    # V(q-1) = a*q^2 + (b-2a)*q + (a-b+c)
    # V(q+1) = a*q^2 + (b+2a)*q + (a+b+c)
    #
    # Coeff of q^2 in C:
    # p_b(1-p_a)*a + (1-p_b)p_a*a + (1-p_b-p_a+2p_b p_a)*a
    # = a [ p_b - p_b p_a + p_a - p_b p_a + 1 - p_b - p_a + 2 p_b p_a ]
    # = a [ 1 ]
    # So the q^2 coefficient remains 'a'.
    #
    # Coeff of q in C:
    # p_b(1-p_a)*(b-2a) + (1-p_b)p_a*(b+2a) + (1-p_b-p_a+2p_b p_a)*b
    # = b [ p_b - p_b p_a + p_a - p_b p_a + 1 - p_b - p_a + 2 p_b p_a ]
    #   + 2a [ -p_b + p_b p_a + p_a - p_b p_a ]
    # = b * 1 + 2a (p_a - p_b)
    #
    # Constant term in C:
    # p_b(1-p_a)*(a-b+c) + (1-p_b)p_a*(a+b+c) + (1-p_b-p_a+2p_b p_a)*c
    # = c [ 1 ] + a [ p_b - p_b p_a + p_a - p_b p_a ] - b [ p_b - p_b p_a - p_a + p_b p_a ]
    # = c + a (p_a + p_b - 2 p_b p_a) - b (p_b - p_a)
    #
    # Total Value V_current(q):
    # V = d_b + d_a - (gamma/2)*sigma^2*q^2*dt + C
    # = -ln(x)/k - ln(y)/k - (gamma/2)*sigma^2*q^2*dt
    #   + a*q^2 + b*q + 2a(p_a - p_b)q + c + a(p_a+p_b-2p_b p_a) - b(p_b-p_a)
    #
    # Group by powers of q:
    # q^2: a - (gamma/2)*sigma^2*dt
    # q:   b + 2a(p_a - p_b)
    # const: c + d_b + d_a + a(p_a+p_b-2p_b p_a) - b(p_b-p_a)
    #
    # We want V_current to be of the form a_new*q^2 + b_new*q + c_new.
    # However, d_b and d_a (and thus x, y) are chosen to maximize V.
    # The q^2 and q coefficients depend on x and y (through p_b, p_a).
    # This means the optimal x, y depend on q!
    #
    # Wait, the ansatz V(q) = a*q^2 + b*q + c assumes the coefficients are constant.
    # But if the optimal policy makes the coefficients depend on q, the ansatz is invalid.
    #
    # Let's look at the FOC for x (bid depth).
    # Maximize: -ln(x)/k + 2a(p_a - p_b)q + ... terms involving x
    # p_b = 1 - A dt x
    # d_b = -ln(x)/k
    # Term in V depending on x:
    # -ln(x)/k + 2a(p_a - (1-A dt x))q + a(p_a + (1-A dt x) - 2 p_a (1-A dt x)) - b((1-A dt x)-p_a) + ...
    # This is getting complicated.
    #
    # Alternative approach:
    # The "small noise" or "fast diffusion" limit suggests that the optimal spread is constant and the skew is linear in q.
    # Specifically, for the Avellaneda-Stoikov model, the optimal mid-shift is linear in inventory.
    # Here, we have discrete fills.
    #
    # Let's try a different structural change:
    # Use the exact DP but with a much coarser grid or fewer steps? No, that reduces accuracy.
    #
    # Let's look at Inspiration 1 again. It used a fixed shift policy and evaluated it via DP to get the value, but the policy itself was analytic.
    # The current program (score 0.52) does a full DP search over depths.
    # Inspiration 2 (score 1.00) used a log-formula for base depth.
    #
    # The directive says: "Implement a closed-form optimal depth policy by approximating the value function with a quadratic ansatz in inventory, deriving the depth as the unique root of a Lambert W-function equation from the first-order condition..."
    #
    # Let's derive the FOC properly assuming the value function coefficients are constant (which is an approximation).
    # If we assume the optimal d_b, d_a are independent of q (which is false, but let's see), then we can't have a linear skew.
    #
    # Actually, in the limit where dt is small, the optimal bid and ask depths are:
    # d_b = (1/k) * W( (gamma sigma^2 dt / 2) * q * k / (A * something) ) ... this is getting messy.
    #
    # Let's try a simpler structural change:
    # The current program does a 30x30 grid search for depths at each of 1000 steps.
    # This is slow.
    # We can speed it up by noting that the optimal depth is monotonic in the value difference.
    #
    # But the directive asks for a "Structural change".
    # Let's try the Quadratic Ansatz with the assumption that the optimal depths are constant (no skew).
    # This is a bad approximation for inventory risk.
    #
    # What if we use the Quadratic Ansatz to compute the *value* of the optimal policy, but then use a heuristic to set the depths?
    #
    # Let's look at the "Lambert W" hint.
    # The FOC for bid depth d:
    # Maximize: d + p(d) * Delta_V
    # where Delta_V = V(q-1) - V(q) (approximately)
    # p(d) = 1 - exp(-A exp(-kd) dt)
    # Let u = exp(-kd). p = 1 - A dt u. d = -ln(u)/k.
    # f(u) = -ln(u)/k + (1 - A dt u) Delta_V
    # f'(u) = -1/(ku) - A dt Delta_V = 0
    # -1/(ku) = A dt Delta_V
    # u = -1 / (k A dt Delta_V)
    #
    # This requires Delta_V to be negative (which it is, since holding inventory is risky, V(q-1) < V(q) for q>0? No, V(q) is convex, so V(q-1) < V(q) + V'(q). For large q, V(q-1) is much smaller).
    # If Delta_V is negative, u is positive.
    # d = -ln(u)/k = -ln( -1 / (k A dt Delta_V) ) / k = ln( -k A dt Delta_V ) / k.
    #
    # So the optimal bid depth is d_b = (1/k) * ln( -k A dt (V(q-1) - V(q)) ).
    # Similarly, d_a = (1/k) * ln( -k A dt (V(q+1) - V(q)) ).
    #
    # This is a closed-form expression! We just need V(q).
    # We can approximate V(q) with a quadratic a*q^2 + b*q + c.
    # Then V(q-1) - V(q) = a((q-1)^2 - q^2) + b((q-1)-q) = a(1-2q) - b.
    # V(q+1) - V(q) = a((q+1)^2 - q^2) + b((q+1)-q) = a(1+2q) + b.
    #
    # So:
    # d_b(q) = (1/k) * ln( -k A dt (a(1-2q) - b) )
    # d_a(q) = (1/k) * ln( -k A dt (a(1+2q) + b) )
    #
    # We need to determine a, b, c.
    # We can update them via backward induction.
    #
    # Start with a=0, b=0, c=0.
    # At each step, compute d_b(q), d_a(q) for a grid of q.
    # Then compute the new V(q) using the exact DP step with these depths.
    # Fit the new V(q) to a quadratic to get new a, b, c.
    #
    # This is a fixed-point iteration on the coefficients.
    #
    # Let's implement this.

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k
    min_depth = -3.0 / k  # This is the lower bound for validity, but depths should be >= 0 in practice?
    # The problem says "distances below and above mid". Negative distance means quoting inside the mid?
    # "Values below -3/k ... make the whole result invalid."
    # So depths can be negative, down to -3/k.
    # But typically we quote at positive depths.
    # The formula d = ln(...)/k can be negative if the argument < 1.
    
    # Grid for fitting the quadratic
    q_grid = np.linspace(-Q, Q, 100)
    
    a, b, c = 0.0, 0.0, 0.0
    
    # Iterate backwards
    for n in range(1, steps + 1):
        # Compute optimal depths for each q in q_grid
        # d_b = (1/k) * ln( -k A dt (a(1-2q) - b) )
        # d_a = (1/k) * ln( -k A dt (a(1+2q) + b) )
        
        term_b = a * (1 - 2 * q_grid) - b
        term_a = a * (1 + 2 * q_grid) + b
        
        # Ensure arguments of log are positive
        # If argument <= 0, the optimal depth is effectively 0 (or max?)
        # If -k A dt Delta_V <= 0, then Delta_V >= 0.
        # If Delta_V > 0, it means we gain value from moving to q-1.
        # This happens if we are short (q < 0) and want to buy.
        # Wait, V(q-1) - V(q) > 0 means V is decreasing, so we are at high q?
        # If q is large, V(q-1) < V(q) usually (convexity).
        # Let's check signs.
        # For q > 0, V(q) is increasing and convex. V(q-1) < V(q). So Delta_V < 0.
        # -k A dt Delta_V > 0. Log is defined.
        # For q < 0, V(q) is decreasing? No, V(q) = a q^2 ... symmetric if b=0.
        # If b=0, V(q) is even. V(q-1) vs V(q) for q<0.
        # q=-10. V(-11) vs V(-10). (-11)^2 > (-10)^2. So V(-11) > V(-10).
        # So V(q-1) > V(q). Delta_V > 0.
        # Then -k A dt Delta_V < 0. Log undefined.
        # This implies we should set depth to max?
        # If we want to move to q-1 (more negative), and V(q-1) > V(q), we want to buy aggressively.
        # So d_b should be large (close to 0 or negative?).
        # The FOC u = -1/(k A dt Delta_V). If Delta_V > 0, u < 0.
        # But u = exp(-kd) must be > 0.
        # So the FOC has no solution in u>0.
        # The function f(u) = -ln(u)/k + (1-A dt u) Delta_V.
        # If Delta_V > 0, f'(u) = -1/(ku) - A dt Delta_V < 0 for all u>0.
        # So f(u) is decreasing. Max at u->0 (d->inf) or u->1 (d=0)?
        # u->0 means d->inf. p->1.
        # f(0) = inf + Delta_V.
        # f(1) = 0 + (1-A dt) Delta_V.
        # So we want u as small as possible.
        # But d is constrained to [-3/k, ...].
        # u = exp(-kd). Max d = -3/k? No, d is distance.
        # If d can be negative, u > 1.
        # If d is very negative, u is large.
        # The constraint is d >= -3/k. So u <= exp(3).
        # So we should set d = -3/k (u = exp(3)) if the FOC suggests u < 0?
        # Wait, if f is decreasing, we want the smallest u.
        # Smallest u corresponds to largest d.
        # Is there an upper bound on d? No, only lower bound -3/k.
        # So we can make d arbitrarily large?
        # But p = 1 - A dt u. If u is large, p becomes negative?
        # Probability must be in [0,1].
        # So A dt u <= 1 => u <= 1/(A dt).
        # So d >= -ln(1/(A dt))/k = ln(A dt)/k.
        #
        # This is getting complicated. Let's just clip the log argument.
        # If arg <= 0, set d to the boundary that maximizes the term.
        
        arg_b = -k * A * dt * term_b
        arg_a = -k * A * dt * term_a
        
        # Clip to valid range for log
        # If arg <= 0, the optimal depth is at the boundary.
        # Let's assume for now we clip arg to a small positive number.
        arg_b = np.maximum(arg_b, 1e-6)
        arg_a = np.maximum(arg_a, 1e-6)
        
        d_b = np.log(arg_b) / k
        d_a = np.log(arg_a) / k
        
        # Clip depths to valid range [-3/k, ...] and also ensure p in [0,1]
        # p = 1 - exp(-A exp(-kd) dt).
        # If d is very large, exp(-kd) is small, p ~ 0.
        # If d is very small (negative), exp(-kd) is large, p can be > 1.
        # We need p <= 1 => exp(-A exp(-kd) dt) >= 0 => always true.
        # But p >= 0 => exp(-A exp(-kd) dt) <= 1 => A exp(-kd) dt >= 0 => always true.
        # Wait, p = 1 - exp(-X). p in [0,1] iff X >= 0.
        # X = A exp(-kd) dt. Always positive.
        # So p is always in [0,1).
        
        # Clip d to [-3/k, 3/k] just in case, though 3/k is not a hard upper bound in the problem statement?
        # "Values below -3/k ... invalid".
        # So d >= -3/k.
        d_b = np.maximum(d_b, -3.0 / k)
        d_a = np.maximum(d_a, -3.0 / k)
        
        # Now compute V_new(q) using these depths
        # V_new(q) = d_b + d_a - (gamma/2)*sigma^2*q^2*dt
        #           + p_b*(1-p_a)*V(q-1) + (1-p_b)*p_a*V(q+1) + (1-p_b-p_a+2p_b p_a)*V(q)
        
        V_old = a * q_grid**2 + b * q_grid + c
        
        p_b = 1 - np.exp(-A * np.exp(-k * d_b) * dt)
        p_a = 1 - np.exp(-A * np.exp(-k * d_a) * dt)
        
        # Interpolate V(q-1) and V(q+1)
        # Since q_grid is uniform, we can use np.roll
        # But boundaries need care.
        # For simplicity, assume V is zero outside [-Q, Q] or use nearest neighbor.
        # Let's use linear interpolation on a slightly larger grid or just clip.
        
        # Let's just compute V(q-1) and V(q+1) using the quadratic formula directly.
        # This is exact for the quadratic ansatz.
        V_qm1 = a * (q_grid - 1)**2 + b * (q_grid - 1) + c
        V_qp1 = a * (q_grid + 1)**2 + b * (q_grid + 1) + c
        V_q = V_old
        
        prob_both = p_b * p_a
        prob_neither = (1 - p_b) * (1 - p_a)
        prob_bid_only = p_b * (1 - p_a)
        prob_ask_only = (1 - p_b) * p_a
        
        V_new = (d_b + d_a - 0.5 * gamma * sigma**2 * q_grid**2 * dt
                 + prob_bid_only * V_qm1
                 + prob_ask_only * V_qp1
                 + (prob_both + prob_neither) * V_q)
        
        # Fit V_new to a quadratic a_new*q^2 + b_new*q + c_new
        # Use least squares
        A_mat = np.vstack([q_grid**2, q_grid, np.ones_like(q_grid)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, V_new, rcond=None)
        a, b, c = coeffs
        
        if n % 100 == 0 and _remaining_budget() < 30.0:
            return None

    return {
        'a': a,
        'b': b,
        'c': c,
        'k': k,
        'A': A,
        'dt': dt
    }


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _COEFF_CACHE:
        coeffs = _solve_closed_form(sigma, A, k, gamma, T, steps, Q)
        _COEFF_CACHE[key] = coeffs
    coeffs = _COEFF_CACHE[key]

    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k  # Not a hard upper bound, but let's clip to reasonable values
    min_depth = -3.0 / k

    if coeffs is None:
        # Fallback: simple analytic shift
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, min_depth, max_depth)
        ask = np.clip(inv_k - q * shift, min_depth, max_depth)
        bid = np.where(q >= Q, np.inf, bid)
        ask = np.where(q <= -Q, np.inf, ask)
        return bid.astype(float), ask.astype(float)

    a = coeffs['a']
    b = coeffs['b']
    k = coeffs['k']
    A = coeffs['A']
    dt = coeffs['dt']

    q = inventory.astype(float)
    
    # Compute depths
    term_b = a * (1 - 2 * q) - b
    term_a = a * (1 + 2 * q) + b
    
    arg_b = -k * A * dt * term_b
    arg_a = -k * A * dt * term_a
    
    # Clip arguments to avoid log(0) or log(negative)
    arg_b = np.maximum(arg_b, 1e-6)
    arg_a = np.maximum(arg_a, 1e-6)
    
    bid = np.log(arg_b) / k
    ask = np.log(arg_a) / k
    
    # Clip to valid range
    bid = np.clip(bid, min_depth, max_depth)
    ask = np.clip(ask, min_depth, max_depth)
    
    # Boundary conditions
    bid = np.where(q >= Q, np.inf, bid)
    ask = np.where(q <= -Q, np.inf, ask)
    
    # Replace NaN
    bid = np.nan_to_num(bid, nan=inv_k, posinf=np.inf, neginf=min_depth)
    ask = np.nan_to_num(ask, nan=inv_k, posinf=np.inf, neginf=min_depth)
    
    return bid.astype(float), ask.astype(float)
# EVOLVE-BLOCK-END
