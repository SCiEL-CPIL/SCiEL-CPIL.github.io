"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for a single additive bidder
    with two goods, each with values in {0, 1, 2, 3}.
    
    Returns:
        dict with keys:
            'allocations': (16, 2) array of allocation probabilities
            'payments': (16,) array of expected payments
    """
    t0 = time.time()
    T_LIMIT = 85.0  # safety margin before 90s kill
    
    def _elapsed():
        return time.time() - t0

    # Type space: 4x4 = 16 types (v1, v2) where v1, v2 in {0, 1, 2, 3}
    values = np.array([0, 1, 2, 3], dtype=np.float64)
    n_values = 4
    n_types = 16
    
    # Map type index to (v1, v2)
    def idx(v1, v2):
        return v1 * n_values + v2

    # Get distribution over types
    probs = _parse_distribution(distribution_spec)
    
    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

    # V matrix: 16 x 2, V[t] = [v1(t), v2(t)]
    V = np.zeros((n_types, 2))
    for t in range(n_types):
        V[t, 0] = values[t // n_values]
        V[t, 1] = values[t % n_values]

    # LP formulation using utility U and allocation q
    # Variables:
    #   q[t, i] for t in 0..15, i in {0, 1}  -> indices 0..31
    #   U[t] for t in 0..15                  -> indices 32..47
    #   p[t] for t in 0..15                  -> indices 48..63
    #
    # Constraints:
    #   1. 0 <= q[t, i] <= 1
    #   2. U[t] >= 0 (IR)
    #   3. p[t] = V[t].q[t] - U[t] (payment consistency, equality)
    #   4. IC: For all t, t':
    #        U[t] - U[t'] - (V[t] - V[t']).q[t'] >= 0
    #        => -U[t] + U[t'] + (V[t]-V[t']).q[t'] <= 0
    #   5. p[t] >= 0 (non-negative payment)
    
    n_q = n_types * 2  # 32
    n_U = n_types      # 16
    n_p = n_types      # 16
    n_vars = n_q + n_U + n_p  # 64

    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_q + n_U + t] = -probs[t]
    
    # Build inequality constraints A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: 16*15 = 240 constraints (excluding t == t')
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            # -U[t]
            row[n_q + t] = -1.0
            # +U[t']
            row[n_q + tp] = 1.0
            # +(V1[t]-V1[t'])*q[t',0] + (V2[t]-V2[t'])*q[t',1]
            row[tp * 2 + 0] = (V[t, 0] - V[tp, 0])
            row[tp * 2 + 1] = (V[t, 1] - V[tp, 1])
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # Non-negative payment: p[t] >= 0 is handled by bounds
    # IR: U[t] >= 0 is handled by bounds
    
    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)
    
    # Equality constraints: payment consistency
    # p[t] - V[t,0]*q[t,0] - V[t,1]*q[t,1] + U[t] = 0
    A_eq = np.zeros((n_types, n_vars))
    b_eq = np.zeros(n_types)
    for t in range(n_types):
        A_eq[t, t * 2 + 0] = -V[t, 0]
        A_eq[t, t * 2 + 1] = -V[t, 1]
        A_eq[t, n_q + t] = 1.0
        A_eq[t, n_q + n_U + t] = 1.0
    
    # Bounds
    bounds = []
    for _ in range(n_q):
        bounds.append((0.0, 1.0))
    for _ in range(n_U):
        bounds.append((0.0, None))
    for _ in range(n_p):
        bounds.append((0.0, None))
    
    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
    
    # Solve LP
    try:
        result = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            A_eq=A_eq, 
            b_eq=b_eq, 
            bounds=bounds, 
            method='highs',
            options={
                'presolve': True,
                'dual_feasibility_tolerance': 1e-12,
                'primal_feasibility_tolerance': 1e-12
            }
        )
        
        if result.success and result.x is not None:
            x = result.x
        else:
            x = _myerson_fallback(probs, values, n_values, n_types)
    except Exception:
        x = _myerson_fallback(probs, values, n_values, n_types)
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for t in range(n_types):
        allocations[t, 0] = x[t * 2 + 0]
        allocations[t, 1] = x[t * 2 + 1]
        payments[t] = x[n_q + n_U + t]
    
    # Clip to ensure numerical validity
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Local refinement: try to increase payments while maintaining IC and IR
    if _elapsed() < T_LIMIT - 5.0:
        payments = _local_refinement(allocations, payments, V, probs)
    
    # Final verification
    if not _verify_ic_ir(allocations, payments, V):
        # If verification fails, return zero mechanism
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
    
    return {
        'allocations': allocations,
        'payments': payments
    }


def _parse_distribution(distribution_spec):
    """Parse distribution specification into probability array of shape (16,)."""
    n_types = 16
    n_values = 4
    values = [0, 1, 2, 3]
    
    def idx(v1, v2):
        return v1 * n_values + v2
    
    try:
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                arr = np.asarray(distribution_spec['probs'], dtype=np.float64)
                if arr.shape == (n_types,):
                    arr = np.maximum(arr, 0.0)
                    s = arr.sum()
                    return arr / s if s > 0 else np.ones(n_types) / n_types
            
            if 'distribution' in distribution_spec:
                d = distribution_spec['distribution']
                if d == 'uniform':
                    return np.ones(n_types) / n_types
                if d == 'geometric':
                    base = 0.7
                    arr = np.array([base**i * base**j for i in range(n_values) for j in range(n_values)])
                    return arr / arr.sum()
            
            if (0, 0) in distribution_spec:
                probs = np.zeros(n_types)
                for (v1, v2), p in distribution_spec.items():
                    probs[idx(v1, v2)] = p
                s = probs.sum()
                if s > 0:
                    return probs / s
                return np.ones(n_types) / n_types
            
            if 0 in distribution_spec:
                probs = np.zeros(n_types)
                for i, p in distribution_spec.items():
                    probs[i] = p
                s = probs.sum()
                if s > 0:
                    return probs / s
                return np.ones(n_types) / n_types
            
            return np.ones(n_types) / n_types
        
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=np.float64).flatten()
            if len(arr) == n_types:
                arr = np.maximum(arr, 0.0)
                s = arr.sum()
                if s > 0:
                    return arr / s
            return np.ones(n_types) / n_types
        
        return np.ones(n_types) / n_types
    except Exception:
        return np.ones(n_types) / n_types


def _myerson_fallback(probs, values, n_values, n_types):
    """Fallback: Myerson-style independent goods with optimal reserves."""
    x = np.zeros(2 * n_types + n_types + n_types)
    
    for good in range(2):
        p_val = np.zeros(n_values)
        for v_idx in range(n_values):
            for v2_idx in range(n_values):
                if good == 0:
                    p_val[v_idx] += probs[v_idx * n_values + v2_idx]
                else:
                    p_val[v_idx] += probs[v2_idx * n_values + v_idx]
        
        best_reserve = 0
        best_rev = 0
        for v_idx, v in enumerate(values):
            p_ge = sum(p_val[vidx] for vidx in range(v_idx, n_values))
            rev = v * p_ge
            if rev > best_rev:
                best_rev = rev
                best_reserve = v
        
        for t in range(n_types):
            if good == 0:
                v_t = t // n_values
            else:
                v_t = t % n_values
            
            v_val = values[v_t]
            if v_val >= best_reserve:
                x[t * 2 + good] = 1.0
            else:
                x[t * 2 + good] = 0.0
    
    for t in range(n_types):
        v1_t = values[t // n_values]
        v2_t = values[t % n_values]
        q1 = x[t * 2 + 0]
        q2 = x[t * 2 + 1]
        x[2 * n_types + n_types + t] = v1_t * q1 + v2_t * q2
    
    return x


def _local_refinement(allocations, payments, V, probs):
    """Local refinement: try to increase payments while maintaining IC and IR."""
    n_types = len(payments)
    p = payments.copy()
    x = allocations[:, 0]
    y = allocations[:, 1]
    
    for _round in range(5):
        improved = False
        for i in range(n_types):
            # Maximum p[i] such that IR holds: p[i] <= x[i]*v1[i] + y[i]*v2[i]
            max_p_i = x[i] * V[i, 0] + y[i] * V[i, 1]
            
            # Check IC constraints involving p[i]
            # For type i reporting as j:
            # ur = x[j]*V[i,0] + y[j]*V[i,1] - p[j]
            # ut = x[i]*V[i,0] + y[i]*V[i,1] - p[i]
            # IC: ur <= ut
            # => x[j]*V[i,0] + y[j]*V[i,1] - p[j] <= x[i]*V[i,0] + y[i]*V[i,1] - p[i]
            # => p[i] <= p[j] + x[i]*V[i,0] + y[i]*V[i,1] - x[j]*V[i,0] - y[j]*V[i,1]
            
            new_max_p_i = max_p_i
            for j in range(n_types):
                if i == j:
                    continue
                upper_j = p[j] + x[i] * V[i, 0] + y[i] * V[i, 1] - x[j] * V[i, 0] - y[j] * V[i, 1]
                new_max_p_i = min(new_max_p_i, upper_j)
            
            new_p_i = max(0.0, min(new_max_p_i, max_p_i))
            if new_p_i > p[i] + 1e-12:
                p[i] = new_p_i
                improved = True
        
        if not improved:
            break
    
    return p


def _verify_ic_ir(allocations, payments, V):
    """Verify that the mechanism satisfies IC and IR constraints."""
    n_types = len(payments)
    x = allocations[:, 0]
    y = allocations[:, 1]
    
    for i in range(n_types):
        # IR: utility >= 0
        ut = x[i] * V[i, 0] + y[i] * V[i, 1] - payments[i]
        if ut < -1e-8:
            return False
        
        # IC: for all j, utility when reporting j <= utility when reporting truthfully
        for j in range(n_types):
            if i == j:
                continue
            ur = x[j] * V[i, 0] + y[j] * V[i, 1] - payments[j]
            if ur - ut > 1e-8:
                return False
    
    return True

# EVOLVE-BLOCK-END
