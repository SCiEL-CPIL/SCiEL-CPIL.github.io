# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit

    # Parse inputs
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0

    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(16) / 16.0

    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]

    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]

    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2

    def solve_single_item(values, joint_probs):
        """
        Solve single-item auction for one good.
        values: array of length 16 with the value for this good at each type
        joint_probs: array of length 16 with joint probabilities
        
        Returns: allocation (16,), payments (16,)
        """
        # Marginal distribution over values for this good
        # Values are in {0,1,2,3}
        marginal = np.zeros(4)
        for v in range(4):
            mask = (values == v)
            marginal[v] = np.sum(joint_probs[mask])

        # For a single item with values in {0,1,2,3}, IC requires:
        # x(v) is non-decreasing in v
        # Payment identity: p(v) = v*x(v) - sum_{v'=0}^{v-1} x(v')
        # (with the convention that the integral of x over [0,v] for discrete values
        #  is sum of x(v') for v' < v)
        # IR: p(v) <= v*x(v) for all v, and p(v) >= 0
        # Actually IR: v*x(v) - p(v) >= 0, i.e., p(v) <= v*x(v)
        # And p(v) >= 0
        
        # Variables: x_0, x_1, x_2, x_3 (allocations for each value level)
        # Constraints:
        #   0 <= x_v <= 1 for all v
        #   x_0 <= x_1 <= x_2 <= x_3 (monotonicity)
        #   p_v = v*x_v - sum_{v'<v} x_{v'} >= 0
        #   p_v = v*x_v - sum_{v'<v} x_{v'} <= v*x_v (always true since sum >= 0)
        
        # Revenue: sum_v marginal[v] * p_v
        # = sum_v marginal[v] * (v*x_v - sum_{v'<v} x_{v'})
        
        # Let's expand:
        # p_0 = 0*x_0 = 0
        # p_1 = 1*x_1 - x_0
        # p_2 = 2*x_2 - x_0 - x_1
        # p_3 = 3*x_3 - x_0 - x_1 - x_2
        
        # Revenue = m_0*0 + m_1*(x_1 - x_0) + m_2*(2*x_2 - x_0 - x_1) + m_3*(3*x_3 - x_0 - x_1 - x_2)
        # = -m_1*x_0 - m_2*x_0 - m_3*x_0 + m_1*x_1 - m_2*x_1 - m_3*x_1 + 2*m_2*x_2 - m_3*x_2 + 3*m_3*x_3
        # = -(m_1+m_2+m_3)*x_0 + (m_1-m_2-m_3)*x_1 + (2*m_2-m_3)*x_2 + 3*m_3*x_3
        
        # Maximize revenue subject to:
        #   x_0 <= x_1, x_1 <= x_2, x_2 <= x_3
        #   x_v >= 0, x_v <= 1
        #   p_v >= 0:
        #     p_1 = x_1 - x_0 >= 0 (same as x_0 <= x_1)
        #     p_2 = 2*x_2 - x_0 - x_1 >= 0
        #     p_3 = 3*x_3 - x_0 - x_1 - x_2 >= 0
        
        c = np.array([
            (m_1 + m_2 + m_3),  # coefficient for x_0 (negative for maximization)
            (m_2 + m_3 - m_1),  # coefficient for x_1
            (m_3 - 2*m_2),      # coefficient for x_2
            -3*m_3              # coefficient for x_3 (negative for maximization)
        ])
        # Wait, linprog minimizes. So we negate the revenue coefficients.
        # Revenue coeff: [-(m1+m2+m3), (m1-m2-m3), (2*m2-m3), 3*m3]
        # Minimize: [m1+m2+m3, m2+m3-m1, m3-2*m2, -3*m3]
        
        c = np.array([
            marginal[1] + marginal[2] + marginal[3],
            marginal[2] + marginal[3] - marginal[1],
            marginal[3] - 2*marginal[2],
            -3*marginal[3]
        ])
        
        # Constraints:
        # x_0 - x_1 <= 0
        # x_1 - x_2 <= 0
        # x_2 - x_3 <= 0
        # -x_0 - x_1 + 2*x_2 >= 0  =>  x_0 + x_1 - 2*x_2 <= 0
        # -x_0 - x_1 - x_2 + 3*x_3 >= 0  =>  x_0 + x_1 + x_2 - 3*x_3 <= 0
        
        A_ub = np.array([
            [1, -1, 0, 0],
            [0, 1, -1, 0],
            [0, 0, 1, -1],
            [1, 1, -2, 0],
            [1, 1, 1, -3]
        ])
        b_ub = np.array([0, 0, 0, 0, 0])
        
        bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})
        
        if not result.success or result.x is None:
            # Fallback: all zero allocation
            x = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            x = result.x
        
        # Compute payments
        p = np.zeros(4)
        p[0] = 0.0
        p[1] = x[1] - x[0]
        p[2] = 2*x[2] - x[0] - x[1]
        p[3] = 3*x[3] - x[0] - x[1] - x[2]
        
        # Ensure non-negative payments
        p = np.maximum(p, 0.0)
        
        # Map back to 16 types
        alloc_16 = np.zeros(16)
        pay_16 = np.zeros(16)
        for i in range(16):
            v = int(values[i])
            alloc_16[i] = x[v]
            pay_16[i] = p[v]
        
        return alloc_16, pay_16

    def verify_and_fix(allocations, payments):
        """Verify IR and IC, fix small violations, return (valid, allocations, payments)."""
        allocations = allocations.copy()
        payments = payments.copy()
        
        # Fix IR violations
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -1e-8:
                payments[i] = max(0.0, np.dot(V[i], allocations[i]))
        
        # Check IC and fix if needed
        max_violation = 0.0
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                viol = util_report - util_truth
                if viol > max_violation:
                    max_violation = viol
        
        if max_violation > 1e-8:
            for iteration in range(10):
                if time.time() > deadline - 5.0:
                    break
                max_violation = 0.0
                violated = False
                for i in range(16):
                    for ip in range(16):
                        util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                        util_truth = np.dot(V[i], allocations[i]) - payments[i]
                        viol = util_report - util_truth
                        if viol > 1e-8:
                            violated = True
                            payments[ip] += viol + 1e-9
                            if payments[ip] < 0:
                                payments[ip] = 0.0
                if not violated:
                    break
            
            max_violation = 0.0
            for i in range(16):
                for ip in range(16):
                    util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                    util_truth = np.dot(V[i], allocations[i]) - payments[i]
                    viol = util_report - util_truth
                    if viol > max_violation:
                        max_violation = viol
            
            if max_violation > 1e-8:
                return False, np.zeros((16, 2)), np.zeros(16)
            
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    val = np.dot(V[i], allocations[i])
                    if val > 0:
                        scale = max(0.0, payments[i]) / val
                        if scale < 1.0:
                            allocations[i] *= scale
                    else:
                        allocations[i] = 0.0
                    payments[i] = max(0.0, np.dot(V[i], allocations[i]))
            
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    return False, np.zeros((16, 2)), np.zeros(16)
            
            for i in range(16):
                for ip in range(16):
                    util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                    util_truth = np.dot(V[i], allocations[i]) - payments[i]
                    if util_report - util_truth > 1e-8:
                        return False, np.zeros((16, 2)), np.zeros(16)
        
        return True, allocations, payments

    try:
        # Solve each good independently
        alloc_0, pay_0 = solve_single_item(V[:, 0], probs)
        alloc_1, pay_1 = solve_single_item(V[:, 1], probs)
        
        allocations = np.column_stack([alloc_0, alloc_1])
        payments = pay_0 + pay_1
        
        # Clamp to valid ranges
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        valid, allocations, payments = verify_and_fix(allocations, payments)
        
        if not valid:
            # Fallback: simple valid mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
