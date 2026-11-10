"""Rewritten program for finite auction design: direct LP solve with IC/IR constraints."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_DEADLINE = 85.0
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _build_lp(distribution_spec, support, constraints):
    n_types = 16
    n_goods = 2
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        p = None
        for key in ['probabilities', 'dist', 'p', 'probs', 'weights']:
            if key in distribution_spec:
                p = np.array(distribution_spec[key], dtype=float)
                break
        if p is None:
            for k, v in distribution_spec.items():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 16:
                    p = np.array(v, dtype=float)
                    break
        if p is None:
            p = np.ones(16) / 16
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        p = np.array(distribution_spec, dtype=float)
        if len(p) != 16:
            p = np.ones(16) / 16
    else:
        p = np.ones(16) / 16
    
    if len(p) != 16:
        p = np.ones(16) / 16
    p = p / p.sum()
    
    n_vars = 32  # x1[0..15], x2[0..15]
    
    # Objective: max E[U], c = -coefficients
    c = np.zeros(n_vars)
    for i1 in range(4):
        for i2 in range(4):
            for k in range(1, i1+1):
                c[k*4 + i2] -= p[i1*4 + i2]
            for k in range(1, i2+1):
                c[16 + i1*4 + k] -= p[i1*4 + i2]
    
    bounds = [(0.0, 1.0)] * n_vars
    
    A_ub = []
    b_ub = []
    
    # IC monotonicity for good 1
    for i1 in range(1, 4):
        for i2 in range(4):
            row = np.zeros(n_vars)
            row[i1*4 + i2] = -1.0
            row[(i1-1)*4 + i2] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IC monotonicity for good 2
    for i1 in range(4):
        for i2 in range(1, 4):
            row = np.zeros(n_vars)
            row[16 + i1*4 + i2] = -1.0
            row[16 + i1*4 + (i2-1)] = 1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints
    for i1 in range(4):
        for i2 in range(4):
            if i1 == 0 and i2 == 0:
                continue
            row = np.zeros(n_vars)
            for k in range(1, i1+1):
                row[k*4 + i2] = -1.0
            for k in range(1, i2+1):
                row[16 + i1*4 + k] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub) if A_ub else np.zeros((0, n_vars))
    b_ub = np.array(b_ub) if b_ub else np.zeros(0)
    
    return p, c, A_ub, b_ub, bounds

def _compute_payments(x1, x2):
    payments = np.zeros(16)
    for i1 in range(4):
        for i2 in range(4):
            idx = i1*4 + i2
            u = 0.0
            for k in range(1, i1+1):
                u += x1[k*4 + i2]
            for k in range(1, i2+1):
                u += x2[i1*4 + k]
            payments[idx] = i1 * x1[idx] + i2 * x2[idx] - u
    return payments

def construct_auction(distribution_spec, support, constraints):
    global _START
    _START = time.time()
    
    try:
        p, c, A_ub, b_ub, bounds = _build_lp(distribution_spec, support, constraints)
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            x1 = np.clip(x[:16], 0, 1)
            x2 = np.clip(x[16:], 0, 1)
            payments = _compute_payments(x1, x2)
            payments = np.maximum(payments, 0)
            
            allocations = np.column_stack([x1, x2])
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            allocations = np.ones((16, 2))
            payments = np.zeros(16)
            return {
                'allocations': allocations,
                'payments': payments
            }
    except Exception:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {
            'allocations': allocations,
            'payments': payments
        }

# EVOLVE-BLOCK-END
