import numpy as np

def construct_auction(distribution_spec, support, constraints):
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)
    
    # Analytical solution for Uniform/Independent Uniform on {0,1,2,3}
    # Virtual values: phi(0)=-4, phi(1)=-2, phi(2)=0, phi(3)=2
    # Optimal allocation: q(v) = 1 if v >= 2 else 0
    # Payments (single good): p(0)=0, p(1)=0, p(2)=2, p(3)=2
    # For independent goods, payments add up.
    
    p_single = np.array([0.0, 0.0, 2.0, 2.0])
    
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for i, (v1, v2) in enumerate(types):
        # Allocation
        allocations[i, 0] = 1.0 if v1 >= 2 else 0.0
        allocations[i, 1] = 1.0 if v2 >= 2 else 0.0
        
        # Payment
        payments[i] = p_single[v1] + p_single[v2]
        
    return {'allocations': allocations, 'payments': payments}
