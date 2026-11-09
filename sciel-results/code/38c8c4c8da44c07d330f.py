"""
Strategy: Direct Integer Linear Programming (ILP) using scipy.optimize.milp.
Instead of graph-based clique search, we formulate the problem as a 0/1 ILP.
Variables: x_i in {0,1} for i in [2, 256].
Objective: Maximize sum(1/i * x_i).
Constraints: x_i + x_j <= 1 for all pairs (i,j) where lcm(i,j) <= 256.
This leverages the HiGHS solver's branch-and-cut capabilities, which often
outperform heuristic graph searches on dense constraint sets.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 255  # Numbers 2 to 256
    numbers = list(range(2, 257))
    
    # 1. Build conflict graph adjacency bitmasks
    # adj_mask[i] is an integer with bit j set if lcm(numbers[i], numbers[j]) <= 256
    adj_mask = [0] * N
    for i in range(N):
        a = numbers[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = numbers[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # 2. Find maximal cliques using Bron-Kerbosch with bitmasks
    cliques = []
    
    def bron_kerbosch(R_mask, P_mask, X_mask):
        if time.time() > deadline - 5.0:
            return
        if P_mask == 0 and X_mask == 0:
            # Extract clique from R_mask
            if R_mask != 0:
                # Count bits
                if bin(R_mask).count("1") > 1:
                    # Convert to list of indices
                    clique_indices = [i for i in range(N) if (R_mask >> i) & 1]
                    cliques.append(clique_indices)
            return
        
        # Choose pivot u from P_mask | X_mask
        # We want to maximize |N(u) & P_mask| to minimize iterations
        U_mask = P_mask | X_mask
        best_u = -1
        max_deg = -1
        
        u_mask = U_mask
        while u_mask:
            lsb = u_mask & -u_mask
            u = lsb.bit_length() - 1
            u_mask ^= lsb
            
            deg = bin(adj_mask[u] & P_mask).count("1")
            if deg > max_deg:
                max_deg = deg
                best_u = u
                
        if best_u == -1:
            return
            
        P_minus_N = P_mask & ~adj_mask[best_u]
        
        v_mask = P_minus_N
        while v_mask:
            lsb = v_mask & -v_mask
            v = lsb.bit_length() - 1
            v_mask ^= lsb
            
            bron_kerbosch(R_mask | lsb, P_mask & adj_mask[v], X_mask & adj_mask[v])
            P_mask = P_mask & ~lsb
            X_mask = X_mask | lsb

    all_nodes_mask = (1 << N) - 1
    bron_kerbosch(0, all_nodes_mask, 0)
    
    # 3. Construct MILP
    num_vars = N
    c = np.zeros(num_vars)
    for i in range(num_vars):
        c[i] = -1.0 / numbers[i]
        
    if cliques:
        num_constraints = len(cliques)
        A_ub = np.zeros((num_constraints, num_vars))
        for k, clique in enumerate(cliques):
            for i in clique:
                A_ub[k, i] = 1.0
        b_ub = np.ones(num_constraints)
    else:
        # Fallback to pairwise constraints
        conflict_pairs = []
        for i in range(N):
            mask = adj_mask[i]
            for j in range(i + 1, N):
                if (mask >> j) & 1:
                    conflict_pairs.append((i, j))
        num_constraints = len(conflict_pairs)
        A_ub = np.zeros((num_constraints, num_vars))
        for k, (i, j) in enumerate(conflict_pairs):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
        b_ub = np.ones(num_constraints)
        
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    integrality = np.ones(num_vars)
    
    if num_constraints == 0:
        return numbers[:]
        
    constraints = LinearConstraint(A_ub, ub=b_ub)
    
    if time.time() > deadline:
        return _greedy_fallback()
        
    try:
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            return _greedy_fallback()
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": remaining_time}
        )
        
        if res.status in [0, 3]:
            if res.x is not None:
                selected_indices = [i + 2 for i in range(num_vars) if res.x[i] > 0.5]
                if selected_indices and _is_valid(selected_indices):
                    return selected_indices
                    
        return _greedy_fallback()
    except Exception:
        return _greedy_fallback()

def _is_valid(subset):
    """Verify that the subset satisfies lcm(a,b) > 256 for all distinct pairs."""
    if not subset:
        return False
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            a = subset[i]
            b = subset[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                return False
    return True

def _greedy_fallback():
    """
    Simple greedy fallback if ILP fails or times out without a good incumbent.
    """
    start_time = time.time()
    deadline = start_time + 5.0
    
    candidates = list(range(2, 257))
    candidates.sort(key=lambda x: -1.0/x)
    
    result = []
    for num in candidates:
        if time.time() > deadline:
            break
        compatible = True
        for existing in result:
            g = gcd(num, existing)
            if num * existing // g <= 256:
                compatible = False
                break
        if compatible:
            result.append(num)
            
    return result if result else [2]

# EVOLVE-BLOCK-END
