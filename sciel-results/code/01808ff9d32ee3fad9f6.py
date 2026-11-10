import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe within 90s
    
    # Parse distribution_spec
    # Assume distribution_spec is a dict or object with 'values' and 'probs'
    # Based on "three fixed distributions", it likely contains the probabilities for each value in {0,1,2,3}
    # for each good, assuming independence or joint distribution.
    # The task says "one additive bidder, two goods, each value in {0,1,2,3}".
    # "three fixed distributions" might mean 3 scenarios or 3 different distributions to average over?
    # "Maximize expected revenue normalized by an independently solved finite-support mechanism LP."
    # Usually, this implies a single distribution.
    # Let's assume distribution_spec is a 4x4 matrix of joint probabilities or two 4-vectors of marginal probabilities.
    # Given "additive bidder", the value is v1 + v2.
    # If independent, P(v1, v2) = P(v1)P(v2).
    
    # Let's infer the structure of distribution_spec.
    # Common formats: 
    # 1. {'probs': [p0, p1, p2, p3]} for each good, assuming independence.
    # 2. A 4x4 matrix.
    # 3. A list of 3 distributions? "three fixed distributions" might refer to the task having 3 test cases.
    # The function is called once per evaluation? "Each evaluation must finish within 60s".
    # "search score is value/seed".
    # I will assume distribution_spec contains the joint probabilities for the 16 reports.
    
    # Let's try to extract probabilities.
    # If it's a dict with 'joint', use that.
    # If it's a dict with 'marginals', compute joint.
    # If it's a numpy array, assume it's 4x4.
    
    probs = None
    if isinstance(distribution_spec, dict):
        if 'joint' in distribution_spec:
            probs = np.array(distribution_spec['joint'])
        elif 'marginals' in distribution_spec:
            m1 = np.array(distribution_spec['marginals'][0])
            m2 = np.array(distribution_spec['marginals'][1])
            probs = np.outer(m1, m2)
        elif 'probs' in distribution_spec:
            # Maybe it's a list of 3 distributions?
            # But the return format is fixed.
            # Let's assume it's a single distribution.
            p = np.array(distribution_spec['probs'])
            if p.shape == (16,):
                probs = p.reshape(4,4)
            elif p.shape == (4,):
                probs = np.outer(p, p)
    elif isinstance(distribution_spec, (np.ndarray, list)):
        arr = np.array(distribution_spec)
        if arr.shape == (4,4):
            probs = arr
        elif arr.shape == (16,):
            probs = arr.reshape(4,4)
        else:
            # Fallback: uniform
            probs = np.ones((4,4)) / 16.0
            
    if probs is None:
        probs = np.ones((4,4)) / 16.0
        
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Values are 0, 1, 2, 3
    values = np.arange(4)
    V1 = values[:, np.newaxis] * np.ones((4,4))
    V2 = np.ones((4,4)) * values[np.newaxis, :]
    
    # Variables: x1_00, x1_01, ..., x1_33, x2_00, ..., x2_33
    # Total 32 variables.
    # Index mapping:
    # idx(i, j, k) where i,j in 0..3, k in 0..1 (0 for good 1, 1 for good 2)
    def idx(i, j, k):
        return i * 4 + j + k * 16
        
    n_vars = 32
    
    # Objective: Maximize sum_v Pr(v) * p(v)
    # p(v) = v1*x1(v) + v2*x2(v) - Integral
    # Integral path: (0,0) -> (i,0) -> (i,j)
    # Int_1(i,j) = sum_{k=0}^{i-1} x1(k,j)
    # Int_2(i,j) = sum_{l=0}^{j-1} x2(i,l)
    # p(i,j) = i*x1(i,j) + j*x2(i,j) - Int_1(i,j) - Int_2(i,j)
    
    # c vector for linprog (minimize -c)
    c = np.zeros(n_vars)
    
    # We need to express the objective in terms of x variables.
    # Obj = sum_{i,j} Pr(i,j) * (i*x1(i,j) + j*x2(i,j) - sum_{k<i} x1(k,j) - sum_{l<j} x2(i,l))
    
    # Coefficient for x1(i,j):
    # From term (i,j): +Pr(i,j)*i
    # From terms (i', j') where i' > i and j' == j: -Pr(i',j) (because x1(i,j) is in Int_1(i',j))
    # Wait, Int_1(i',j) = sum_{k=0}^{i'-1} x1(k,j).
    # So x1(i,j) appears in Int_1(i',j) for all i' > i.
    # Coeff_x1(i,j) = Pr(i,j)*i - sum_{i'=i+1}^{3} Pr(i',j)
    
    # Coefficient for x2(i,j):
    # From term (i,j): +Pr(i,j)*j
    # From terms (i', j') where i' == i and j' > j: -Pr(i,j') (because x2(i,j) is in Int_2(i,j'))
    # Coeff_x2(i,j) = Pr(i,j)*j - sum_{j'=j+1}^{3} Pr(i,j')
    
    for i in range(4):
        for j in range(4):
            coeff1 = probs[i,j] * i
            for ip in range(i+1, 4):
                coeff1 -= probs[ip,j]
            c[idx(i,j,0)] = -coeff1 # Minimize -Obj
            
            coeff2 = probs[i,j] * j
            for jp in range(j+1, 4):
                coeff2 -= probs[i,jp]
            c[idx(i,j,1)] = -coeff2

    # Constraints:
    # 1. 0 <= x <= 1 (handled by bounds)
    # 2. Monotonicity: x1(i,j) <= x1(i',j') if i<=i', j<=j'
    # 3. IR: p(i,j) >= 0
    
    A_ub = []
    b_ub = []
    
    # Monotonicity Constraints
    # For each good k
    for k in [0, 1]:
        for i in range(4):
            for j in range(4):
                for ip in range(i, 4):
                    for jp in range(j, 4):
                        if (i,j) == (ip,jp):
                            continue
                        # x(i,j) - x(ip,jp) <= 0
                        row = np.zeros(n_vars)
                        row[idx(i,j,k)] = 1
                        row[idx(ip,jp,k)] = -1
                        A_ub.append(row)
                        b_ub.append(0)
                        
    # IR Constraints: p(i,j) >= 0 => -p(i,j) <= 0
    # p(i,j) = i*x1(i,j) + j*x2(i,j) - sum_{k<i} x1(k,j) - sum_{l<j} x2(i,l)
    for i in range(4):
        for j in range(4):
            row = np.zeros(n_vars)
            row[idx(i,j,0)] = -i
            row[idx(i,j,1)] = -j
            for k in range(i):
                row[idx(k,j,0)] += 1
            for l in range(j):
                row[idx(i,l,1)] += 1
            A_ub.append(row)
            b_ub.append(0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, 1)] * n_vars
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            # Fallback to uniform or simple heuristic if LP fails
            x = np.ones((16,2)) * 0.5
            # Calculate payments
            p = np.zeros(16)
            for idx_flat in range(16):
                i = idx_flat // 4
                j = idx_flat % 4
                p[idx_flat] = i*x[idx_flat,0] + j*x[idx_flat,1]
            return {"allocations": x, "payments": p}
            
        x_flat = res.x
        x = np.zeros((16,2))
        for idx_flat in range(16):
            i = idx_flat // 4
            j = idx_flat % 4
            x[idx_flat, 0] = x_flat[idx(i,j,0)]
            x[idx_flat, 1] = x_flat[idx(i,j,1)]
            
        # Calculate payments
        p = np.zeros(16)
        for idx_flat in range(16):
            i = idx_flat // 4
            j = idx_flat % 4
            int1 = sum(x[k,j,0] for k in range(i))
            int2 = sum(x[i,l,1] for l in range(j))
            p[idx_flat] = i*x[idx_flat,0] + j*x[idx_flat,1] - int1 - int2
            
        # Check time
        if time.time() > deadline:
             return {"allocations": x, "payments": p}
             
        return {"allocations": x, "payments": p}
        
    except Exception as e:
        # Fallback
        x = np.ones((16,2)) * 0.5
        p = np.zeros(16)
        for idx_flat in range(16):
            i = idx_flat // 4
            j = idx_flat % 4
            p[idx_flat] = i*x[idx_flat,0] + j*x[idx_flat,1]
        return {"allocations": x, "payments": p}

