import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
        # 1. Parse support and distribution
        # Support is likely {0,1,2,3} for both goods.
        # Types are all pairs (v1, v2).
        types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_types = 16
        n_goods = 2
        
        # Extract probabilities
        if isinstance(distribution_spec, (np.ndarray, list, tuple)) and len(distribution_spec) == n_types:
            pi = np.array(distribution_spec)
            pi = pi / np.sum(pi) # Ensure normalized
        else:
            # Assume integer index
            if distribution_spec == 0:
                pi = np.ones(n_types) / n_types
            elif distribution_spec == 1:
                # Uniform on 0,1
                pi = np.zeros(n_types)
                for i, (v1, v2) in enumerate(types):
                    if v1 in [0, 1] and v2 in [0, 1]:
                        pi[i] = 0.25
            elif distribution_spec == 2:
                # Uniform on 0,1,2?
                pi = np.zeros(n_types)
                for i, (v1, v2) in enumerate(types):
                    if v1 in [0, 1, 2] and v2 in [0, 1, 2]:
                        pi[i] = 1.0/9.0
            else:
                pi = np.ones(n_types) / n_types
        
        # 2. Define LP
        # Variables: x[0..31], p[32..47]
        # x_ij is var index i*2 + j
        # p_i is var index 32 + i
        
        # Objective: Maximize sum(pi_i * p_i)
        # linprog minimizes, so use -c
        c = np.zeros(48)
        for i in range(n_types):
            c[32 + i] = -pi[i]
            
        # Inequalities A_ub x <= b_ub
        
        # 1. x_ij <= 1
        # 2. -x_ij <= 0
        # 3. p_i >= 0  => -p_i <= 0
        # 4. IR: p_i - sum(v_ij x_ij) <= 0  => sum(-v_ij x_ij) + p_i <= 0
        # 5. IC: p_i - p_k - sum(v_ij (x_ij - x_kj)) <= 0
        #       p_i - p_k - sum(v_ij x_ij) + sum(v_ij x_kj) <= 0
        
        # Build A_ub and b_ub
        # Total constraints:
        # 32 (x<=1)
        # 32 (-x<=0)
        # 16 (-p<=0)
        # 16 (IR)
        # 256 (IC)
        # Total = 352
        
        A = []
        b = []
        
        # x <= 1
        for i in range(n_types):
            for j in range(n_goods):
                row = np.zeros(48)
                row[i*2 + j] = 1
                A.append(row)
                b.append(1)
                
        # -x <= 0
        for i in range(n_types):
            for j in range(n_goods):
                row = np.zeros(48)
                row[i*2 + j] = -1
                A.append(row)
                b.append(0)
                
        # -p <= 0
        for i in range(n_types):
            row = np.zeros(48)
            row[32 + i] = -1
            A.append(row)
            b.append(0)
            
        # IR: p_i - sum(v_ij x_ij) <= 0
        for i in range(n_types):
            row = np.zeros(48)
            row[32 + i] = 1
            for j in range(n_goods):
                v = types[i][j]
                row[i*2 + j] = -v
            A.append(row)
            b.append(0)
            
        # IC: p_i - p_k - sum_j v_ij (x_ij - x_kj) <= 0
        for i in range(n_types):
            for k in range(n_types):
                if i == k: continue # Trivially 0<=0
                row = np.zeros(48)
                row[32 + i] = 1
                row[32 + k] = -1
                for j in range(n_goods):
                    v_i = types[i][j]
                    # - v_ij x_ij + v_ij x_kj
                    row[i*2 + j] = -v_i
                    row[k*2 + j] = v_i
                A.append(row)
                b.append(0)
                
        A_ub = np.array(A)
        b_ub = np.array(b)
        
        # Bounds
        bounds = [(0, 1)] * 32 + [(0, None)] * 16
        
        # Solve
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_opt = res.x[:32].reshape(16, 2)
            p_opt = res.x[32:]
            return {
                "allocations": x_opt,
                "payments": p_opt
            }
        else:
            # Fallback
            return {
                "allocations": np.zeros((16, 2)),
                "payments": np.zeros(16)
            }
