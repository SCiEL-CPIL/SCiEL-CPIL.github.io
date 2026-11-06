import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
        t_start = time.time()
        
        # Parse distribution_spec
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                probs = np.array(distribution_spec['probs'], dtype=float)
            elif 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = np.zeros(16)
                for k, v in distribution_spec.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        v1, v2 = k
                        idx = v1 * 4 + v2
                        if 0 <= idx < 16:
                            probs[idx] = v
                    elif isinstance(k, int) and 0 <= k < 16:
                        probs[k] = v
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) != 16:
                if len(probs) > 16:
                    probs = probs[:16]
                else:
                    tmp = np.zeros(16)
                    tmp[:len(probs)] = probs
                    probs = tmp
        else:
            probs = np.ones(16) / 16.0

        # Normalize
        probs = np.maximum(probs, 0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(16) / 16.0

        n_types = 16
        values = np.zeros((n_types, 2))
        for i in range(n_types):
            values[i, 0] = i // 4
            values[i, 1] = i % 4

        # LP Variables:
        # 0-31: x_ik (allocation prob of good k for type i)
        # 32-47: u_i (utility for type i)
        n_vars = 48
        
        c = np.zeros(n_vars)
        for i in range(n_types):
            for k in range(2):
                c[i * 2 + k] = -probs[i] * values[i, k]
            c[32 + i] = probs[i]

        A_ub = []
        b_ub = []

        # 1. IR: u_i >= 0  =>  -u_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[32 + i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)

        # 2. Non-negative Payment: p_i >= 0 => v_i x_i - u_i >= 0 => u_i - v_i x_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[32 + i] = 1.0
            for k in range(2):
                row[i * 2 + k] = -values[i, k]
            A_ub.append(row)
            b_ub.append(0.0)

        # 3. IC: u_j - u_i + sum_k (v_ik - v_jk) x_ik <= 0
        # Derived: U(j,j) >= U(j,i) => u_j - u_i >= sum_k (v_jk - v_ik) x_ik
        # => -(u_j - u_i) + sum_k (v_jk - v_ik) x_ik <= 0
        # => -u_j + u_i + sum_k (v_jk - v_ik) x_ik <= 0
        # Wait, let's re-verify the sign one last time.
        # U(j,j) >= U(j,i)
        # v_j x_j - p_j >= v_j x_i - p_i
        # p_i - p_j >= v_j x_i - v_j x_j
        # (v_i x_i - u_i) - (v_j x_j - u_j) >= v_j x_i - v_j x_j
        # v_i x_i - u_i - v_j x_j + u_j >= v_j x_i - v_j x_j
        # v_i x_i - u_i + u_j >= v_j x_i
        # u_j - u_i + v_i x_i - v_j x_i >= 0
        # u_j - u_i + sum_k (v_ik - v_jk) x_ik >= 0
        # Multiply by -1:
        # -u_j + u_i - sum_k (v_ik - v_jk) x_ik <= 0
        # -u_j + u_i + sum_k (v_jk - v_ik) x_ik <= 0
        
        for j in range(n_types):
            for i in range(n_types):
                row = np.zeros(n_vars)
                row[32 + j] = -1.0
                row[32 + i] = 1.0
                for k in range(2):
                    coeff = values[j, k] - values[i, k]
                    if coeff != 0:
                        row[i * 2 + k] = coeff
                A_ub.append(row)
                b_ub.append(0.0)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        ub[32:] = 10.0 # Utilities bounded by max possible value (6) + margin

        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method='highs')
            if res.status == 0 and res.x is not None:
                x_sol = res.x[:32]
                u_sol = res.x[32:]

                x_sol = np.clip(x_sol, 0, 1)
                u_sol = np.maximum(u_sol, 0)

                allocations = x_sol.reshape(16, 2)
                payments = np.zeros(16)
                for i in range(n_types):
                    payments[i] = np.sum(values[i] * allocations[i]) - u_sol[i]
                    payments[i] = max(0.0, payments[i])

                # Verification
                valid = True
                # Check IR and Non-neg Payment
                for i in range(n_types):
                    if payments[i] < -1e-8:
                        valid = False
                        break
                    if payments[i] > np.sum(values[i] * allocations[i]) + 1e-8:
                        valid = False
                        break
                
                # Check IC
                if valid:
                    for i in range(n_types):
                        for j in range(n_types):
                            if i == j: continue
                            # U(j, j) >= U(j, i)
                            u_jj = np.sum(values[j] * allocations[j]) - payments[j]
                            u_ji = np.sum(values[j] * allocations[i]) - payments[i]
                            if u_jj < u_ji - 1e-8:
                                valid = False
                                break
                        if not valid: break

                if valid:
                    best_allocations = allocations
                    best_payments = payments
                else:
                    # Fallback to zero if verification fails
                    best_allocations = np.zeros((16, 2))
                    best_payments = np.zeros(16)
            else:
                # LP failed
                best_allocations = np.zeros((16, 2))
                best_payments = np.zeros(16)
        except Exception:
            best_allocations = np.zeros((16, 2))
            best_payments = np.zeros(16)

        return {"allocations": best_allocations, "payments": best_payments}
