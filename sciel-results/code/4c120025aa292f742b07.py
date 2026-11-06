import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 84.0

    # Parse distribution
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

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    probs = np.clip(probs, 0, 1)

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

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    best_revenue = 0.0

    is_grid = True
    for i, (v1, v2) in enumerate(support_list):
        if v1 != i // 4 or v2 != i % 4:
            is_grid = False
            break

    if is_grid:
        # Approach 1: 32-var LP with monotonicity (grid-specific, tighter)
        try:
            n = 32
            obj_c = np.zeros(n)
            for i in range(4):
                for j in range(4):
                    idx1 = 4*i + j
                    idx2 = 16 + 4*i + j
                    c1 = i * probs[i*4+j]
                    for ip in range(i+1, 4):
                        c1 -= probs[ip*4+j]
                    c2 = j * probs[i*4+j]
                    for jp in range(j+1, 4):
                        c2 -= probs[i*4+jp]
                    obj_c[idx1] = -c1
                    obj_c[idx2] = -c2

            A_rows = []
            b_rows = []

            # Monotonicity: x(i,j) >= x(i-1,j) and x(i,j) >= x(i,j-1)
            for i in range(1, 4):
                for j in range(4):
                    r = np.zeros(n); r[4*(i-1)+j] = 1; r[4*i+j] = -1
                    A_rows.append(r); b_rows.append(0)
                    r = np.zeros(n); r[16+4*(i-1)+j] = 1; r[16+4*i+j] = -1
                    A_rows.append(r); b_rows.append(0)
            for i in range(4):
                for j in range(1, 4):
                    r = np.zeros(n); r[4*i+j-1] = 1; r[4*i+j] = -1
                    A_rows.append(r); b_rows.append(0)
                    r = np.zeros(n); r[16+4*i+j-1] = 1; r[16+4*i+j] = -1
                    A_rows.append(r); b_rows.append(0)

            # Non-negative payment: p_{i,j} >= 0
            for i in range(4):
                for j in range(4):
                    r = np.zeros(n)
                    r[4*i+j] = -i
                    r[16+4*i+j] = -j
                    for k in range(1, i+1):
                        r[4*(k-1)+j] += 1
                    for k in range(1, j+1):
                        r[16+4*i+k-1] += 1
                    A_rows.append(r); b_rows.append(0)

            A_ub = np.array(A_rows)
            b_ub = np.array(b_rows)
            bounds = [(0, 1)] * n

            res = linprog(obj_c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True})
            if res.success:
                x1 = np.clip(res.x[:16], 0, 1)
                x2 = np.clip(res.x[16:32], 0, 1)
                alloc = np.zeros((16, 2))
                pays = np.zeros(16)
                for i in range(4):
                    for j in range(4):
                        idx = i*4+j
                        alloc[idx, 0] = x1[4*i+j]
                        alloc[idx, 1] = x2[16+4*i+j]
                        p = i*x1[4*i+j] + j*x2[16+4*i+j]
                        for k in range(1, i+1):
                            p -= x1[4*(k-1)+j]
                        for k in range(1, j+1):
                            p -= x2[16+4*i+k-1]
                        pays[idx] = max(0.0, p)

                # Verify IC and IR
                ok = True
                for idx in range(16):
                    if np.dot(V[idx], alloc[idx]) - pays[idx] < -1e-8:
                        ok = False
                        break
                if ok:
                    for i in range(16):
                        for ip in range(16):
                            if np.dot(V[i], alloc[ip]) - pays[ip] - (np.dot(V[i], alloc[i]) - pays[i]) > 1e-8:
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    rev = np.dot(probs, pays)
                    if rev > best_revenue:
                        best_revenue = rev
                        best_allocations = alloc.copy()
                        best_payments = pays.copy()
        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # Approach 2: General 48-var LP (works for any support)
    try:
        n = 48
        c = np.zeros(n)
        for i in range(16):
            c[32+i] = -probs[i]

        # IR: V_i . alloc_i - pay_i >= 0
        A_ir = np.zeros((16, n))
        for i in range(16):
            A_ir[i, 2*i] = -V[i, 0]
            A_ir[i, 2*i+1] = -V[i, 1]
            A_ir[i, 32+i] = 1

        # IC: V_i . alloc_ip - pay_ip <= V_i . alloc_i - pay_i  for all i, ip
        A_ic = np.zeros((256, n))
        for i in range(16):
            for ip in range(16):
                r = i*16 + ip
                A_ic[r, 2*ip] = V[i, 0]
                A_ic[r, 2*ip+1] = V[i, 1]
                A_ic[r, 32+ip] = -1
                A_ic[r, 2*i] = -V[i, 0]
                A_ic[r, 2*i+1] = -V[i, 1]
                A_ic[r, 32+i] = 1

        A = np.vstack([A_ir, A_ic])
        b = np.concatenate([np.zeros(16), np.zeros(256)])
        bounds = [(0,1)]*32 + [(0, None)]*16

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                      options={'presolve': True})
        if res.success:
            alloc = np.clip(res.x[:32].reshape(16, 2), 0, 1)
            pays = np.maximum(res.x[32:], 0)

            ok = True
            for i in range(16):
                if np.dot(V[i], alloc[i]) - pays[i] < -1e-8:
                    ok = False
                    break
            if ok:
                for i in range(16):
                    for ip in range(16):
                        if np.dot(V[i], alloc[ip]) - pays[ip] - (np.dot(V[i], alloc[i]) - pays[i]) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                rev = np.dot(probs, pays)
                if rev > best_revenue:
                    best_revenue = rev
                    best_allocations = alloc.copy()
                    best_payments = pays.copy()
    except Exception:
        pass

    # Approach 3: Direct enumeration over threshold mechanisms for single-dimension cases
    # For the grid case, try simple threshold mechanisms as a fast check
    if is_grid:
        try:
            # Try separable threshold: for each good, pick a threshold per other-good level
            # x1(i,j) = 1 if i >= t1(j), else 0; x2(i,j) = 1 if j >= t2(i), else 0
            # t1(j) in {0,1,2,3,4}, t2(i) in {0,1,2,3,4}
            # This gives 5^4 * 5^4 = 625 mechanisms to check
            for t1 in range(5**4):
                if time.time() > deadline:
                    break
                tt1 = [(t1 >> (4*k)) % 5 for k in range(4)]
                for t2 in range(5**4):
                    if time.time() > deadline:
                        break
                    tt2 = [(t2 >> (4*k)) % 5 for k in range(4)]
                    alloc = np.zeros((16, 2))
                    for i in range(4):
                        for j in range(4):
                            alloc[i*4+j, 0] = 1.0 if i >= tt1[j] else 0.0
                            alloc[i*4+j, 1] = 1.0 if j >= tt2[i] else 0.0
                    # Compute payments via integral
                    pays = np.zeros(16)
                    for i in range(4):
                        for j in range(4):
                            p = i * alloc[i*4+j, 0] + j * alloc[i*4+j, 1]
                            for k in range(1, i+1):
                                p -= alloc[(k-1)*4+j, 0]
                            for k in range(1, j+1):
                                p -= alloc[i*4+(k-1), 1]
                            pays[i*4+j] = max(0.0, p)
                    # Verify
                    ok = True
                    for idx in range(16):
                        if np.dot(V[idx], alloc[idx]) - pays[idx] < -1e-8:
                            ok = False
                            break
                    if ok:
                        for ii in range(16):
                            for ip in range(16):
                                if np.dot(V[ii], alloc[ip]) - pays[ip] - (np.dot(V[ii], alloc[ii]) - pays[ii]) > 1e-8:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        rev = np.dot(probs, pays)
                        if rev > best_revenue:
                            best_revenue = rev
                            best_allocations = alloc.copy()
                            best_payments = pays.copy()
        except Exception:
            pass

    return {'allocations': best_allocations, 'payments': best_payments}
