# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0

    def evaluate(S, W):
        n = len(S)
        if n == 0:
            return 0.0
        if n == 1:
            return 0.0
        cumW = np.concatenate([[0.0], np.cumsum(W)])
        P = 0.0
        for k in range(n):
            wk = W[k]
            for l in range(n):
                threshold = 2.0 * S[l] - S[k]
                if threshold <= 0:
                    continue
                max_j = np.searchsorted(S, threshold - S, side='left')
                prefix_sums = cumW[max_j]
                P += wk * W[l] * np.dot(W, prefix_sums)
        return P

    # Analytic 2-point solution
    p2 = (15.0 - np.sqrt(33.0)) / 16.0
    global_best_P = evaluate(np.array([0.0, 1.0]), np.array([p2, 1.0 - p2]))
    global_best_S = np.array([0.0, 1.0])
    global_best_W = np.array([p2, 1.0 - p2])

    # 3-point analytic grid search: support {0, M, 1}, weights {a, b, c}
    # where a + b + c = 1, a >= 0, b >= 0, c >= 0, 0 <= M <= 1
    # P = sum over all (x1,x2,x3,x4) of prod(w_i) * I(x1+x2+x3 < 2*x4)
    # With 3 points, we can enumerate all 3^4 = 81 combinations analytically.
    
    def P_3point(M, a, b, c):
        """Compute P for 3-point support {0, M, 1} with weights {a, b, c}."""
        S = np.array([0.0, M, 1.0])
        W = np.array([a, b, c])
        n = 3
        P = 0.0
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * S[l] - S[k]
                if threshold <= 0:
                    continue
                for i in range(n):
                    for j in range(n):
                        if S[i] + S[j] < threshold:
                            P += W[k] * W[l] * W[i] * W[j]
        return P

    # Coarse grid over M and weight allocation
    # M in [0, 1], weights: a = P(X=0), b = P(X=M), c = P(X=1), a+b+c=1
    # Parameterize: a in [0,1], b in [0, 1-a], c = 1-a-b
    
    best_P_3 = 0.0
    best_M = 0.5
    best_a = 0.4
    best_b = 0.2
    
    # Coarse grid
    M_vals = np.linspace(0.1, 0.9, 19)
    a_vals = np.linspace(0.1, 0.8, 18)
    b_vals = np.linspace(0.05, 0.5, 12)
    
    for M in M_vals:
        for a in a_vals:
            for b in b_vals:
                c = 1.0 - a - b
                if c < 0.01:
                    continue
                P = P_3point(M, a, b, c)
                if P > best_P_3:
                    best_P_3 = P
                    best_M = M
                    best_a = a
                    best_b = b
    
    if best_P_3 > global_best_P:
        global_best_P = best_P_3
        global_best_S = np.array([0.0, best_M, 1.0])
        global_best_W = np.array([best_a, best_b, 1.0 - best_a - best_b])

    # Fine grid around the best found
    if time.time() < deadline - 100.0:
        M_fine = np.linspace(max(0.05, best_M - 0.1), min(0.95, best_M + 0.1), 25)
        a_fine = np.linspace(max(0.05, best_a - 0.1), min(0.85, best_a + 0.1), 25)
        b_fine = np.linspace(max(0.01, best_b - 0.1), min(0.6, best_b + 0.1), 20)
        
        for M in M_fine:
            for a in a_fine:
                for b in b_fine:
                    c = 1.0 - a - b
                    if c < 0.01 or c > 0.9:
                        continue
                    P = P_3point(M, a, b, c)
                    if P > global_best_P:
                        global_best_P = P
                        global_best_S = np.array([0.0, M, 1.0])
                        global_best_W = np.array([a, b, c])

    # Even finer grid
    if time.time() < deadline - 50.0:
        S_curr = global_best_S
        W_curr = global_best_W
        if len(S_curr) == 3:
            M_curr = S_curr[1]
            a_curr = W_curr[0]
            b_curr = W_curr[1]
            
            M_very_fine = np.linspace(max(0.05, M_curr - 0.03), min(0.95, M_curr + 0.03), 30)
            a_very_fine = np.linspace(max(0.05, a_curr - 0.03), min(0.85, a_curr + 0.03), 30)
            b_very_fine = np.linspace(max(0.01, b_curr - 0.03), min(0.6, b_curr + 0.03), 25)
            
            for M in M_very_fine:
                for a in a_very_fine:
                    for b in b_very_fine:
                        c = 1.0 - a - b
                        if c < 0.01 or c > 0.9:
                            continue
                        P = P_3point(M, a, b, c)
                        if P > global_best_P:
                            global_best_P = P
                            global_best_S = np.array([0.0, M, 1.0])
                            global_best_W = np.array([a, b, c])

    # Try 4-point structured candidates
    if time.time() < deadline - 30.0:
        S_4_candidates = [
            ([0.0, 0.25, 0.75, 1.0], [0.45, 0.05, 0.05, 0.45]),
            ([0.0, 0.3, 0.7, 1.0], [0.4, 0.1, 0.1, 0.4]),
            ([0.0, 0.2, 0.8, 1.0], [0.5, 0.1, 0.1, 0.3]),
            ([0.0, 0.35, 0.65, 1.0], [0.42, 0.08, 0.08, 0.42]),
            ([0.0, 0.4, 0.6, 1.0], [0.4, 0.1, 0.1, 0.4]),
        ]
        for S_c, W_c in S_4_candidates:
            P_c = evaluate(np.array(S_c), np.array(W_c))
            if P_c > global_best_P:
                global_best_P = P_c
                global_best_S = np.array(S_c)
                global_best_W = np.array(W_c)

    # Final validity check
    global_best_S = np.sort(global_best_S)
    global_best_W = np.clip(global_best_W, 0.0, None)
    w_sum = np.sum(global_best_W)
    if w_sum < 1e-9:
        global_best_S = np.array([0.0, 1.0])
        global_best_W = np.array([0.5, 0.5])
    else:
        global_best_W /= w_sum
        global_best_S = np.clip(global_best_S, 0.0, None)
    
    return global_best_S.astype(np.float64), global_best_W.astype(np.float64)

# EVOLVE-BLOCK-END
