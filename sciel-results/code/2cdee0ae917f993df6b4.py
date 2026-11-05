import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute DFT for search grid
    M_search = 4096
    theta = 2 * np.pi * np.arange(M_search) / M_search
    E = np.exp(1j * np.outer(theta, np.arange(N)))
    cols = E.T  # (N, M)

    def supnorm_fast(c):
        P = E @ c
        return float(np.max(np.abs(P)))

    def local_search(c, time_limit):
        c = c.copy().astype(float)
        P = E @ c
        current_val = float(np.max(np.abs(P)))
        best_val = current_val
        best_c = c.copy()

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_1flip_val = current_val
            best_1flip_idx = -1

            # 1-flip
            for i in range(N):
                P_new = P - 2 * c[i] * cols[i]
                val = float(np.max(np.abs(P_new)))
                if val < best_1flip_val - 1e-12:
                    best_1flip_val = val
                    best_1flip_idx = i

            if best_1flip_idx != -1:
                i = best_1flip_idx
                c[i] *= -1
                P = E @ c
                current_val = best_1flip_val
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_c = c.copy()
                improved = True
                continue

            # 2-flip
            best_2flip_val = current_val
            best_2flip = None
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P - 2 * c[i] * cols[i] - 2 * c[j] * cols[j]
                    val = float(np.max(np.abs(P_new)))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)

            if best_2flip is not None:
                i, j = best_2flip
                c[i] *= -1
                c[j] *= -1
                P = E @ c
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_val = current_val
                    best_c = c.copy()
                improved = True
                continue

            # 3-flip (only if stuck and time remains)
            if time.time() < time_limit - 0.5:
                best_3flip_val = current_val
                best_3flip = None
                for i in range(N - 2):
                    for j in range(i + 1, N - 1):
                        for k in range(j + 1, N):
                            P_new = P - 2 * c[i] * cols[i] - 2 * c[j] * cols[j] - 2 * c[k] * cols[k]
                            val = float(np.max(np.abs(P_new)))
                            if val < best_3flip_val - 1e-12:
                                best_3flip_val = val
                                best_3flip = (i, j, k)

                if best_3flip is not None:
                    i, j, k = best_3flip
                    c[i] *= -1
                    c[j] *= -1
                    c[k] *= -1
                    P = E @ c
                    current_val = best_3flip_val
                    if current_val < best_val - 1e-12:
                        best_val = current_val
                        best_c = c.copy()
                    improved = True
                    continue

        return best_c, best_val

    candidates = []

    # 1. Golay-Rudin-Shapiro (Length 32 truncated)
    p = [1]
    q = [1]
    while len(p) < 32:
        p_new = [a + b for a, b in zip(p, q)]
        q_new = [a - b for a, b in zip(p, q)]
        p, q = p_new, q_new
    candidates.append(np.array(p[:24], dtype=float))
    candidates.append(np.array(p[8:32], dtype=float))
    candidates.append(np.array(p[4:28], dtype=float))

    # 2. Random
    for _ in range(30):
        candidates.append(np.random.choice([-1, 1], size=N).astype(float))

    # 3. Spectral Synthesis (Low Autocorrelation)
    for _ in range(30):
        if time.time() > deadline - 40:
            break
        c = np.random.choice([-1, 1], size=N).astype(float)
        # Quick local search on autocorrelation
        def autocorr_cost(c):
            r = np.correlate(c, c, mode='full')
            r = r[N-1:]
            return np.sum(np.abs(r)**2)
        
        c_curr = c.copy()
        cost_curr = autocorr_cost(c_curr)
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            best_cost = cost_curr
            best_idx = -1
            for i in range(N):
                c_try = c_curr.copy()
                c_try[i] *= -1
                cost_try = autocorr_cost(c_try)
                if cost_try < best_cost:
                    best_cost = cost_try
                    best_idx = i
            if best_idx != -1:
                c_curr[best_idx] *= -1
                cost_curr = best_cost
                improved = True
        candidates.append(c_curr)

    best_overall_c = None
    best_overall_val = float('inf')

    # Run local search on candidates
    # Allocate time dynamically
    num_cands = len(candidates)
    time_per = max(1.0, (deadline - time.time() - 10) / max(1, num_cands))
    
    for c0 in candidates:
        if time.time() > deadline - 10:
            break
        c_best, v_best = local_search(c0, time.time() + time_per)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best

    # Final Polish with High Resolution
    if best_overall_c is not None:
        M_final = 16384
        theta_final = 2 * np.pi * np.arange(M_final) / M_final
        E_final = np.exp(1j * np.outer(theta_final, np.arange(N)))
        cols_final = E_final.T

        c_curr = best_overall_c.copy()
        P_curr = E_final @ c_curr
        curr_val = float(np.max(np.abs(P_curr)))
        best_val = curr_val
        best_c = c_curr.copy()

        for _ in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            best_1flip_val = curr_val
            best_1flip_idx = -1
            for i in range(N):
                P_new = P_curr - 2 * c_curr[i] * cols_final[i]
                val = float(np.max(np.abs(P_new)))
                if val < best_1flip_val - 1e-12:
                    best_1flip_val = val
                    best_1flip_idx = i
            if best_1flip_idx != -1:
                i = best_1flip_idx
                c_curr[i] *= -1
                P_curr = E_final @ c_curr
                curr_val = best_1flip_val
                if curr_val < best_val - 1e-12:
                    best_val = curr_val
                    best_c = c_curr.copy()
                improved = True
            else:
                best_2flip_val = curr_val
                best_2flip = None
                for i in range(N - 1):
                    for j in range(i + 1, N):
                        P_new = P_curr - 2 * c_curr[i] * cols_final[i] - 2 * c_curr[j] * cols_final[j]
                        val = float(np.max(np.abs(P_new)))
                        if val < best_2flip_val - 1e-12:
                            best_2flip_val = val
                            best_2flip = (i, j)
                if best_2flip is not None:
                    i, j = best_2flip
                    c_curr[i] *= -1
                    c_curr[j] *= -1
                    P_curr = E_final @ c_curr
                    curr_val = best_2flip_val
                    if curr_val < best_val - 1e-12:
                        best_val = curr_val
                        best_c = c_curr.copy()
                    improved = True
            if not improved:
                break

        if best_val < best_overall_val:
            best_overall_val = best_val
            best_overall_c = best_c

    return [int(c) for c in best_overall_c]
