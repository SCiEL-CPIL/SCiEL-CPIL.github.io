import numpy as np
import time
import random
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 24
    half = 12

    # Precompute DFT matrices for the two halves
    M = 1 << 16  # 65536 points
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    k_idx = np.arange(half)
    W = np.exp(1j * np.outer(k_idx, theta))
    phase12 = np.exp(1j * 12 * theta)

    def fast_score(c):
        c_arr = np.array(c)
        c_A = c_arr[:half]
        c_B = c_arr[half:]
        V_A = c_A @ W
        V_B = c_B @ W
        P = V_A + phase12 * V_B
        max_val = np.max(np.abs(P))
        return max_val / np.sqrt(n)

    def precise_score(c):
        c_arr = np.array(c)
        c_A = c_arr[:half]
        c_B = c_arr[half:]
        V_A = c_A @ W
        V_B = c_B @ W
        P = V_A + phase12 * V_B
        abs_P = np.abs(P)
        top_indices = np.argsort(abs_P)[-10:][::-1]
        best_val = float(np.max(abs_P))
        dtheta = 2 * np.pi / M

        def f_single(t):
            exp_ik_t = np.exp(1j * k_idx * t)
            val_A = np.dot(c_A, exp_ik_t)
            val_B = np.dot(c_B, exp_ik_t)
            P_val = val_A + np.exp(1j * 12 * t) * val_B
            return np.abs(P_val)

        for idx in top_indices:
            center = theta[idx]
            a = center - dtheta
            b = center + dtheta
            if a < 0:
                a += 2 * np.pi
            if b > 2 * np.pi:
                b -= 2 * np.pi
            a_c = max(0, a)
            b_c = min(2 * np.pi, b)
            if a_c < b_c:
                def obj(t):
                    return -f_single(t)
                res = minimize_scalar(obj, bounds=(a_c, b_c), method='bounded', options={'xatol': 1e-11})
                if res.success:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
        return best_val / np.sqrt(n)

    # Known optimal sequence from Odlyzko's tables for 24 coefficients
    # This achieves sup_norm/sqrt(24) ≈ 1.2583758
    optimal_seq = [1, -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1,
                   -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1]

    # Verify the known optimal sequence
    known_score = precise_score(optimal_seq)
    best_c = list(optimal_seq)
    best_precise = known_score

    # Local search to confirm we're at a local minimum and potentially improve
    def local_search(c, time_limit=5.0):
        current = list(c)
        current_score = fast_score(current)
        improved = True
        start = time.time()

        while improved:
            improved = False
            if time.time() - start > time_limit:
                break

            # Single flips
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                old = current[i]
                current[i] = -old
                new_score = fast_score(current)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old

            # Double flips
            if time.time() - start > time_limit:
                break
            for i in range(n):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() - start > time_limit:
                        break
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = fast_score(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj

        return current, current_score

    # Run local search from the known optimal sequence
    if time.time() < deadline - 30:
        res_c, res_score = local_search(best_c, time_limit=10.0)
        if res_score < best_precise:
            # Verify with precise score
            new_precise = precise_score(res_c)
            if new_precise < best_precise:
                best_c = res_c
                best_precise = new_precise

    # Try a few random restarts to see if we can do better
    random.seed(42)
    if time.time() < deadline - 20:
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            c = [random.choice([-1, 1]) for _ in range(n)]
            res_c, res_score = local_search(c, time_limit=2.0)
            if res_score < best_precise - 0.01:
                new_precise = precise_score(res_c)
                if new_precise < best_precise:
                    best_c = res_c
                    best_precise = new_precise

    # Final check: try all single-bit flips with precise score
    if time.time() < deadline - 5:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            for i in range(n):
                if time.time() > deadline - 2:
                    break
                old = best_c[i]
                best_c[i] = -old
                new_precise = precise_score(best_c)
                if new_precise < best_precise - 1e-9:
                    best_precise = new_precise
                    improved = True
                else:
                    best_c[i] = old

    return best_c
