import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    N_COEFFS = 24
    sqrt_N = math.sqrt(N_COEFFS)

    # Known good starting points
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]

    rs_poly = get_rs_poly(N_COEFFS)

    known_solutions = [
        rs_poly,
        [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1],
    ]

    # --- Fast Evaluation Machinery ---
    N_FFT = 4096

    def evaluate_fast_search(coeffs):
        c_padded = np.zeros(N_FFT, dtype=np.float64)
        c_padded[:N_COEFFS] = coeffs
        vals = np.fft.fft(c_padded)
        mags = np.abs(vals)

        max_idx = np.argmax(mags)
        theta_max = 2 * np.pi * max_idx / N_FFT

        def local_eval_fast(theta):
            z = complex(math.cos(theta), math.sin(theta))
            val = 0.0 + 0.0j
            z_power = 1.0 + 0.0j
            for k in range(N_COEFFS):
                val += coeffs[k] * z_power
                z_power *= z
            return abs(val)

        a = theta_max - math.pi / N_FFT
        b = theta_max + math.pi / N_FFT
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = local_eval_fast(c)
        fd = local_eval_fast(d)

        for _ in range(15):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = local_eval_fast(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = local_eval_fast(c)

        refined_max = max(fc, fd)

        # Check second peak
        max_idx2 = max_idx
        if max_idx > 0:
            mags_copy = mags.copy()
            mags_copy[max_idx] = 0
            max_idx2 = np.argmax(mags_copy)
        else:
            mags_copy = mags.copy()
            mags_copy[0] = 0
            max_idx2 = np.argmax(mags_copy)

        if mags[max_idx2] > 0.98 * mags[max_idx]:
            theta_max2 = 2 * np.pi * max_idx2 / N_FFT
            a2 = theta_max2 - math.pi / N_FFT
            b2 = theta_max2 + math.pi / N_FFT
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = local_eval_fast(c2)
            fd2 = local_eval_fast(d2)

            for _ in range(15):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = local_eval_fast(d2)
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = local_eval_fast(c2)

            refined_max = max(refined_max, max(fc2, fd2))

        return refined_max

    # Initialize best
    best_coeffs = None
    best_score = float('inf')

    for sol in known_solutions:
        score = evaluate_fast_search(sol) / sqrt_N
        if score < best_score:
            best_score = score
            best_coeffs = sol[:]

    # --- Continuous Relaxation with Projected Gradient Descent ---
    # We optimize f(x) = max_k |DFT(x)_k| where x is continuous, then project to {+1,-1}
    # The gradient of max |DFT| at the peak gives us direction to decrease the sup norm

    def continuous_relaxation_search(initial_coeffs, max_iters=2000):
        """
        Use continuous relaxation: treat coefficients as real variables in [-1, 1],
        use gradient descent on the sup-norm of the DFT, then project back to {+1, -1}.
        """
        x = np.array(initial_coeffs, dtype=np.float64)
        best_x = x.copy()
        best_val = float('inf')

        # Precompute frequency points for evaluation
        n_freq = 256  # Number of frequency points to consider
        thetas = np.linspace(0, 2 * np.pi, n_freq, endpoint=False)
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        def compute_dft_mags(x):
            """Compute |DFT(x)| at the n_freq frequency points."""
            # x has length N_COEFFS, evaluate at n_freq points
            # DFT_k = sum_{j=0}^{N-1} x_j * exp(-i * 2*pi*k*j/n_freq)
            # We can vectorize this
            # Shape: (n_freq, N_COEFFS)
            k_idx = np.arange(n_freq)[:, None]  # (n_freq, 1)
            j_idx = np.arange(N_COEFFS)[None, :]  # (1, N_COEFFS)
            phases = -2.0 * np.pi * k_idx * j_idx / n_freq
            # DFT values: (n_freq,)
            dft_vals = np.exp(1j * phases) @ x  # (n_freq,)
            return np.abs(dft_vals)

        def compute_grad_supnorm(x):
            """
            Compute gradient of max_k |DFT(x)_k| with respect to x.
            For the active peak k*, grad = Re(exp(-i*2*pi*k* j/n) * sign(DFT_k*))
            """
            mags = compute_dft_mags(x)
            k_star = np.argmax(mags)
            # Compute DFT value at k_star
            j_idx = np.arange(N_COEFFS)
            phase = -2.0 * np.pi * k_star * j_idx / n_freq
            dft_k = np.exp(1j * phase) @ x
            sign_dft = np.sign(dft_k) if abs(dft_k) > 1e-10 else 1.0
            # Gradient: Re(exp(-i*phase) * sign_dft)
            grad = np.real(np.exp(1j * phase) * sign_dft)
            return grad, k_star, mags[k_star]

        lr = 0.1  # Learning rate
        for it in range(max_iters):
            if time.time() - start_time > deadline - 10:
                break

            grad, k_star, peak_val = compute_grad_supnorm(x)

            # Gradient descent step
            x = x - lr * grad

            # Project to [-1, 1] box
            x = np.clip(x, -1.0, 1.0)

            # Project to {+1, -1}
            x_proj = np.sign(x)
            x_proj[x_proj == 0] = 1.0

            # Evaluate the projected solution
            proj_coeffs = [int(v) for v in x_proj]
            proj_score = evaluate_fast_search(proj_coeffs) / sqrt_N

            if proj_score < best_val:
                best_val = proj_score
                best_x = x_proj.copy()

            # Also check if we're making progress in continuous space
            if it % 50 == 0:
                # Adaptive learning rate
                if peak_val > best_val * sqrt_N:
                    lr = max(0.01, lr * 0.95)
                else:
                    lr = min(0.3, lr * 1.05)

        return [int(v) for v in best_x], best_val

    # Run continuous relaxation from multiple starting points
    for sol in known_solutions:
        if time.time() - start_time > deadline - 20:
            break
        relaxed_coeffs, relaxed_score = continuous_relaxation_search(sol, max_iters=1500)
        if relaxed_score < best_score:
            best_score = relaxed_score
            best_coeffs = relaxed_coeffs[:]

    # Also try random initializations for continuous relaxation
    n_random_starts = 5
    for _ in range(n_random_starts):
        if time.time() - start_time > deadline - 15:
            break
        random_coeffs = [random.choice([1, -1]) for _ in range(N_COEFFS)]
        relaxed_coeffs, relaxed_score = continuous_relaxation_search(random_coeffs, max_iters=1000)
        if relaxed_score < best_score:
            best_score = relaxed_score
            best_coeffs = relaxed_coeffs[:]

    # --- Discrete Local Search Polish ---
    # After continuous relaxation, do discrete hill climbing to fine-tune
    current_coeffs = best_coeffs[:]
    current_score = best_score

    # Simple hill climbing: try all single flips, take best improving move
    for iteration in range(500):
        if time.time() - start_time > deadline - 5:
            break

        best_move_idx = -1
        best_move_score = current_score
        best_new_coeffs = None

        for i in range(N_COEFFS):
            new_coeffs = current_coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            new_score = evaluate_fast_search(new_coeffs) / sqrt_N

            if new_score < best_move_score - 1e-12:
                best_move_score = new_score
                best_move_idx = i
                best_new_coeffs = new_coeffs

        if best_move_idx == -1:
            # No improving move found; try a random kick
            if iteration > 100:
                kick_size = random.choice([1, 2, 3])
                kick_indices = random.sample(range(N_COEFFS), kick_size)
                for k in kick_indices:
                    current_coeffs[k] = -current_coeffs[k]
                current_score = evaluate_fast_search(current_coeffs) / sqrt_N
            else:
                break
        else:
            current_coeffs = best_new_coeffs
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs[:]

    # Final precise evaluation and polish
    candidate = best_coeffs[:]
    precise_improved = True
    while precise_improved and time.time() - start_time < deadline - 2:
        precise_improved = False
        for i in range(N_COEFFS):
            new_coeffs = candidate[:]
            new_coeffs[i] = -new_coeffs[i]

            def evaluate_precise(coeffs):
                N_FFT_HP = 8192
                c_padded = np.zeros(N_FFT_HP, dtype=np.float64)
                c_padded[:N_COEFFS] = coeffs
                vals = np.fft.fft(c_padded)
                mags = np.abs(vals)
                max_idx = np.argmax(mags)
                theta_max = 2 * np.pi * max_idx / N_FFT_HP

                def local_eval_hp(theta):
                    z = complex(math.cos(theta), math.sin(theta))
                    val = 0.0 + 0.0j
                    z_power = 1.0 + 0.0j
                    for k in range(N_COEFFS):
                        val += coeffs[k] * z_power
                        z_power *= z
                    return abs(val)

                a = theta_max - math.pi / N_FFT_HP
                b = theta_max + math.pi / N_FFT_HP
                gr = (math.sqrt(5) - 1) / 2
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = local_eval_hp(c)
                fd = local_eval_hp(d)

                for _ in range(30):
                    if fc < fd:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = local_eval_hp(d)
                    else:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = local_eval_hp(c)

                refined_max = max(fc, fd)

                # Check second peak
                max_idx2 = max_idx
                if max_idx > 0:
                    mags_copy = mags.copy()
                    mags_copy[max_idx] = 0
                    max_idx2 = np.argmax(mags_copy)
                else:
                    mags_copy = mags.copy()
                    mags_copy[0] = 0
                    max_idx2 = np.argmax(mags_copy)

                if mags[max_idx2] > 0.98 * mags[max_idx]:
                    theta_max2 = 2 * np.pi * max_idx2 / N_FFT_HP
                    a2 = theta_max2 - math.pi / N_FFT_HP
                    b2 = theta_max2 + math.pi / N_FFT_HP
                    c2 = b2 - gr * (b2 - a2)
                    d2 = a2 + gr * (b2 - a2)
                    fc2 = local_eval_hp(c2)
                    fd2 = local_eval_hp(d2)

                    for _ in range(30):
                        if fc2 < fd2:
                            a2 = c2
                            c2 = d2
                            fc2 = fd2
                            d2 = a2 + gr * (b2 - a2)
                            fd2 = local_eval_hp(d2)
                        else:
                            b2 = d2
                            d2 = c2
                            fd2 = fc2
                            c2 = b2 - gr * (b2 - a2)
                            fc2 = local_eval_hp(c2)

                    refined_max = max(refined_max, max(fc2, fd2))

                return refined_max

            new_score = evaluate_precise(new_coeffs) / sqrt_N
            if new_score < best_score - 1e-12:
                candidate = new_coeffs
                best_score = new_score
                best_coeffs = candidate[:]
                precise_improved = True

    return best_coeffs
# EVOLVE-BLOCK-END
