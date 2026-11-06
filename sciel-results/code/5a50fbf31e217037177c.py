import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin

    n = 24
    sqrt_n = np.sqrt(n)

    # Use a fine grid for local search evaluation
    num_points = 4096
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * thetas)

    # Precompute powers matrix Z (num_points x n)
    Z = np.ones((num_points, n), dtype=np.complex128)
    for k in range(1, n):
        Z[:, k] = Z[:, k-1] * z

    def get_sup_norm(coeffs):
        vals = Z @ coeffs
        return np.max(np.abs(vals))

    def refine_peak(coeffs, theta_init):
        # Refine the peak around theta_init using golden section
        def f(theta):
            z_point = np.exp(1j * theta)
            p_val = 0.0
            zp = 1.0 + 0.0j
            for c in coeffs:
                p_val += c * zp
                zp *= z_point
            return np.abs(p_val)

        # Determine a small interval around theta_init
        delta = 2 * np.pi / num_points
        a = theta_init - delta
        b = theta_init + delta

        gr = (np.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        fc = f(c)
        fd = f(d)

        for _ in range(40):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)

        return f((a + b) / 2)

    # Generate the Rudin-Shapiro sequence of length 24
    # r_0 = 1, r_{2n} = r_n, r_{2n+1} = -r_n
    def rudin_shapiro(n):
        seq = []
        for i in range(n):
            # Recursive approach is cleaner
            def r(i):
                if i == 0:
                    return 1
                if i % 2 == 0:
                    return r(i // 2)
                else:
                    return -r((i - 1) // 2)
            seq.append(r(i))
        return np.array(seq, dtype=np.float64)

    candidates = []
    candidates.append(rudin_shapiro(n))
    candidates.append(np.ones(n, dtype=np.float64))
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.float64))

    # Add some random starts
    np.random.seed(42)
    for _ in range(15):
        candidates.append(np.random.choice([1, -1], n).astype(np.float64))

    best_coeffs = None
    best_score = float('inf')

    for start_coeffs in candidates:
        if time.time() > deadline - 10.0:
            break

        coeffs = start_coeffs.copy()
        current_max = get_sup_norm(coeffs)

        # Local Search: First-improvement hill climbing
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for k in range(n):
                if time.time() > deadline - 5.0:
                    break

                old_val = coeffs[k]
                coeffs[k] = -old_val

                new_max = get_sup_norm(coeffs)

                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    break
                else:
                    coeffs[k] = old_val

        # Refine the best peak
        vals = Z @ coeffs
        abs_vals = np.abs(vals)
        idx = np.argmax(abs_vals)
        theta_max = thetas[idx]

        refined_max = refine_peak(coeffs, theta_max)

        if refined_max < best_score:
            best_score = refined_max
            best_coeffs = coeffs.copy()

    if best_coeffs is not None:
        return best_coeffs.astype(int).tolist()
    else:
        return [1] * n

BET: {mechanism: "Structured seed (Rudin-Shapiro) plus first-improvement hill climbing with high-resolution refinement should reliably reach the known optimum.", q: 0.98, delta: [-0.015, -0.015]}
