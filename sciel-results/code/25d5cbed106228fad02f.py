"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 630
    safety_margin = 30  # seconds
    best_score = float('inf')
    best_z = None

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return deadline - time.time() - safety_margin

    # Precompute Laguerre zeros for various degrees using numpy
    def laguerre_zeros(n, alpha):
        """Compute zeros of L_n^(alpha)(x) using companion matrix eigenvalues."""
        n = int(n)
        a = alpha
        diag = np.array([2.0*k - 1.0 + a for k in range(1, n+1)])
        subdiag = np.array([np.sqrt(k*(k+a)) for k in range(1, n)])
        J = np.diag(diag)
        for i in range(n-1):
            J[i, i+1] = subdiag[i]
            J[i+1, i] = subdiag[i]
        eigvals = np.linalg.eigvalsh(J)
        return np.sort(eigvals)

    # Bessel J_{1/2} zeros: J_{1/2}(x) = sqrt(2/(pi*x)) * sin(sqrt(x))
    # Zeros at sqrt(x) = k*pi => x = k^2 * pi^2
    # 12 smallest: z_i = i^2 * pi^2 for i = 1, ..., 12
    def bessel_j_half_zeros(count):
        """Return the 'count' smallest positive zeros of J_{1/2}(x)."""
        return [float(i**2 * math.pi**2) for i in range(1, count + 1)]

    alpha = -0.5

    # Build candidate sets, prioritizing Bessel zeros
    candidate_sets = []

    # Priority 1: Bessel J_{1/2} zeros (the directed strategy)
    try:
        z_bessel = bessel_j_half_zeros(12)
        if len(z_bessel) == 12 and all(z > 0 for z in z_bessel) and len(set(z_bessel)) == 12:
            candidate_sets.append(z_bessel)
    except:
        pass

    # Priority 2: Zeros of L_12^(-0.5)(x)
    try:
        z12 = laguerre_zeros(12, alpha)
        if len(z12) == 12 and all(z > 0 for z in z12) and len(set(np.round(z12, 10))) == 12:
            candidate_sets.append(list(np.round(z12, 10)))
    except:
        pass

    # Priority 3: Zeros of L_13^(-0.5)(x), take first 12
    try:
        z13 = laguerre_zeros(13, alpha)
        if len(z13) >= 12:
            z13_12 = z13[:12]
            if all(z > 0 for z in z13_12) and len(set(np.round(z13_12, 10))) == 12:
                candidate_sets.append(list(np.round(z13_12, 10)))
    except:
        pass

    # Priority 4: Geometric sequences
    for ratio in [1.1, 1.2, 1.3, 1.5, 2.0]:
        for start in [0.1, 0.5, 1.0, 2.0, 5.0]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
                candidate_sets.append(z)

    # Priority 5: Log-spaced
    for lo, hi in [(0.1, 100), (0.5, 50), (1, 20), (0.1, 10), (1, 10)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Priority 6: Uniform spaced
    for lo, hi in [(0.1, 100), (0.5, 50), (1, 20), (0.1, 10), (1, 10), (5, 100)]:
        z = list(np.linspace(lo, hi, 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Priority 7: Zeros of L_n^(-0.5) for various n, take 12
    for n in [12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40, 50]:
        try:
            zn = laguerre_zeros(n, alpha)
            if len(zn) >= 12:
                sub = zn[:12]
                if all(z > 0 for z in sub) and len(set(np.round(sub, 10))) == 12:
                    candidate_sets.append(list(np.round(sub, 10)))
                sub2 = zn[-12:]
                if all(z > 0 for z in sub2) and len(set(np.round(sub2, 10))) == 12:
                    candidate_sets.append(list(np.round(sub2, 10)))
                if n >= 24:
                    sub3 = zn[::2][:12]
                    if len(sub3) == 12 and all(z > 0 for z in sub3) and len(set(np.round(sub3, 10))) == 12:
                        candidate_sets.append(list(np.round(sub3, 10)))
        except:
            pass

    # Deduplicate candidate sets
    unique_sets = []
    seen = set()
    for cs in candidate_sets:
        key = tuple(round(v, 6) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(cs)

    # Numerical pre-screening: estimate B using float arithmetic before exact sympy
    def numerical_prescreen(z_list):
        """
        Quick numerical estimate of B using float Laguerre polynomials.
        Returns estimated B/(2*pi), or inf if something goes wrong.
        This is a heuristic to avoid expensive exact evaluations.
        """
        try:
            from scipy.special import eval_genlaguerre
        except ImportError:
            return float('inf')

        alpha_val = -0.5
        z_floats = sorted([float(zi) for zi in z_list])

        # Build the 26x26 system numerically
        # L_2k^(alpha)(x) for k=0..25
        # Evaluate at 0, z_i, and derivatives at 0, z_i

        def L_2k(x_val, k):
            return eval_genlaguerre(2*k, alpha_val, x_val)

        def dL_2k(x_val, k):
            # Derivative of L_n^(alpha)(x) = -L_{n-1}^{(alpha+1)}(x)
            if 2*k == 0:
                return 0.0
            return -eval_genlaguerre(2*k - 1, alpha_val + 1, x_val)

        A_rows = []
        b_vec = []

        # g(0) = 0
        row = [L_2k(0.0, k) for k in range(26)]
        A_rows.append(row)
        b_vec.append(0.0)

        # g'(0) = 1
        row = [dL_2k(0.0, k) for k in range(26)]
        A_rows.append(row)
        b_vec.append(1.0)

        for zi in z_floats:
            # g(z_i) = 0
            row = [L_2k(zi, k) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0.0)

            # g'(z_i) = 0
            row = [dL_2k(zi, k) for k in range(26)]
            A_rows.append(row)
            b_vec.append(0.0)

        A_mat = np.array(A_rows)
        b_mat = np.array(b_vec)

        # Check condition number
        try:
            cond = np.linalg.cond(A_mat)
            if cond > 1e15:
                return float('inf')
        except:
            return float('inf')

        # Solve
        try:
            c_sol = np.linalg.solve(A_mat, b_mat)
        except:
            return float('inf')

        # Build g(x) = sum c_k * L_2k(x)
        # We need to find roots of q(x) = g(x) / (x * prod (x-z_i)^2)
        # Numerically, find where g(x) = 0 (excluding the known roots)
        # and divide by the known factor

        # Build the denominator polynomial coefficients
        # denom(x) = x * prod_i (x - z_i)^2
        # This is a degree 25 polynomial

        # Let's construct denom as a polynomial
        # Start with x
        denom_coeffs = [1.0, 0.0]  # x = 1*x + 0

        for zi in z_floats:
            # Multiply by (x - zi)^2 = x^2 - 2*zi*x + zi^2
            sq_coeffs = [1.0, -2.0*zi, zi*zi]
            # Polynomial multiplication
            new_coeffs = [0.0] * (len(denom_coeffs) + len(sq_coeffs) - 1)
            for i, ci in enumerate(denom_coeffs):
                for j, cj in enumerate(sq_coeffs):
                    new_coeffs[i+j] += ci * cj
            denom_coeffs = new_coeffs

        # Now find roots of g(x) - we need to work with the polynomial form
        # This is complex numerically. Instead, let's sample g(x) on a grid
        # and look for sign changes beyond the known roots.

        # Actually, let's just return a rough estimate based on the condition
        # If the system is well-conditioned, B is likely moderate.
        # Use condition number as a proxy: lower cond -> likely lower B

        # For now, return a score based on how "spread out" the z_i are
        # Bessel zeros grow as i^2, which is quite spread
        # This is a very rough heuristic
        max_z = max(z_floats)
        min_z = min(z_floats)
        spread_ratio = max_z / min_z if min_z > 0 else float('inf')

        # Lower spread might be better? Or higher?
        # Let's just use the condition number as a rough indicator
        return cond / 1e10  # Scaled to be in a reasonable range

    # Now evaluate each candidate set
    def evaluate(z_list):
        """Evaluate B/(2*pi) for given z_list using sympy exact arithmetic."""
        import sympy as sp
        x = sp.Symbol('x')
        alpha_val = sp.Rational(-1, 2)

        # Convert z_i to rational
        z_rats = [sp.Rational(float(zi)) for zi in z_list]
        z_rats_sorted = sorted(z_rats)

        # Check distinct and positive
        if len(set(z_rats_sorted)) != 12:
            return float('inf')
        if any(z <= 0 for z in z_rats_sorted):
            return float('inf')

        # Build L_2k^(alpha)(x) for k=0..25
        L_terms = []
        for k in range(26):
            L = sp.laguerre(2*k, x, alpha_val)
            L_terms.append(L)

        # Build the linear system
        A_rows = []
        b_vec = []

        # g(0) = 0
        row = []
        for k in range(26):
            row.append(L_terms[k].subs(x, 0))
        A_rows.append(row)
        b_vec.append(0)

        # g'(0) = 1
        row = []
        for k in range(26):
            row.append(sp.diff(L_terms[k], x).subs(x, 0))
        A_rows.append(row)
        b_vec.append(1)

        for zi in z_rats_sorted:
            # g(z_i) = 0
            row = []
            for k in range(26):
                row.append(L_terms[k].subs(x, zi))
            A_rows.append(row)
            b_vec.append(0)

            # g'(z_i) = 0
            row = []
            for k in range(26):
                row.append(sp.diff(L_terms[k], x).subs(x, zi))
            A_rows.append(row)
            b_vec.append(0)

        A_mat = sp.Matrix(A_rows)
        b_mat = sp.Matrix(b_vec)

        # Check if singular
        if A_mat.det() == 0:
            return float('inf')

        # Solve
        try:
            c_sol = A_mat.LUsolve(b_mat)
        except:
            return float('inf')

        # Build g(x)
        g_sol = sum(c_sol[k] * L_terms[k] for k in range(26))
        g_sol = sp.expand(g_sol)

        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rats_sorted:
            denom = denom * (x - zi)**2
        denom = sp.expand(denom)

        # q(x) = g(x) / denom, exact polynomial division
        q_poly, remainder = sp.div(g_sol, denom, x)
        if remainder != 0:
            # The division should be exact if g has the right roots
            # If not, something is wrong
            return float('inf')

        # q_poly is a polynomial of degree 50 - (1+24) = 25
        # Find real roots
        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

        # For each real root r, check sign change at r - 1e-198 and r + 1e-198
        B = 0
        for r in real_roots:
            if r > 0:
                r_float = float(r)
                delta = 1e-198
                try:
                    val_neg = q_poly.subs(x, r - delta)
                    val_pos = q_poly.subs(x, r + delta)
                    vn = float(val_neg)
                    vp = float(val_pos)
                    if vn * vp < 0:
                        if r_float > B:
                            B = r_float
                except:
                    pass

        return B / (2 * np.pi)

    # Evaluate candidates, starting with the most promising
    for i, cs in enumerate(unique_sets):
        if time_remaining() < 60:
            break
        try:
            # Quick prescreen first
            prescreen_score = numerical_prescreen(cs)

            # Only do full evaluation if prescreen looks promising
            # or if it's one of the first few candidates
            if prescreen_score < float('inf') and (i < 3 or prescreen_score < 100):
                score = evaluate(cs)
                if score < best_score:
                    best_score = score
                    best_z = cs
                    print(f"Candidate {i}: score={score:.6f}, prescreen={prescreen_score:.4f}")
            elif i < 2:
                # Always evaluate the first two candidates (Bessel and Laguerre)
                score = evaluate(cs)
                if score < best_score:
                    best_score = score
                    best_z = cs
                    print(f"Candidate {i}: score={score:.6f} (forced)")
        except Exception as e:
            pass

    if best_z is not None:
        return best_z
    else:
        # Fallback: Bessel zeros
        return bessel_j_half_zeros(12)

# EVOLVE-BLOCK-END
