"""Combined program: fast numerical pre-screening + exact sympy evaluation."""
import time
import numpy as np

def construct():
    deadline = time.time() + 620  # Safety margin before 630s kill
    best_score = float('inf')
    best_z = None

    # --- Fast numerical evaluation (proxy) ---
    def evaluate_numeric(z_list):
        """Fast floating-point evaluation to approximate B/(2*pi)."""
        try:
            from scipy.special import eval_genlaguerre
            from scipy.optimize import brentq
            import numpy as np
            
            alpha = -0.5
            z_arr = np.array(sorted(z_list), dtype=np.float64)
            if np.any(z_arr <= 0) or len(set(np.round(z_arr, 12))) != 12:
                return float('inf')
            
            # Build Laguerre polynomials L_{2k}^{alpha}(x) for k=0..25
            # g(x) = sum c_k L_{2k}(x)
            # Conditions: g(0)=0, g'(0)=1, g(z_i)=g'(z_i)=0 for all i
            # 26 equations, 26 unknowns
            
            # Evaluate L_{2k}(x) and L_{2k}'(x) at needed points
            points = np.concatenate([[0.0], z_arr])
            # We need L_{2k}(p) and L_{2k}'(p) for p in points, k=0..25
            
            # L_n^{alpha}(x) derivative: d/dx L_n^{alpha}(x) = -L_{n-1}^{alpha+1}(x)
            # So L_{2k}'(x) = -L_{2k-1}^{alpha+1}(x) for k>=1, and L_0'=0
            
            A_rows = []
            b_vec = []
            
            # g(0) = 0
            row = np.array([eval_genlaguerre(2*k, alpha, 0.0) for k in range(26)])
            A_rows.append(row)
            b_vec.append(0.0)
            
            # g'(0) = 1
            # L_0'(0) = 0, L_{2k}'(0) = -L_{2k-1}^{alpha+1}(0) for k>=1
            row = np.zeros(26)
            row[0] = 0.0
            for k in range(1, 26):
                row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, 0.0)
            A_rows.append(row)
            b_vec.append(1.0)
            
            for zi in z_arr:
                # g(zi) = 0
                row = np.array([eval_genlaguerre(2*k, alpha, zi) for k in range(26)])
                A_rows.append(row)
                b_vec.append(0.0)
                
                # g'(zi) = 0
                row = np.zeros(26)
                row[0] = 0.0
                for k in range(1, 26):
                    row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, zi)
                A_rows.append(row)
                b_vec.append(0.0)
            
            A_mat = np.array(A_rows)
            b_mat = np.array(b_vec)
            
            # Check condition number
            cond = np.linalg.cond(A_mat)
            if cond > 1e15:
                return float('inf')
            
            try:
                c_sol = np.linalg.solve(A_mat, b_mat)
            except:
                return float('inf')
            
            # Build g(x) evaluation function
            def g_func(x_val):
                return sum(c_sol[k] * eval_genlaguerre(2*k, alpha, x_val) for k in range(26))
            
            # Denominator: x * prod (x - z_i)^2
            def denom_func(x_val):
                result = x_val
                for zi in z_arr:
                    result *= (x_val - zi) ** 2
                return result
            
            def q_func(x_val):
                d = denom_func(x_val)
                if abs(d) < 1e-300:
                    return 0.0
                return g_func(x_val) / d
            
            # Find real roots of q(x) by scanning for sign changes
            # q is degree 25, roots likely in (0, max(z)+something)
            x_max = max(z_arr[-1] * 2, 100.0)
            x_min = 1e-6
            
            # Scan for sign changes
            n_scan = 2000
            x_scan = np.linspace(x_min, x_max, n_scan)
            q_scan = np.array([q_func(xi) for xi in x_scan])
            
            # Find sign changes
            signs = np.sign(q_scan)
            sign_changes = []
            for i in range(n_scan - 1):
                if signs[i] * signs[i+1] < 0:
                    # Sign change between x_scan[i] and x_scan[i+1]
                    try:
                        root = brentq(q_func, x_scan[i], x_scan[i+1], xtol=1e-14, rtol=1e-12)
                        sign_changes.append(root)
                    except:
                        pass
            
            # Also check for roots very close to 0
            if len(sign_changes) == 0:
                return 0.0
            
            B = max(sign_changes)
            return B / (2 * np.pi)
            
        except Exception:
            return float('inf')

    # --- Exact sympy evaluation (for final verification) ---
    def evaluate_exact(z_list):
        """Exact rational evaluation using sympy."""
        import sympy as sp
        x = sp.Symbol('x')
        alpha_val = sp.Rational(-1, 2)

        z_rats = sorted([sp.Rational(float(zi)) for zi in z_list])
        if len(set(z_rats)) != 12:
            return float('inf')
        if any(z <= 0 for z in z_rats):
            return float('inf')

        # Precompute L_{2k}^{alpha}(x) and their derivatives
        L_terms = []
        L_derivs = []
        for k in range(26):
            L = sp.laguerre(2*k, x, alpha_val)
            L_terms.append(L)
            L_derivs.append(sp.diff(L, x))

        # Build system matrix
        A_rows = []
        b_vec = []

        # g(0) = 0
        A_rows.append([L_terms[k].subs(x, 0) for k in range(26)])
        b_vec.append(0)

        # g'(0) = 1
        A_rows.append([L_derivs[k].subs(x, 0) for k in range(26)])
        b_vec.append(1)

        for zi in z_rats:
            A_rows.append([L_terms[k].subs(x, zi) for k in range(26)])
            b_vec.append(0)
            A_rows.append([L_derivs[k].subs(x, zi) for k in range(26)])
            b_vec.append(0)

        A_mat = sp.Matrix(A_rows)
        b_mat = sp.Matrix(b_vec)

        if A_mat.det() == 0:
            return float('inf')

        try:
            c_sol = A_mat.LUsolve(b_mat)
        except:
            return float('inf')

        g_sol = sum(c_sol[k] * L_terms[k] for k in range(26))
        g_sol = sp.expand(g_sol)

        denom = x
        for zi in z_rats:
            denom = denom * (x - zi)**2
        denom = sp.expand(denom)

        q_poly, remainder = sp.div(g_sol, denom, x)

        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

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
                    if vn * vp < 0 and r_float > B:
                        B = r_float
                except:
                    pass

        return B / (2 * np.pi)

    # --- Generate candidate sets ---
    # Reduced candidate pool to ensure fast numerical screening
    candidates = []

    # Strategy 1: Arithmetic progression (subset)
    for start in [0.5, 1.0, 2.0]:
        for step in [0.5, 1.0, 1.5]:
            z = [start + step * i for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                candidates.append(z)

    # Strategy 2: Geometric progression (subset)
    for start in [0.5, 1.0]:
        for ratio in [1.1, 1.2, 1.3]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                candidates.append(z)

    # Strategy 3: Power-law (subset)
    for c in [0.5, 1.0]:
        for p in [0.8, 1.0, 1.2]:
            z = [c * ((i + 1) ** p) for i in range(12)]
            if all(zi > 0 for zi in z) and len(set(z)) == 12:
                candidates.append(z)

    # Strategy 4: Log-spaced (subset)
    for lo, hi in [(0.5, 10), (1.0, 20)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        if all(zi > 0 for zi in z) and len(set(z)) == 12:
            candidates.append(z)

    # Strategy 5: Zeros of Laguerre polynomials
    def laguerre_zeros_fast(n, alpha):
        diag = np.array([2.0*k - 1.0 + alpha for k in range(1, n+1)])
        subdiag = np.array([np.sqrt(k*(k+alpha)) for k in range(1, n)])
        J = np.diag(diag)
        for i in range(n-1):
            J[i, i+1] = subdiag[i]
            J[i+1, i] = subdiag[i]
        return np.sort(np.linalg.eigvalsh(J))

    alpha = -0.5
    for n in [12, 14, 16]:
        try:
            zn = laguerre_zeros_fast(n, alpha)
            if len(zn) >= 12 and all(z > 0 for z in zn[:12]):
                candidates.append(list(zn[:12]))
            if len(zn) >= 12 and all(z > 0 for z in zn[-12:]):
                candidates.append(list(zn[-12:]))
        except:
            pass

    # Strategy 6: Seed and scaled seed
    candidates.append([4.0 * (i + 1) for i in range(12)])  # seed
    candidates.append([2.0 * (i + 1) for i in range(12)])
    candidates.append([1.0 * (i + 1) for i in range(12)])

    # Deduplicate
    unique_candidates = []
    seen = set()
    for cs in candidates:
        key = tuple(round(v, 4) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cs)

    # --- Phase 1: Fast numerical screening ---
    numeric_scores = []
    for i, cs in enumerate(unique_candidates):
        if time.time() > deadline - 120:  # Reserve 120s for exact eval
            break
        score = evaluate_numeric(cs)
        numeric_scores.append((score, cs))
        if i % 20 == 0:
            valid_scores = [s for s, _ in numeric_scores if s < float('inf')]
            best_so_far = min(valid_scores) if valid_scores else float('inf')
            print(f"Numeric screened {i+1}/{len(unique_candidates)}, best so far: {best_so_far:.6f}")

    # Sort by numeric score
    numeric_scores.sort(key=lambda x: x[0])

    # --- Phase 2: Exact evaluation of top candidates ---
    # Evaluate top N candidates with exact sympy
    # Limit to 2 to save time for the heavy symbolic algebra
    max_exact_evals = 2
    evaluated = 0
    for score_num, cs in numeric_scores:
        if time.time() > deadline - 60:  # Reserve more time for safety
            break
        if evaluated >= max_exact_evals:
            break
        if score_num >= float('inf'):
            continue
        try:
            score_exact = evaluate_exact(cs)
            evaluated += 1
            if score_exact < best_score:
                best_score = score_exact
                best_z = cs
                print(f"Exact eval {evaluated}: score={score_exact:.6f}, z={cs[:3]}...")
        except Exception as e:
            print(f"Exact eval error: {e}")

    # If no exact evaluation succeeded, return best numeric candidate
    # The numeric screen ensures they are likely non-singular
    if best_z is None:
        valid_numeric = [cs for s, cs in numeric_scores if s < float('inf')]
        if valid_numeric:
            # Return the one with the best numeric score
            best_z = valid_numeric[0]
        else:
            # Fallback to seed
            best_z = [4.0 * (i + 1) for i in range(12)]
    else:
        # We have an exact score, but we should ensure best_z corresponds to the best exact score
        # The loop above updates best_z only when score_exact < best_score, so it's correct.
        pass

    return best_z
