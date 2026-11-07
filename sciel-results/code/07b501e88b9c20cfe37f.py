# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

# ---------------------------------------------------------------------------
# Exact Hermite polynomial coefficients (physicists' convention, from sympy).
# H_n(x) = sum_k h[n][k] * x^k  (only even k present here).
# ---------------------------------------------------------------------------
_H = {
    0:  {0: 1},
    4:  {0: 12, 2: -48, 4: 16},
    8:  {0: 10080, 2: -60480, 4: 30240, 6: -4608, 8: 256},
    12: {0: 10321920, 2: -54432000, 4: 47056320, 6: -15930720, 8: 2257920, 10: -147456, 12: 4096},
}

def _g_coeffs(c0, c1, c2):
    """Return dict a[k] = coefficient of x^(2k) in g(x) for k=0..6 (a[0]=0)."""
    c3 = -(c0 * 1 + c1 * 12 + c2 * 10080) / 10321920.0
    a = [0.0] * 7
    for k in range(1, 7):
        a[k] = (c1 * _H[4].get(k, 0.0)
                + c2 * _H[8].get(k, 0.0)
                + c3 * _H[12].get(k, 0.0))
    return a

def _Q_coeffs(a):
    """Q(y) = g/sqrt(y)^2 -> Q(y)=sum_{j=0}^{5} a[j+1] y^j. Return list [q0..q5]."""
    return [a[j + 1] for j in range(6)]

def _Q_at(y, q):
    # Horner
    v = q[5]
    for j in range(4, -1, -1):
        v = v * y + q[j]
    return v

def _sign_change_roots(q):
    """Return sorted list of non-negative real roots of Q where the sign changes."""
    # degree-5 -> numpy roots
    if abs(q[5]) < 1e-15:
        # drop leading zeros
        cc = list(reversed(q))
        while cc and abs(cc[0]) < 1e-15:
            cc.pop(0)
        if not cc:
            return []
        rr = np.roots(cc)
    else:
        rr = np.roots(np.array(list(reversed(q))))
    out = []
    for z in rr:
        if abs(z.imag) < 1e-9 and z.real >= -1e-9:
            y = max(0.0, z.real)
            # verify sign change
            e = 1e-7 * max(1.0, y)
            lm = _Q_at(y - e, q)
            lp = _Q_at(y + e, q)
            if lm * lp < 0:
                out.append(y)
    return sorted(out)

def _value_from_q(q):
    roots = _sign_change_roots(q)
    if not roots:
        return float('inf')
    return max(roots) / (2.0 * np.pi)

def _value(c0, c1, c2):
    a = _g_coeffs(c0, c1, c2)
    q = _Q_coeffs(a)
    return _value_from_q(q)

# ---------------------------------------------------------------------------
# Non-negative Bernstein basis on [Y, inf):  x^2*A(x)+x*B(x)+C(x) >= 0  on [Y,inf)
# where A,B,C are polynomials of degree <=3,2,1.  Each nonneg poly of degree d
# on [Y,inf) is represented by non-negative Bernstein coeffs b_0..b_d:
#   P(x) = sum_{i=0}^{d} b_i * B_i^d(x),  b_i >= 0
# We convert to monomial basis to read off the contribution to Q's coefficients.
# ---------------------------------------------------------------------------
def _bernstein_monomial(d, Y):
    """Return list M[i] of length d+1 where M[i] = monomial coeffs (degree 0..d)
    of the i-th Bernstein basis polynomial B_i^d on [Y,inf)."""
    # B_i^d(x) = C(d,i) ((x-Y)/x)^i (1/x)^{d-i}  for x>0
    # = C(d,i) * sum_{j=0}^{i} (-1)^j C(i,j) Y^j x^{-(i-j)} * x^{-(d-i)}
    # = C(d,i) * sum_{j=0}^{i} (-1)^j C(i,j) Y^j x^{j-d}
    from math import comb
    M = []
    for i in range(d + 1):
        coeffs = [0.0] * (d + 1)   # coeffs[k] = coefficient of x^k
        for j in range(i + 1):
            term = comb(d, i) * ((-1.0) ** j) * comb(i, j) * (Y ** j)
            # power of x is (j - d)
            p = j - d
            if 0 <= p <= d:
                coeffs[p] += term
        M.append(coeffs)
    return M

def _nonneg_poly_monomial(d, Y, b):
    """b: list of d+1 non-negative Bernstein coeffs. Return monomial coeffs (len d+1)."""
    M = _bernstein_monomial(d, Y)
    out = [0.0] * (d + 1)
    for i in range(d + 1):
        for k in range(d + 1):
            out[k] += b[i] * M[i][k]
    return out

# Q(y) = y^2 A(y) + y B(y) + C(y)
# A deg<=3 (bA 0..3), B deg<=2 (bB 0..2), C deg<=1 (bC 0..1)
# Q monomial coeff q_k (k=0..5):
#   from y^2 A: A_{k-2} for k>=2
#   from y B:   B_{k-1} for k>=1
#   from C:     C_k for k<=1
def _Q_from_bernstein(bA, bB, bC, Y):
    A = _nonneg_poly_monomial(3, Y, bA)   # len4
    B = _nonneg_poly_monomial(2, Y, bB)   # len3
    C = _nonneg_poly_monomial(1, Y, bC)   # len2
    q = [0.0] * 6
    for k in range(6):
        v = 0.0
        if k >= 2:
            v += A[k - 2]
        if k >= 1:
            v += B[k - 1]
        if k <= 1:
            v += C[k]
        q[k] = v
    return q

# ---------------------------------------------------------------------------
# Linear program: find Q with Q(y)>=0 for all y>=Y  (i.e. no sign change for y>=Y),
# minimizing a proxy so the largest sign-changing root below Y is small.
# Variables: bA(0..3), bB(0..2), bC(0..1) all >= 0.
# We minimise t where t >= q_k for the low-order coefficients that push roots up.
# Simpler: just minimise the sum of the "positive push" of Q at sample points in [0,Y]
# is nonlinear.  Instead we directly minimise the largest root via bisection on Y
# and, for feasibility, only require Q>=0 on [Y,inf).  To make it a real optimisation
# (not just feasibility) we add objective = minimise q_0 + q_1 + ... weighted so that
# Q is as "low" as possible near 0, which tends to pull the largest root down.
# ---------------------------------------------------------------------------
def _solve_lp(Y):
    """Return (q, feasible). Minimise weighted low coeffs subject to Q>=0 on [Y,inf)."""
    nA, nB, nC = 4, 3, 2
    nvar = nA + nB + nC
    # Precompute contribution of each variable to each q_k
    # Build M tables
    MA = _bernstein_monomial(3, Y)   # each len4
    MB = _bernstein_monomial(2, Y)   # each len3
    MC = _bernstein_monomial(1, Y)   # each len2
    # c_matrix[k, var] = contribution to q_k
    c = np.zeros((6, nvar))
    # A vars (0..3) -> q_{k} += A_{k-2} for k>=2  => A_m contributes to q_{m+2}
    for i in range(nA):
        for k in range(4):
            c[k + 2, i] += MA[i][k]
    # B vars (4..6) -> q_k += B_{k-1} for k>=1 => B_m contributes to q_{m+1}
    for i in range(nB):
        for k in range(3):
            c[k + 1, nA + i] += MB[i][k]
    # C vars (7..8) -> q_k += C_k for k<=1 => C_m contributes to q_m
    for i in range(nC):
        for k in range(2):
            c[k, nA + nB + i] += MC[i][k]

    # Objective: minimise a weighted sum that discourages Q from being large/positive
    # in the region that creates high roots.  Use weights favouring low q_k.
    # We want Q small (ideally <=0) on [0,Y] so no roots there either, but Q>=0 on [Y,inf)
    # forces Q(Y)>=0.  Minimise total "area" proxy = sum q_k * Y^k (i.e. Q integrated-ish).
    w = np.array([1.0, Y, Y**2, Y**3, Y**4, Y**5])
    c_obj = c.T.dot(w)   # length nvar

    # Bounds: all >= 0
    bounds = [(0.0, None)] * nvar

    # We need at least one degree of freedom; add a tiny requirement that Q is not
    # identically zero by fixing q_5 to be bounded?  Actually Q>=0 on [Y,inf) with all
    # Bernstein coeffs >=0 already ensures Q>=0; the all-zero solution is feasible and
    # optimal (minimises a positive-weighted sum).  To avoid trivial zero we instead
    # MAXIMISE the low-order negativity is impossible with >=0 constraints.
    #
    # New approach: the LP as stated is degenerate (zero solution).  Instead we directly
    # search over Y by bisection for the *smallest* Y such that there exists a Q with
    # Q>=0 on [Y,inf) AND Q has a sign change somewhere in [0,Y].  But that's the hard part.
    #
    # Practical strategy: for a target Y, we want the largest sign-changing root <= Y.
    # Minimise it by making Q as negative as possible just above 0 while Q>=0 on [Y,inf).
    # This is a feasibility+objective problem.  We'll minimise q_0 (the value at 0) which
    # we want negative, subject to Q>=0 on [Y,inf).  Since q_0 = C_0 = bC_0*MC[0][0],
    # and MC[0][0] = C(1,0)*(-1)^0*C(0,0)*Y^0 = 1, so q_0 = bC_0 >= 0.  Can't be negative!
    #
    # Realisation: Q>=0 on [Y,inf) with this SOS form forces Q>=0 everywhere on [Y,inf) but
    # says nothing below Y.  However the SOS representation x^2A+xB+C with A,B,C>=0 on [Y,inf)
    # is only guaranteed on [Y,inf).  Below Y, Q can be anything (negative) because the
    # Bernstein non-negativity only holds for x>=Y.  But q_0 = C(0) and C>=0 on [Y,inf) does
    # NOT force C(0)>=0.  Indeed C(0)=bC_0*MC[0][0] where MC[0][0] is the x^0 coeff of
    # B_0^1 on [Y,inf) = C(1,0)*((x-Y)/x)^0*(1/x)^1 = 1/x, so at x=0 it's singular!
    #
    # The Bernstein basis on [Y,inf) is singular at 0, so the monomial conversion is only
    # valid for x>0 and q_0 is NOT simply bC_0.  Let me recompute: B_0^1(x)=1/x, B_1^1(x)=(x-Y)/x=1-Y/x.
    # C(x)=bC_0*(1/x)+bC_1*(1-Y/x).  At x->0, C-> -inf unless bC_1=0.  So this representation
    # is ill-suited for reading q_0.  The whole [Y,inf) Bernstein approach is getting messy.

    # ABANDON the LP.  Use a robust direct method instead.
    raise RuntimeError("LP path abandoned; using direct search")

# ---------------------------------------------------------------------------
# Robust direct method: parameterise Q to have a small largest root.
# Strategy: we want Q(y) to be a polynomial of degree 5 with Q>=0 for y>=y* (so no sign
# change above y*) and a sign change at y* (the largest one).  The simplest family:
#   Q(y) = (y - y*) * S(y)   where S(y) >= 0 for y >= y* and S has no roots above y*,
#   and Q changes sign at y* (so S(y*)>0 and the (y-y*) factor gives the sign change).
#   For Q to have NO other sign-changing roots above y*, S must be >=0 for all y>=y*.
#   Take S(y) = (y - t)^2 * (y + a) + b  with t,y* chosen so S>=0 on [y*,inf).
#   Simplest: S(y) = (y - y*)^2 + m  (m>0)  -> S>=0 everywhere, S(y*)=m>0.
#   Then Q(y) = (y - y*)((y-y*)^2 + m) = (y-y*)^3 + m(y-y*).
#   This Q has a sign change ONLY at y* (triple-ish: (y-y*)^3 changes sign, m(y-y*) too,
#   sum changes sign at y*).  Largest sign-changing root = y*.  value = y*/(2pi).
#   We just need this Q to be REALISABLE as g/x^2 for some c0,c1,c2.
#
# Q(y) = (y-y*)^3 + m(y-y*) = y^3 - 3y* y^2 + (3y*^2 + m) y - (y*^3 + m y*)
#   q0 = -(y*^3 + m y*), q1 = 3y*^2 + m, q2 = -3y*, q3 = 1, q4 = 0, q5 = 0.
# But we need q4 and q5 to match the structure from c0,c1,c2 (q5 = a6 = c3*4096, q4=a5=0).
# Our family has q5=0 -> c3=0 -> then a6=0 ok, but a5=0 always ok.  With c3=0:
#   a1 = -48 c1 - 60480 c2
#   a2 = 16 c1 + 30240 c2
#   a3 = -4608 c2
#   a4 = 256 c2
#   a5 = 0
#   a6 = 0
# So q = [a1,a2,a3,a4,0,0] = [q0,q1,q2,q3,0,0].  Our family has q5=0,q4=0.  q4=a4=256 c2 must be 0
# -> c2=0.  Then a3=0, a2=16 c1, a1=-48 c1.  So q=[-48c1,16c1,0,0,0,0] = 16c1*(-3,1,0,0,0,0).
# That's Q(y)=16c1(y-3).  Only one free scale.  Largest sign-changing root = 3 (fixed!).  value=3/(2pi).
# Not good; the c3=0 restriction is too tight.
#
# So we MUST use c3 != 0 to get q5 != 0.  Let's use the full family:
#   Q(y) = (y - y*)((y - y*)^2 + m) + q5 y^5 + q4 y^4   (add high-order terms that vanish at y*)
#   Actually to keep largest root = y*, we need Q(y) >= 0 for y > y* (no sign change above).
#   Q(y) = (y-y*)^3 + m(y-y*) + q5 y^5 + q4 y^4.  For large y, q5 y^5 dominates; need q5>0 for Q>0 at inf.
#   This is getting complicated.  Let me just do a DIRECT numerical optimisation over (c0,c1,c2)
#   using a robust derivative-free method but with a MUCH better objective evaluation and
#   a global + local two-stage search, plus the analytic insight to guide starting points.
#
# Given the time budget and that the seed already scores 1.0, a reliable improvement comes
# from a careful global search over the 3 coefficients with the exact value function, using
# multiple random restarts and a local polish.  The value function is piecewise smooth
# (changes only when a root crosses a sign-change threshold), so a dense grid + local polish
# should find a better point than the single Nelder-Mead in the seed.

def _value_safe(c):
    try:
        return _value(float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        return float('inf')

def construct():
    start = time.time()
    deadline = start + 170.0   # safety margin under 180s

    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed.copy()
    best_v = _value_safe(seed)

    rng = np.random.RandomState(12345)

    # ---- Stage 1: broad global search via random restarts + Nelder-Mead ----
    def _nm(x0, iters=300):
        from scipy.optimize import minimize
        def obj(x):
            return _value_safe(x)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': iters, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            return res.x, res.fun
        except Exception:
            return x0, obj(x0)

    # Candidate starting points: seed, and a grid of plausible coefficients.
    starts = [seed.copy()]
    # Grid around seed
    for dc0 in np.linspace(-2.0, 0.5, 7):
        for dc1 in np.linspace(-0.2, 0.4, 7):
            for dc2 in np.linspace(-0.05, 0.15, 5):
                starts.append(seed + np.array([dc0, dc1, dc2]))
    # Random broad
    for _ in range(40):
        starts.append(rng.randn(3) * 0.8)

    # Evaluate all starts, keep best few, polish
    scored = []
    for s in starts:
        if time.time() > deadline - 30.0:
            break
        v = _value_safe(s)
        scored.append((v, s))
    scored.sort(key=lambda t: t[0])

    # Polish the top candidates
    top = scored[:12]
    for v, s in top:
        if time.time() > deadline - 20.0:
            break
        xc, vc = _nm(s, iters=400)
        if vc < best_v:
            best_v = vc
            best_c = xc.copy()

    # ---- Stage 2: fine local polish around the best ----
    if time.time() < deadline - 10.0:
        xc, vc = _nm(best_c, iters=800)
        if vc < best_v:
            best_v = vc
            best_c = xc.copy()

    # ---- Stage 3: tiny jitter search to escape flat regions ----
    if time.time() < deadline - 5.0:
        for _ in range(60):
            if time.time() > deadline - 5.0:
                break
            jitter = rng.randn(3) * 1e-4
            c = best_c + jitter
            v = _value_safe(c)
            if v < best_v:
                best_v = v
                best_c = c.copy()
                xc, vc = _nm(c, iters=300)
                if vc < best_v:
                    best_v = vc
                    best_c = xc.copy()

    # Final safety: ensure finite
    if not np.isfinite(best_v):
        return seed.tolist()
    return [float(best_c[0]), float(best_c[1]), float(best_c[2])]
# EVOLVE-BLOCK-END
