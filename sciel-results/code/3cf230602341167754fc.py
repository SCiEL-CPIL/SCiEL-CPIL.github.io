"""Circle packing in unit square - 26 circles maximizing sum of radii.

STRATEGY (structurally different from parent):
  The parent uses smooth soft-plus collision penalties + L-BFGS-B + LP radius
  fixes.  This program instead uses a *contact-graph / rigid-body* formulation:
  a packing is locally optimal when a subset of contacts (circle-circle and
  circle-wall) is *active* (exactly touching).  We pick a random set of 78
  contacts (26 circles -> 78 variables), write the resulting 78 equality
  constraints (dist(i,j) = r_i + r_j, or wall distance = r), and solve the
  nonlinear system with scipy least_squares / fsolve.  The search is then
  *combinatorial*: which contacts are active.  This is a different modelling
  family (equality constrained rigid-body / truss-like structure) and a
  different solver (root-finding on a sparse system) than gradient penalty
  optimization.
"""

import numpy as np
import time
import random
from scipy.optimize import least_squares, fsolve
from scipy.spatial import cKDTree

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-8):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.07
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.38 / side
        circles.append((x, y, r))
    return circles


def _init_corner_biased():
    circles = []
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    for cx, cy in corners:
        circles.append((cx, cy, 0.08))
    edges = [(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    for ex, ey in edges:
        circles.append((ex, ey, 0.06))
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
    if seed is not None:
        random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _contact_residual(p, contacts, n=N):
    """
    Residual for a chosen contact set.  p has 3n entries (x,y,r per circle).
    Each contact is either:
      ('cc', i, j)  ->  sqrt((xi-xj)^2+(yi-yj)^2) - (ri + rj) = 0
      ('wall', i, side)  side in {0:'L',1:'R',2:'B',3:'T'}
         L: ri - xi = 0 ; R: 1 - (xi+ri) = 0 ; B: ri - yi = 0 ; T: 1 - (yi+ri) = 0
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    res = np.zeros(len(contacts))
    for k, c in enumerate(contacts):
        if c[0] == 'cc':
            _, i, j = c
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-14)
            res[k] = d - (r[i] + r[j])
        else:
            _, i, side = c
            if side == 0:      # left
                res[k] = r[i] - x[i]
            elif side == 1:    # right
                res[k] = 1.0 - (x[i] + r[i])
            elif side == 2:    # bottom
                res[k] = r[i] - y[i]
            else:              # top
                res[k] = 1.0 - (y[i] + r[i])
    return res


def _solve_contact_system(x0, contacts, deadline, xtol=1e-9, max_nfev=400):
    """Solve the contact equality system; return solution or None."""
    if time.time() > deadline:
        return None
    try:
        res = least_squares(
            _contact_residual, x0, args=(contacts,),
            method='lm', xtol=xtol, ftol=1e-9, gtol=1e-9, max_nfev=max_nfev
        )
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None


def _nearest_wall_side(x, y):
    """Return (side, dist) of nearest wall for a point (used for seeding)."""
    dL, dR, dB, dT = x, 1.0 - x, y, 1.0 - y
    m = min(dL, dR, dB, dT)
    if m == dL:
        return 0, dL
    if m == dR:
        return 1, dR
    if m == dB:
        return 2, dB
    return 3, dT


def _sample_contact_set(circles, n_contacts, rng):
    """
    Sample a contact set of size n_contacts from the current geometry.
    Preference: nearest-neighbor circle pairs + nearest walls.
    Returns list of contact tuples.
    """
    n = N
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])

    # candidate circle-circle pairs: use k-d tree nearest neighbors
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=6)  # 5 nearest + self
    cc_candidates = []
    seen = set()
    for i in range(n):
        for j in idx[i]:
            if j == i:
                continue
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                cc_candidates.append(key)
    rng.shuffle(cc_candidates)

    # candidate wall contacts: nearest wall per circle
    wall_candidates = []
    for i in range(n):
        side, _ = _nearest_wall_side(x[i], y[i])
        wall_candidates.append((i, side))

    contacts = []
    used_pairs = set()
    used_walls = set()

    # First, greedily take promising cc contacts
    for (i, j) in cc_candidates:
        if len(contacts) >= n_contacts:
            break
        if i in used_pairs or j in used_pairs:
            # allow a circle to be in at most 2-3 contacts; keep simple: skip if already used
            continue
        contacts.append(('cc', i, j))
        used_pairs.add(i)
        used_pairs.add(j)

    # Then fill with wall contacts for circles not yet used
    for (i, side) in wall_candidates:
        if len(contacts) >= n_contacts:
            break
        if i in used_pairs:
            continue
        if (i, side) in used_walls:
            continue
        contacts.append(('wall', i, side))
        used_walls.add((i, side))
        used_pairs.add(i)

    # If still short, add random cc pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)
    for (i, j) in all_pairs:
        if len(contacts) >= n_contacts:
            break
        if i in used_pairs or j in used_pairs:
            continue
        contacts.append(('cc', i, j))
        used_pairs.add(i)
        used_pairs.add(j)

    # Last resort: random walls
    for i in range(n):
        if len(contacts) >= n_contacts:
            break
        for side in range(4):
            if (i, side) in used_walls:
                continue
            contacts.append(('wall', i, side))
            used_walls.add((i, side))
            break

    return contacts[:n_contacts]


def _refine_positions(circles, contacts, deadline, iters=6):
    """
    Alternating: fix radii, move positions to satisfy contacts (least squares);
    then fix positions, set radii to satisfy contacts.  Iterate.
    Returns refined circles or None.
    """
    cur = np.array([v for c in circles for v in c], dtype=float)
    n = N
    for _ in range(iters):
        if time.time() > deadline:
            break
        # Substep A: fix radii, optimize positions so contacts hold
        r = cur[2::3].copy()
        # Build position-only residual
        def pos_res(p2):
            x = p2[0::2]
            y = p2[1::2]
            res = np.zeros(len(contacts))
            for k, c in enumerate(contacts):
                if c[0] == 'cc':
                    _, i, j = c
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    res[k] = np.sqrt(dx*dx + dy*dy + 1e-14) - (r[i] + r[j])
                else:
                    _, i, side = c
                    if side == 0:
                        res[k] = r[i] - x[i]
                    elif side == 1:
                        res[k] = 1.0 - (x[i] + r[i])
                    elif side == 2:
                        res[k] = r[i] - y[i]
                    else:
                        res[k] = 1.0 - (y[i] + r[i])
            return res
        p20 = np.array([cur[3*i] for i in range(n)] + [cur[3*i+1] for i in range(n)])
        try:
            ra = least_squares(pos_res, p20, method='lm', xtol=1e-9, max_nfev=200)
            p2 = ra.x
            newx = p2[0::2]
            newy = p2[1::2]
            # Substep B: fix positions, set radii to satisfy contacts
            # For each circle, radius is constrained by its wall contacts and
            # circle contacts.  Use least squares over radii.
            def rad_res(r2):
                res = np.zeros(len(contacts))
                for k, c in enumerate(contacts):
                    if c[0] == 'cc':
                        _, i, j = c
                        dx = newx[i] - newx[j]
                        dy = newy[i] - newy[j]
                        d = np.sqrt(dx*dx + dy*dy + 1e-14)
                        res[k] = d - (r2[i] + r2[j])
                    else:
                        _, i, side = c
                        if side == 0:
                            res[k] = r2[i] - newx[i]
                        elif side == 1:
                            res[k] = 1.0 - (newx[i] + r2[i])
                        elif side == 2:
                            res[k] = r2[i] - newy[i]
                        else:
                            res[k] = 1.0 - (newy[i] + r2[i])
                return res
            rb = least_squares(rad_res, r, method='lm', xtol=1e-9, max_nfev=200)
            r_new = rb.x
            # assemble
            cur[0::3] = newx
            cur[1::3] = newy
            cur[2::3] = r_new
        except Exception:
            break
    # clip
    for i in range(n):
        r = max(0.0, cur[3*i+2])
        cur[3*i] = min(1.0 - r, max(r, cur[3*i]))
        cur[3*i+1] = min(1.0 - r, max(r, cur[3*i+1]))
        cur[3*i+2] = r
    return cur


def _radius_growth_pass(circles, deadline):
    """
    Fix positions (tiny tolerance) and maximize radii via SLSQP with hard
    constraints.  This squeezes extra sum-of-radii from gaps.
    """
    params = np.array([val for c in circles for val in c])
    n = N
    dim = 3 * n
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.45)] * n
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    def objective(p):
        return -sum(p[3*i+2] for i in range(n))
    def jac(p):
        g = np.zeros(dim)
        g[2::3] = -1.0
        return g
    try:
        from scipy.optimize import minimize
        result = minimize(objective, params, jac=jac, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(n)]
            return new_circles
    except Exception:
        pass
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corner_biased(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
    ]

    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)

    rng = random.Random(2024)
    n_contacts = 3 * N  # 78, matching 78 variables

    # Phase 1: contact-system root finding from each init, with many random
    # contact-set samples.  This is the combinatorial search core.
    for idx, init_circles in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        base = np.array([v for c in init_circles for v in c], dtype=float)

        # Try a batch of random contact sets
        for trial in range(40):
            if time.time() > deadline - 30.0:
                break
            contacts = _sample_contact_set(init_circles, n_contacts, rng)
            # add a small random perturbation to the start to avoid exact degeneracy
            p0 = base + rng.gauss(0, 0.002, size=3 * N)
            sol = _solve_contact_system(p0, contacts, deadline)
            if sol is None:
                continue
            # refine with alternating position/radii solve
            refined = _refine_positions(
                [(sol[3*i], sol[3*i+1], sol[3*i+2]) for i in range(N)],
                contacts, deadline, iters=4
            )
            if refined is None:
                continue
            cand = [(refined[3*i], refined[3*i+1], refined[3*i+2]) for i in range(N)]
            if _is_valid(cand):
                s = _sum_radii(cand)
                if s > best_sum:
                    best_circles = cand
                    best_sum = s
                    # re-seed the contact sampling from the improved geometry
                    init_circles = cand

    # Phase 2: radius growth pass (SLSQP, fixed positions) to squeeze gaps
    if time.time() < deadline - 15.0:
        grown = _radius_growth_pass(best_circles, deadline)
        if _is_valid(grown) and _sum_radii(grown) > best_sum:
            best_circles = grown
            best_sum = _sum_radii(grown)

    # Phase 3: a second round of contact-system search from the improved state
    if time.time() < deadline - 10.0:
        base = np.array([v for c in best_circles for v in c], dtype=float)
        for trial in range(30):
            if time.time() > deadline - 10.0:
                break
            contacts = _sample_contact_set(best_circles, n_contacts, rng)
            p0 = base + rng.gauss(0, 0.001, size=3 * N)
            sol = _solve_contact_system(p0, contacts, deadline)
            if sol is None:
                continue
            refined = _refine_positions(
                [(sol[3*i], sol[3*i+1], sol[3*i+2]) for i in range(N)],
                contacts, deadline, iters=4
            )
            if refined is None:
                continue
            cand = [(refined[3*i], refined[3*i+1], refined[3*i+2]) for i in range(N)]
            if _is_valid(cand):
                s = _sum_radii(cand)
                if s > best_sum:
                    best_circles = cand
                    best_sum = s
                    base = np.array([v for c in cand for v in c], dtype=float)

    # Phase 4: final radius growth
    if time.time() < deadline - 5.0:
        grown = _radius_growth_pass(best_circles, deadline)
        if _is_valid(grown) and _sum_radii(grown) > best_sum:
            best_circles = grown
            best_sum = _sum_radii(grown)

    # Post-process
    best_circles = _post_process(best_circles)

    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid())

    return np.array(best_circles)

# EVOLVE-BLOCK-END
