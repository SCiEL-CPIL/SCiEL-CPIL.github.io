import time
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m_val = n // 2

    # Precompute edge indices
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i

    # Precompute edge arrays for fast objective
    eu = np.array([u for u, v in edges], dtype=np.int32)
    ev = np.array([v for u, v in edges], dtype=np.int32)

    def build_S(signs):
        S = np.zeros((n, n), dtype=np.float64)
        S[eu, ev] = signs
        S[ev, eu] = signs
        return S

    def get_objective(signs):
        S = build_S(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))

    def is_connected(signs):
        # Union-find on 2n lift vertices
        parent = list(range(2 * n))
        rank = [0] * (2 * n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for i, (u, v) in enumerate(edges):
            s = signs[i]
            u0, u1 = 2 * u, 2 * u + 1
            v0, v1 = 2 * v, 2 * v + 1
            if s == 1:
                union(u0, v0)
                union(u1, v1)
            else:
                union(u0, v1)
                union(u1, v0)

        root = find(0)
        return all(find(i) == root for i in range(2 * n))

    # Precompute 4-cycles (side faces of prism)
    cycles_4 = []
    for i in range(m_val):
        v1 = i
        v2 = (i + 1) % m_val
        v3 = m_val + i
        v4 = m_val + (i + 1) % m_val
        e1 = edge_idx.get((v1, v2))
        e2 = edge_idx.get((v2, v4))
        e3 = edge_idx.get((v4, v3))
        e4 = edge_idx.get((v3, v1))
        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
            cycles_4.append([e1, e2, e3, e4])

    # Also precompute top cycle and bottom cycle edge lists
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            top_cycle_edges.append(i)
        elif u >= m_val and v >= m_val:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)

    rng = np.random.default_rng(3)

    best_signs = None
    best_obj = float('inf')

    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj:
            best_obj = obj
            best_signs = signs.copy()

    def local_search(initial_signs, time_limit=None):
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline
        current_signs = initial_signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        while time.time() < time_limit:
            improved = False

            # 1. Single flips
            best_flip_idx = -1
            best_flip_obj = current_obj
            for i in range(num_edges):
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < best_flip_obj - 1e-12:
                        best_flip_obj = obj
                        best_flip_idx = i
                current_signs[i] = old

            if best_flip_idx != -1:
                current_signs[best_flip_idx] = -current_signs[best_flip_idx]
                current_obj = best_flip_obj
                update_best(current_signs, current_obj)
                improved = True
            else:
                # 2. 4-cycle flips (preserve connectivity)
                best_cycle_idx = -1
                best_cycle_obj = current_obj
                for c_idx, cyc in enumerate(cycles_4):
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < best_cycle_obj - 1e-12:
                        best_cycle_obj = obj
                        best_cycle_idx = c_idx
                    for e in cyc:
                        current_signs[e] = -current_signs[e]

                if best_cycle_idx != -1:
                    for e in cycles_4[best_cycle_idx]:
                        current_signs[e] = -current_signs[e]
                    current_obj = best_cycle_obj
                    update_best(current_signs, current_obj)
                    improved = True
                else:
                    # 3. Top/bottom cycle flips
                    found = False
                    for cyc in [top_cycle_edges, bottom_cycle_edges]:
                        if len(cyc) == 0:
                            continue
                        for e in cyc:
                                current_signs[e] = -current_signs[e]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-12:
                                current_obj = obj
                                update_best(current_signs, current_obj)
                                found = True
                                improved = True
                                break
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                    if found:
                        continue

                    # 4. Pair flips (systematic, limited)
                    best_pair = None
                    best_pair_obj = current_obj
                    for a in range(num_edges):
                        if time.time() > time_limit:
                            break
                        for b in range(a + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < best_pair_obj - 1e-12:
                                    best_pair_obj = obj
                                    best_pair = (a, b)
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]

                    if best_pair is not None:
                        a, b = best_pair
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        current_obj = best_pair_obj
                        update_best(current_signs, current_obj)
                        improved = True
                    else:
                        # 5. Random triple flips
                        found_triple = False
                        for _ in range(20):
                            if time.time() > time_limit:
                                break
                            idxs = rng.choice(num_edges, size=3, replace=False)
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-12:
                                    current_obj = obj
                                    update_best(current_signs, current_obj)
                                    found_triple = True
                                    improved = True
                                    break
                            for e in idxs:
                                current_signs[e] = -current_signs[e]
                        if not found_triple:
                            break

    def polish(signs):
        nonlocal best_signs, best_obj
        current_signs = signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)

        for _ in range(60):
            if time.time() > deadline - 1.0:
                break
            improved = False
            # Single flip sweep
            for i in range(num_edges):
                if time.time() > deadline - 1.0:
                    break
                old = current_signs[i]
                current_signs[i] = -old
                if is_connected(current_signs):
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        current_signs[i] = old
                else:
                    current_signs[i] = old
            if not improved:
                # 4-cycle sweep
                for cyc in cycles_4:
                    if time.time() > deadline - 1.0:
                        break
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                    obj = get_objective(current_signs)
                    if obj < current_obj - 1e-15:
                        current_obj = obj
                        improved = True
                    else:
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
            if not improved:
                # Pair sweep
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                        if is_connected(current_signs):
                            obj = get_objective(current_signs)
                            if obj < current_obj - 1e-15:
                                current_obj = obj
                                improved = True
                                break
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
                    if improved:
                        break
            if not improved:
                # Triple flip sweep (limited)
                for a in range(num_edges):
                    if time.time() > deadline - 1.0:
                        break
                    for b in range(a + 1, num_edges):
                        if time.time() > deadline - 1.0:
                            break
                        for c in range(b + 1, num_edges):
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                            if is_connected(current_signs):
                                obj = get_objective(current_signs)
                                if obj < current_obj - 1e-15:
                                    current_obj = obj
                                    improved = True
                                    break
                            current_signs[a] = -current_signs[a]
                            current_signs[b] = -current_signs[b]
                            current_signs[c] = -current_signs[c]
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        update_best(current_signs, current_obj)

    # ====== SDP-based spectral norm minimization ======
    def sdp_optimize():
        """
        Formulate as: minimize t such that -t*I <= S(x) <= t*I,
        where S(x) = sum_i x_i * (E_{u_i,v_i} + E_{v_i,u_i}).
        
        We solve the SDP relaxation:
        min t
        s.t.  [[t*I, A(x)], [A(x)^T, I]] >= 0  (Schur complement for ||A||_2 <= t)
        
        Actually simpler: minimize max eigenvalue of S(x) in absolute value.
        Use the fact that ||S||_2 = max_{||v||=1} v^T S v (for symmetric S).
        
        We'll use a penalty method with scipy to minimize the spectral radius
        over continuous x in [-1,1]^E, then round.
        """
        from scipy.optimize import minimize
        
        def objective(x):
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev = np.linalg.eigvalsh(S)
            return float(np.max(np.abs(ev)))
        
        def gradient(x):
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev, evec = np.linalg.eigh(S)
            abs_ev = np.abs(ev)
            max_val = np.max(abs_ev)
            # Gradient of max|eig| w.r.t. x_i
            # For the eigenvalue achieving max|.|, grad = sign(ev_k) * 2 * v_k[u]*v_k[v]
            k = np.argmax(abs_ev)
            sign_k = np.sign(ev[k])
            v_k = evec[:, k]
            grad = sign_k * 2.0 * (v_k[eu] * v_k[ev])
            return grad
        
        # Also try minimizing sum of squares of eigenvalues (Frobenius norm) as a proxy
        def frob_objective(x):
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            ev = np.linalg.eigvalsh(S)
            return float(np.sum(ev**2))
        
        def frob_gradient(x):
            S = np.zeros((n, n), dtype=np.float64)
            S[eu, ev] = x
            S[ev, eu] = x
            # d/dx_i sum(ev^2) = 2 * tr(S * dS/dx_i) = 2 * sum_jk S[jk] * dS[jk]/dx_i
            # dS[jk]/dx_i = 1 if (j,k)==(u_i,v_i) or (v_i,u_i), else 0
            # So grad_i = 2 * (S[u_i,v_i] + S[v_i,u_i]) = 2 * 2 * x_i = 4*x_i
            # Wait, that's for Frobenius of S, not sum of eigenvalues squared.
            # Actually sum(ev^2) = tr(S^2) = sum_jk S[jk]*S[kj]
            # d/dx_i tr(S^2) = 2 * sum_jk S[jk] * dS[kj]/dx_i
            # dS[kj]/dx_i = 1 if (k,j) = (u_i,v_i) or (v_i,u_i)
            # So = 2 * (S[v_i,u_i] + S[u_i,v_i]) = 2 * (x_i + x_i) = 4*x_i
            grad = 4.0 * x
            return grad
        
        best_x = None
        best_f = float('inf')
        
        # Multiple initializations
        inits = [
            np.ones(num_edges),
            -np.ones(num_edges),
            rng.uniform(-1, 1, size=num_edges),
            rng.uniform(-1, 1, size=num_edges),
            rng.uniform(-1, 1, size=num_edges),
        ]
        # Structured: alternate on top cycle
        x_struct = np.ones(num_edges)
        for i, (u, v) in enumerate(edges):
            if u < m_val and v < m_val:
                x_struct[i] = 1.0 if (u % 2 == 0) else -1.0
            elif u >= m_val and v >= m_val:
                x_struct[i] = 1.0 if ((u - m_val) % 2 == 0) else -1.0
        inits.append(x_struct)
        
        # Structured: alternate on matching
        x_match = np.ones(num_edges)
        for i, (u, v) in enumerate(edges):
            if u < m_val and v >= m_val:
                x_match[i] = 1.0 if (u % 2 == 0) else -1.0
        inits.append(x_match)
        
        for x0 in inits:
            if time.time() > deadline - 25.0:
                break
            # Try spectral radius objective
            try:
                res = minimize(
                    objective, x0, jac=gradient,
                    method='L-BFGS-B',
                    bounds=[(-1, 1)] * num_edges,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                f = res.fun
                if f < best_f:
                    best_f = f
                    best_x = res.x.copy()
            except Exception:
                pass
            
            # Also try Frobenius objective (smoother)
            try:
                res2 = minimize(
                    frob_objective, x0, jac=frob_gradient,
                    method='L-BFGS-B',
                    bounds=[(-1, 1)] * num_edges,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                # Evaluate the actual spectral radius of the result
                S2 = np.zeros((n, n), dtype=np.float64)
                S2[eu, ev] = res2.x
                S2[ev, eu] = res2.x
                ev2 = np.linalg.eigvalsh(S2)
                f2 = float(np.max(np.abs(ev2)))
                if f2 < best_f:
                    best_f = f2
                    best_x = res2.x.copy()
            except Exception:
                pass
        
        if best_x is not None:
            # Gaussian randomization: sample signs based on the continuous solution
            # sign_i = +1 with probability related to x_i
            # Use: P(sign_i = +1) = (1 + x_i) / 2
            best_signs_from_sdp = None
            best_sdp_obj = float('inf')
            
            n_samples = 200
            for _ in range(n_samples):
                if time.time() > deadline - 20.0:
                    break
                # Sample signs using the continuous solution as bias
                probs = (1.0 + best_x) / 2.0
                probs = np.clip(probs, 0.01, 0.99)
                signs = np.where(rng.random(num_edges) < probs, 1, -1).astype(np.int8)
                
                if is_connected(signs):
                    obj = get_objective(signs)
                    if obj < best_sdp_obj:
                        best_sdp_obj = obj
                        best_signs_from_sdp = signs.copy()
            
            if best_signs_from_sdp is not None:
                return best_signs_from_sdp, best_sdp_obj
        
        return None, float('inf')

    # ====== Exhaustive search for small m ======
    def exhaustive_search():
        """Exhaustive search over all 2^E sign assignments. Only feasible for small E."""
        if num_edges > 16:
            return None, float('inf')
        best = None
        best_f = float('inf')
        for bits in product([1, -1], repeat=num_edges):
            signs = np.array(bits, dtype=np.int8)
            if not is_connected(signs):
                continue
            f = get_objective(signs)
            if f < best_f:
                best_f = f
                best = signs.copy()
        return best, best_f

    # ====== Simulated Annealing ======
    def simulated_annealing(initial_signs, time_limit=None):
        """SA with connectivity constraint, using Metropolis criterion."""
        nonlocal best_signs, best_obj
        if time_limit is None:
            time_limit = deadline - 5.0
        
        current_signs = initial_signs.copy()
        if not is_connected(current_signs):
            return
        current_obj = get_objective(current_signs)
        update_best(current_signs, current_obj)
        
        T_start = 0.1
        T_end = 0.001
        cooling = 0.999
        
        T = T_start
        while time.time() < time_limit and T > T_end:
            # Pick a random move: single flip, 4-cycle flip, or pair flip
            move_type = rng.integers(0, 3)
            
            if move_type == 0:
                # Single flip
                i = rng.integers(0, num_edges)
                current_signs[i] = -current_signs[i]
            elif move_type == 1 and cycles_4:
                # 4-cycle flip
                cyc = cycles_4[rng.integers(0, len(cycles_4))]
                for e in cyc:
                    current_signs[e] = -current_signs[e]
            else:
                # Pair flip
                a, b = rng.choice(num_edges, size=2, replace=False)
                current_signs[a] = -current_signs[a]
                current_signs[b] = -current_signs[b]
            
            if is_connected(current_signs):
                new_obj = get_objective(current_signs)
                delta = new_obj - current_obj
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current_obj = new_obj
                    update_best(current_signs, current_obj)
                else:
                    # Revert
                    if move_type == 0:
                        current_signs[i] = -current_signs[i]
                    elif move_type == 1:
                        for e in cyc:
                            current_signs[e] = -current_signs[e]
                    else:
                        current_signs[a] = -current_signs[a]
                        current_signs[b] = -current_signs[b]
            else:
                # Revert if disconnected
                if move_type == 0:
                    current_signs[i] = -current_signs[i]
                elif move_type == 1:
                    for e in cyc:
                        current_signs[e] = -current_signs[e]
                else:
                    current_signs[a] = -current_signs[a]
                    current_signs[b] = -current_signs[b]
            
            T *= cooling

    # ====== Main search ======

    # 1. Exhaustive search for m=4 (num_edges = 3*4 = 12, 2^12 = 4096 - feasible)
    if m_val == 4:
        ex_signs, ex_f = exhaustive_search()
        if ex_signs is not None:
            update_best(ex_signs, ex_f)

    # 2. SDP-based optimization with Gaussian randomization
    sdp_signs, sdp_f = sdp_optimize()
    if sdp_signs is not None:
        update_best(sdp_signs, sdp_f)
        # Refine with local search
        local_search(sdp_signs, deadline - 15.0)
        polish(sdp_signs)

    # 3. Simulated Annealing from multiple starts
    sa_starts = [
        np.ones(num_edges, dtype=np.int8),
        -np.ones(num_edges, dtype=np.int8),
        rng.choice([-1, 1], size=num_edges).astype(np.int8),
        rng.choice([-1, 1], size=num_edges).astype(np.int8),
    ]
    # Alternating top
    sa_alt = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            sa_alt[i] = 1 if (u % 2 == 0) else -1
    sa_starts.append(sa_alt)
    
    for start in sa_starts:
        if time.time() > deadline - 10.0:
            break
        simulated_annealing(start, deadline - 10.0)

    # 4. Standard local search from multiple initializations
    signs = np.ones(num_edges, dtype=np.int8)
    if is_connected(signs):
        local_search(signs, deadline - 8.0)

    # Alternating top
    signs = np.ones(num_edges, dtype=np.int8)
    for i, (u, v) in enumerate(edges):
        if u < m_val and v < m_val:
            signs[i] = 1 if (u % 2 == 0) else -1
    if is_connected(signs):
        local_search(signs, deadline - 5.0)

    # Random restarts with local search
    for _ in range(100):
        if time.time() > deadline - 3.0:
            break
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        local_search(signs, deadline - 3.0)

    # Final polish
    if best_signs is not None:
        polish(best_signs)

    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)

    # Verify final result is valid
    if not is_connected(best_signs):
        # Fallback: try all +1
        fallback = np.ones(num_edges, dtype=np.int8)
        if is_connected(fallback):
            best_signs = fallback
        else:
            # Try alternating patterns
            for pattern in range(4):
                test = np.ones(num_edges, dtype=np.int8)
                for i, (u, v) in enumerate(edges):
                    if (u // m_val) == (v // m_val):  # same cycle
                        if (u % 2) != (pattern % 2):
                            test[i] = -1
                if is_connected(test):
                    best_signs = test
                    break

    return best_signs
