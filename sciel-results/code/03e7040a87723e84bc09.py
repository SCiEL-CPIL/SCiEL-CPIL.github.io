"""Program for spectral graph lifts on prism graphs using circulant structure."""
import numpy as np
from collections import deque


def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix of the base graph from signs."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(A):
    """Max absolute eigenvalue of symmetric A."""
    vals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(vals)))


def _is_connected_lift(graph, signs):
    """Check whether the 2-lift is connected via BFS labeling."""
    n = graph['n']
    edges = graph['edges']
    adj = {i: [] for i in range(n)}
    for (u, v), s in zip(edges, signs):
        adj[u].append((v, s))
        adj[v].append((u, s))

    label = [-1] * n
    label[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for (v, s) in adj[u]:
            expected = label[u] if s > 0 else 1 - label[u]
            if label[v] == -1:
                label[v] = expected
                q.append(v)
            else:
                if label[v] != expected:
                    return True
    return False


def _build_prism_adj(m, top_s, bot_s, rung_s):
    """Build 2m x 2m signed adjacency for prism graph."""
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    for i in range(m):
        j = (i + 1) % m
        A[i, j] = top_s[i]
        A[j, i] = top_s[i]
        A[m + i, m + j] = bot_s[i]
        A[m + j, m + i] = bot_s[i]
        A[i, m + i] = rung_s[i]
        A[m + i, i] = rung_s[i]
    return A


def _spectral_radius_prism(m, top_s, bot_s, rung_s):
    """Compute spectral radius of prism signed adjacency."""
    A = _build_prism_adj(m, top_s, bot_s, rung_s)
    return _spectral_radius(A)


def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge, minimizing spectral radius
    of the signed adjacency matrix, subject to the 2-lift being connected.
    
    Uses continuous phase-space optimization exploiting circulant structure.
    """
    rng = np.random.default_rng(3)
    n_edges = len(graph['edges'])
    m = graph['n'] // 2
    edges = graph['edges']
    
    # Classify edges
    top_edges = []
    bottom_edges = []
    rung_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                pos = u
            else:
                pos = m - 1
            top_edges.append((idx, pos))
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = m - 1
            bottom_edges.append((idx, pos))
        else:
            i = min(u, v)
            rung_edges.append((idx, i))
    
    top_edges.sort(key=lambda x: x[1])
    bottom_edges.sort(key=lambda x: x[1])
    rung_edges.sort(key=lambda x: x[1])
    
    def eval_continuous(top_s, bot_s, rung_s):
        return _spectral_radius_prism(m, top_s, bot_s, rung_s)
    
    def eval_discrete(signs):
        A = _build_signed_adj(graph, signs)
        return _spectral_radius(A)
    
    def is_valid(signs):
        return _is_connected_lift(graph, signs)
    
    # Continuous optimization over phases
    # top_s[i] = cos(alpha[i]), bot_s[i] = cos(beta[i]), rung_s[i] = cos(gamma[i])
    
    best_obj = np.inf
    best_alpha = None
    best_beta = None
    best_gamma = None
    
    n_trials = 20
    steps_per_trial = 800
    
    for trial in range(n_trials):
        if trial == 0:
            # Alternating pattern
            alpha = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            beta = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            gamma = np.zeros(m)
        elif trial == 1:
            # All +1 except one rung -1
            alpha = np.zeros(m)
            beta = np.zeros(m)
            gamma = np.zeros(m)
            gamma[0] = np.pi
        elif trial == 2:
            # All -1 on cycles, +1 on rungs
            alpha = np.full(m, np.pi)
            beta = np.full(m, np.pi)
            gamma = np.zeros(m)
        elif trial == 3:
            # Mixed: alternating top, alternating bottom, alternating rungs
            alpha = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            beta = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            gamma = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
        else:
            alpha = rng.uniform(0, np.pi, size=m)
            beta = rng.uniform(0, np.pi, size=m)
            gamma = rng.uniform(0, np.pi, size=m)
        
        lr = 0.05
        momentum = 0.85
        vel_a = np.zeros(m)
        vel_b = np.zeros(m)
        vel_g = np.zeros(m)
        
        for step in range(steps_per_trial):
            top_s = np.cos(alpha)
            bot_s = np.cos(beta)
            rung_s = np.cos(gamma)
            
            obj = eval_continuous(top_s, bot_s, rung_s)
            
            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha.copy()
                best_beta = beta.copy()
                best_gamma = gamma.copy()
            
            # Finite difference gradients
            h = 1e-4
            grad_a = np.zeros(m)
            grad_b = np.zeros(m)
            grad_g = np.zeros(m)
            
            for i in range(m):
                ap = alpha.copy(); ap[i] += h
                am = alpha.copy(); am[i] -= h
                op = eval_continuous(np.cos(ap), bot_s, rung_s)
                om = eval_continuous(np.cos(am), bot_s, rung_s)
                grad_a[i] = (op - om) / (2 * h)
                
                bp = beta.copy(); bp[i] += h
                bm = beta.copy(); bm[i] -= h
                op = eval_continuous(top_s, np.cos(bp), rung_s)
                om = eval_continuous(top_s, np.cos(bm), rung_s)
                grad_b[i] = (op - om) / (2 * h)
                
                gp = gamma.copy(); gp[i] += h
                gm = gamma.copy(); gm[i] -= h
                op = eval_continuous(top_s, bot_s, np.cos(gp))
                om = eval_continuous(top_s, bot_s, np.cos(gm))
                grad_g[i] = (op - om) / (2 * h)
            
            vel_a = momentum * vel_a - lr * grad_a
            vel_b = momentum * vel_b - lr * grad_b
            vel_g = momentum * vel_g - lr * grad_g
            
            alpha = np.clip(alpha + vel_a, 0, np.pi)
            beta = np.clip(beta + vel_b, 0, np.pi)
            gamma = np.clip(gamma + vel_g, 0, np.pi)
            
            lr *= 0.997
        
        # Final evaluation
        top_s = np.cos(best_alpha)
        bot_s = np.cos(best_beta)
        rung_s = np.cos(best_gamma)
        obj = eval_continuous(top_s, bot_s, rung_s)
        if obj < best_obj:
            best_obj = obj
    
    # Round to discrete signs
    top_signs = np.where(np.cos(best_alpha) >= 0, 1, -1).astype(int)
    bot_signs = np.where(np.cos(best_beta) >= 0, 1, -1).astype(int)
    rung_signs = np.where(np.cos(best_gamma) >= 0, 1, -1).astype(int)
    
    signs = np.zeros(n_edges, dtype=int)
    for idx, pos in top_edges:
        signs[idx] = top_signs[pos]
    for idx, pos in bottom_edges:
        signs[idx] = bot_signs[pos]
    for idx, pos in rung_edges:
        signs[idx] = rung_signs[pos]
    
    # Ensure connectivity
    if not is_valid(signs):
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if is_valid(cand):
                signs = cand
                break
        else:
            for _ in range(10000):
                cand = rng.choice([-1, 1], size=n_edges).astype(int)
                if is_valid(cand):
                    signs = cand
                    break
    
    # Local search
    current_val = eval_discrete(signs)
    improved = True
    max_iters = 50000
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if not is_valid(cand):
                continue
            v = eval_discrete(cand)
            if v < current_val - 1e-12:
                signs = cand
                current_val = v
                improved = True
                break
    
    # Multiple random restarts with local search
    best_signs = signs.copy()
    best_val = current_val
    
    for restart in range(60):
        cand = rng.choice([-1, 1], size=n_edges).astype(int)
        if not is_valid(cand):
            for i in range(n_edges):
                c2 = cand.copy()
                c2[i] = -c2[i]
                if is_valid(c2):
                    cand = c2
                    break
            else:
                continue
        
        val = eval_discrete(cand)
        
        for _ in range(5000):
            imp = False
            for i in range(n_edges):
                c2 = cand.copy()
                c2[i] = -c2[i]
                if not is_valid(c2):
                    continue
                v2 = eval_discrete(c2)
                if v2 < val - 1e-12:
                    cand = c2
                    val = v2
                    imp = True
                    break
            if not imp:
                break
        
        if val < best_val:
            best_val = val
            best_signs = cand.copy()
    
    return best_signs.astype(int)
