import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit

    def ratio(h):
        h = np.asarray(h, dtype=float)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Autoconvolution via FFT for speed
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
        sup_conv = np.max(conv)
        if sup_conv == 0:
            return 0.0
        return l2_sq / (l1 * sup_conv)

    # The ratio is scale-invariant: ratio(c*h) = ratio(h) for c>0.
    # So we can fix sum(h)=1 and optimize over the simplex.
    # With sum(h)=1, ratio = sum(h^2) / max(h*h)
    
    best_val = 0.0
    best_h = None

    def try_candidate(h):
        nonlocal best_val, best_h
        h = np.asarray(h, dtype=float)
        h = np.clip(h, 0, None)
        s = np.sum(h)
        if s == 0:
            return
        h = h / s  # normalize to simplex
        val = ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()

    # Strategy: The ratio with sum=1 is sum(h^2)/max(conv).
    # To maximize, we want h to be "spread out" so that sum(h^2) is large relative to max(conv).
    # Actually, for a fixed support, sum(h^2) is maximized when h is concentrated, but then conv is also large.
    # Let's think: if h is uniform on k bins, sum(h^2)=1/k, max(conv)=k*(1/k)^2=1/k. So ratio=1.
    # If h is a delta, sum(h^2)=1, max(conv)=1, ratio=1.
    # So uniform and delta both give 1. We need something in between.
    
    # Let's try: h with two non-zero entries at positions i,j with values a, 1-a.
    # sum(h^2) = a^2+(1-a)^2
    # conv: the autoconvolution of [a at i, 1-a at j] has max = max(a^2, (1-a)^2, a(1-a)) if |i-j|>0
    # Actually conv at lag 0: a^2+(1-a)^2, at lag |i-j|: 2*a*(1-a)
    # So max(conv) = max(a^2+(1-a)^2, 2*a*(1-a))
    # If a=0.5, a^2+(1-a)^2=0.5, 2*a*(1-a)=0.5, ratio=1
    # If a=0.9, a^2+(1-a)^2=0.82, 2*0.9*0.1=0.18, max=0.82, ratio=1
    # Hmm, seems like it's always 1 for 2-point support? Let me check.
    # sum(h^2) = a^2+(1-a)^2, max_conv = max(a^2+(1-a)^2, 2a(1-a))
    # Since a^2+(1-a)^2 = 2a^2-2a+1 and 2a(1-a)=2a-2a^2
    # 2a^2-2a+1 >= 2a-2a^2 iff 4a^2-4a+1>=0 iff (2a-1)^2>=0, always true.
    # So max_conv = a^2+(1-a)^2 = sum(h^2), ratio=1.
    
    # So 2-point support always gives ratio 1. Need more structure.
    
    # Let's try 3 points. Or better, let's just do numerical optimization.
    
    # Phase 1: Try various structured initializations
    for support_size in range(3, min(30, N+1)):
        if time.time() > deadline:
            break
        
        # Uniform
        h0 = np.zeros(N)
        h0[:support_size] = 1.0/support_size
        try_candidate(h0)
        
        # Cosine window
        if support_size >= 3:
            h0 = np.zeros(N)
            x = np.linspace(-1, 1, support_size)
            w = 0.5*(1+np.cos(np.pi*x))
            w = np.clip(w, 1e-10, None)
            h0[:support_size] = w/w.sum()
            try_candidate(h0)
        
        # Gaussian-like
        if support_size >= 5:
            h0 = np.zeros(N)
            x = np.arange(support_size) - support_size//2
            w = np.exp(-x**2/(2*max(1,support_size//4)**2))
            h0[:support_size] = w/w.sum()
            try_candidate(h0)
        
        # Random
        for _ in range(3):
            if time.time() > deadline:
                break
            h0 = np.zeros(N)
            h0[:support_size] = np.random.dirichlet(np.ones(support_size))
            try_candidate(h0)

    # Phase 2: Local optimization using projected gradient or SLSQP on simplex
    # We optimize over h with sum(h)=1, h>=0.
    # Use scipy minimize with SLSQP.
    
    def neg_ratio_simplex(x):
        return -ratio(x)
    
    # Try optimizing from best candidate and several random starts
    starts = []
    if best_h is not None:
        starts.append(best_h.copy())
    
    for support_size in [5, 10, 15, 20, 25, 30]:
        for _ in range(2):
            h0 = np.zeros(N)
            h0[:support_size] = np.random.dirichlet(np.ones(support_size))
            starts.append(h0)
    
    # Also try some structured starts
    for k in [4, 6, 8, 12, 16]:
        if k <= N:
            h0 = np.zeros(N)
            x = np.linspace(-1, 1, k)
            w = 0.5*(1+np.cos(np.pi*x))
            w = np.clip(w, 1e-10, None)
            h0[:k] = w/w.sum()
            starts.append(h0)

    for h0 in starts:
        if time.time() > deadline:
            break
        
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                           'jac': lambda x: np.ones(N)}]
            bounds = [(0, 1)] * N
            
            result = minimize(neg_ratio_simplex, h0, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-10})
            
            if result.success or -result.fun > best_val:
                try_candidate(result.x)
        except Exception:
            pass

    # Phase 3: Coordinate ascent refinement
    # Fix sum=1, then do coordinate-wise updates
    if best_h is not None:
        h = best_h.copy()
        for it in range(500):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_h = h[i]
                # Optimize h[i] with others fixed, maintaining sum=1
                # Let s = 1 - h[i], others fixed. We vary h[i] in [0,1]
                # ratio = (sum(h^2)) / max(conv)
                # This is complicated because conv depends on h[i].
                # Instead, just try a few perturbations.
                for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                    h_new = h.copy()
                    h_new[i] = max(0, h[i] + delta)
                    # Renormalize
                    s = np.sum(h_new)
                    if s > 0:
                        h_new = h_new/s
                        val = ratio(h_new)
                        if val > best_val:
                            best_val = val
                            best_h = h_new.copy()
                            h = h_new.copy()
                            improved = True
                            break
            if not improved:
                break

    # Final: ensure valid
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    else:
        best_h = np.clip(best_h, 0, None)
        s = np.sum(best_h)
        if s > 0:
            best_h = best_h/s
        else:
            best_h = np.zeros(N)
            best_h[0] = 1.0

    return best_h.tolist()
# EVOLVE-BLOCK-END
