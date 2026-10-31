import numpy as np
import time

N = 100

def _max_overlap(h):
    """Compute the maximum overlap S_k = sum_{i=0}^{N-1-k} h[i]*(1-h[i+k]) for all k."""
    max_val = 0.0
    for k in range(N):
        # Vectorized: h[i] * (1 - h[i+k]) for i = 0..N-1-k
        prod = h[:N-k] * (1.0 - h[k:])
        s_k = np.sum(prod)
        if s_k > max_val:
            max_val = s_k
    return max_val

def _project_to_valid(h):
    """Project h to satisfy: sum(h) == N/2, 0 <= h_i <= 1."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    # We need sum(h) = N/2. Current sum may differ.
    # Add a constant offset, but must keep in [0,1].
    # If we add delta to all, new sum = old_sum + N*delta.
    # delta = (N/2 - old_sum) / N
    # But this might push some values out of [0,1].
    # Iterative water-filling approach:
    target = N / 2.0
    for _ in range(50):
        s = np.sum(h)
        if abs(s - target) < 1e-12:
            break
        delta = (target - s) / N
        h = h + delta
        # Clip and re-normalize
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment: distribute the remaining error across free indices
    s = np.sum(h)
    diff = target - s
    if abs(diff) > 1e-12:
        # Find indices that can absorb the change
        if diff > 0:
            # Need to increase sum; find indices < 1
            free = np.where(h < 1.0 - 1e-12)[0]
        else:
            # Need to decrease sum; find indices > 0
            free = np.where(h > 1e-12)[0]
        if len(free) > 0:
            per = diff / len(free)
            h[free] += per
            h = np.clip(h, 0.0, 1.0)
    return h

def _score(h):
    """Return the raw overlap value (lower is better)."""
    return _max_overlap(h)

def _construct_from_cosine(A, phi):
    """Construct h_i = 0.5 + A * cos(2*pi*i/N + phi)."""
    i = np.arange(N)
    h = 0.5 + A * np.cos(2.0 * np.pi * i / N + phi)
    return h

def _construct_from_sine(A, phi):
    """Construct h_i = 0.5 + A * sin(2*pi*i/N + phi)."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    return h

def _construct_from_trend(A, B):
    """h_i = 0.5 + A*(i/N - 0.5) + B*(i/N - 0.5)^2, ensuring sum = N/2."""
    x = np.arange(N) / N - 0.5
    h = 0.5 + A * x + B * x * x
    # The linear term sums to ~0, the quadratic term sums to ~N/12 * B
    # We'll project to valid
    return h

def _construct_from_two_cos(A1, phi1, A2, phi2):
    """h_i = 0.5 + A1*cos(2*pi*i/N + phi1) + A2*cos(2*pi*i/N + phi2)."""
    i = np.arange(N)
    h = 0.5 + A1 * np.cos(2.0 * np.pi * i / N + phi1) + A2 * np.cos(2.0 * np.pi * i / N + phi2)
    return h

def _construct_from_three_cos(A1, phi1, A2, phi2, A3, phi3):
    """h_i = 0.5 + A1*cos(2*pi*i/N + phi1) + A2*cos(2*pi*i/N + phi2) + A3*cos(2*pi*i/N + phi3)."""
    i = np.arange(N)
    h = (0.5 + A1 * np.cos(2.0 * np.pi * i / N + phi1) + 
                A2 * np.cos(2.0 * np.pi * i / N + phi2) + 
                A3 * np.cos(2.0 * np.pi * i / N + phi3))
    return h

def _construct_from_fourier(coeffs):
    """
    coeffs: array of shape (K, 2) where coeffs[k] = (A_k, phi_k) for k=1..K
    h_i = 0.5 + sum_k A_k * cos(2*pi*k*i/N + phi_k)
    """
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k_idx, (A, phi) in enumerate(coeffs, start=1):
        h += A * np.cos(2.0 * np.pi * k_idx * i / N + phi)
    return h

def _local_search(h0, max_iters=200, step=0.01, deadline=None):
    """
    Simple coordinate-descent-like local search.
    At each step, try small perturbations to each coordinate.
    """
    h = h0.copy()
    best_val = _score(h)
    i = 0
    while i < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for j in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Try increasing and decreasing h[j]
            for delta in [step, -step]:
                h_new = h.copy()
                h_new[j] += delta
                h_new = _project_to_valid(h_new)
                val = _score(h_new)
                if val < best_val - 1e-12:
                    h = h_new
                    best_val = val
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
        i += 1
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _score(best_h)
    
    # Phase 1: Grid search over low-dimensional parameter families
    
    # Family 1: Single cosine
    for A in np.linspace(0, 0.5, 11):
        for phi in np.linspace(0, 2*np.pi, 10):
            if time.time() > deadline:
                return best_h
            h = _construct_from_cosine(A, phi)
            h = _project_to_valid(h)
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Family 2: Single sine
    for A in np.linspace(0, 0.5, 11):
        for phi in np.linspace(0, 2*np.pi, 10):
            if time.time() > deadline:
                return best_h
            h = _construct_from_sine(A, phi)
            h = _project_to_valid(h)
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Family 3: Linear trend
    for A in np.linspace(-0.5, 0.5, 11):
        for B in np.linspace(-0.5, 0.5, 11):
            if time.time() > deadline:
                return best_h
            h = _construct_from_trend(A, B)
            h = _project_to_valid(h)
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Family 4: Two cosines
    for A1 in [0.0, 0.25, 0.5]:
        for A2 in [0.0, 0.25, 0.5]:
            for phi1 in [0, np.pi/2, np.pi, 3*np.pi/2]:
                for phi2 in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    if time.time() > deadline:
                        return best_h
                    h = _construct_from_two_cos(A1, phi1, A2, phi2)
                    h = _project_to_valid(h)
                    val = _score(h)
                    if val < best_val:
                        best_val = val
                        best_h = h.copy()
    
    # Family 5: Three cosines (coarse)
    for A1 in [0.0, 0.3, 0.5]:
        for A2 in [0.0, 0.2, 0.4]:
            for A3 in [0.0, 0.1, 0.3]:
                for phi1 in [0, np.pi/2]:
                    for phi2 in [0, np.pi/2]:
                        for phi3 in [0, np.pi/2]:
                            if time.time() > deadline:
                                return best_h
                            h = _construct_from_three_cos(A1, phi1, A2, phi2, A3, phi3)
                            h = _project_to_valid(h)
                            val = _score(h)
                            if val < best_val:
                                best_val = val
                                best_h = h.copy()
    
    # Phase 2: Local search from best found
    if time.time() < deadline - 10:
        h_ls, val_ls = _local_search(best_h, max_iters=300, step=0.02, deadline=deadline)
        if val_ls < best_val:
            best_h = h_ls
            best_val = val_ls
    
    # Phase 3: Try a few random Fourier-based initializations and local search
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > deadline:
                break
            # Random 3-term Fourier
            coeffs = []
            for k in range(1, 4):
                A = rng.uniform(0, 0.4)
                phi = rng.uniform(0, 2*np.pi)
                coeffs.append((A, phi))
            h = _construct_from_fourier(coeffs)
            h = _project_to_valid(h)
            val = _score(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            # Local search from here
            h_ls, val_ls = _local_search(h, max_iters=100, step=0.02, deadline=deadline)
            if val_ls < best_val:
                best_val = val_ls
                best_h = h_ls.copy()
    
    # Final validation
    best_h = _project_to_valid(best_h)
    return best_h
