"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _incremental_ac_update(ac, h, i, delta):
    """
    Update linear autoconvolution ac in-place when h[i] changes by delta.
    ac[k] = sum_j h[j] h[k-j].
    Change in ac[k] = delta * (sum_{j != i} h[j] h[k-j] + h[i] h[k-i] ... )
    Actually, ac_new[k] = ac_old[k] + delta * (sum_j h[j] h[k-j] - 2*h[i]*h[k-i] + ... )
    Let's derive:
    ac[k] = sum_j h[j] h[k-j]
    If h[i] -> h[i] + delta:
    ac_new[k] = sum_{j != i} h[j] h[k-j] + (h[i]+delta) h[k-i]
               = ac[k] - h[i] h[k-i] + (h[i]+delta) h[k-i]
               = ac[k] + delta * h[k-i]
    Wait, this is only if k-i is a valid index in h.
    Also, if j = i is in the sum, we have h[i] h[k-i].
    If k-i is out of bounds [0, N-1], that term is 0.
    So ac_new[k] = ac[k] + delta * h[k-i] (if 0 <= k-i < N)
    
    BUT, we also have the symmetric term? No, autoconvolution is sum_j h[j]h[k-j].
    The term where j=i is h[i]h[k-i].
    The term where k-j=i => j=k-i is h[k-i]h[i].
    If i != k-i, these are two distinct terms in the sum (unless j=k-i=i, i.e., k=2i).
    So if k != 2i, the change is delta*h[k-i] + delta*h[i] (from the j=k-i term).
    If k == 2i, the term is h[i]^2, which becomes (h[i]+delta)^2 = h[i]^2 + 2delta h[i] + delta^2.
    Change is 2delta h[i] + delta^2.
    
    So:
    If k == 2i: ac[k] += 2*delta*h[i] + delta**2
    Else:
      if 0 <= k-i < N: ac[k] += delta * h[k-i]
      if 0 <= i-k < N: ac[k] += delta * h[i-k]  # This is the j=k-i term, which is h[k-i]h[i]. Wait.
      
    Let's re-evaluate.
    ac[k] = sum_{j=0}^{N-1} h[j] h[k-j]
    Term 1: j=i. Contribution: h[i] h[k-i]. (Valid if 0<=k-i<N)
    Term 2: j=k-i. Contribution: h[k-i] h[i]. (Valid if 0<=k-i<N)
    
    If i != k-i (i.e., k != 2i):
      The sum contains h[i]h[k-i] AND h[k-i]h[i].
      Both change.
      New h[i] = h[i]+delta.
      New h[k-i] = h[k-i] (unchanged).
      So Term 1 becomes (h[i]+delta)h[k-i] = h[i]h[k-i] + delta h[k-i].
      Term 2 becomes h[k-i](h[i]+delta) = h[k-i]h[i] + delta h[k-i].
      Total change: 2 * delta * h[k-i].
      
    If i == k-i (k=2i):
      The sum contains h[i]^2 only once (since j=i is the only index where k-j=i).
      New term: (h[i]+delta)^2 = h[i]^2 + 2delta h[i] + delta^2.
      Change: 2delta h[i] + delta^2.
      
    So the update rule is:
    if 0 <= k-i < N:
        if k == 2i:
            ac[k] += 2*delta*h[i] + delta**2
        else:
            ac[k] += 2*delta*h[k-i]
    # Note: if k-i is out of bounds, the term is 0 in both old and new, so no change.
    
    This allows O(1) update per k? No, we need to update all k where the term is non-zero.
    For a fixed i, which k are affected?
    k-i must be in [0, N-1] => k in [i, i+N-1].
    So we update ac[k] for k in [i, i+N-1].
    That's N updates per flip.
    Since ac has length 2N-1, and we do this for each flip, it's O(N) per flip.
    This is much faster than O(N log N) FFT for small N?
    For N=400, FFT is 2N-1=799 -> next power of 2 is 1024.
    FFT cost ~ 1024 * log2(1024) = 1024*10 = 10240 ops.
    Incremental update cost ~ 400 ops.
    So incremental is ~25x faster!
    """
    n = len(h)
    # k ranges from i to i+n-1
    start = i
    end = i + n
    # Clamp to ac bounds [0, 2n-2]
    start = max(0, start)
    end = min(2*n-1, end)
    
    if start >= end:
        return
    
    # Vectorized update
    # For k in [start, end-1], k-i is in [0, n-1]
    idx = np.arange(start, end)
    h_idx = idx - i  # This is k-i
    
    # Check if k == 2i
    mask_diag = (idx == 2*i)
    
    # General case: 2 * delta * h[k-i]
    # But if k-i is out of bounds? No, by construction k-i is in [0, n-1] for k in [i, i+n-1].
    # So h_idx is always valid.
    
    updates = 2.0 * delta * h[h_idx]
    
    # Diagonal case: k=2i
    # The formula 2*delta*h[k-i] gives 2*delta*h[i].
    # But the true change is 2*delta*h[i] + delta^2.
    # So we need to add delta^2 to the diagonal element.
    if np.any(mask_diag):
        diag_pos = np.where(mask_diag)[0][0]
        updates[diag_pos] += delta**2
        
    ac[start:end] += updates

def _local_search_fast(h, ac, time_budget):
    """
    Fast local search using incremental autoconvolution updates.
    h: current heights
    ac: current autoconvolution (length 2N-1)
    Returns best_h, best_score, best_ac
    """
    n = len(h)
    best_h = h.copy()
    best_ac = ac.copy()
    best_score = _score(best_h)
    t_start = time.time()
    
    # Precompute indices for random access or just iterate
    # We'll do a few passes of greedy flips.
    
    for _ in range(10): # Max passes
        if time.time() > time_budget:
            break
        improved = False
        # Randomize order to escape local minima
        order = np.random.permutation(n)
        for i in order:
            if time.time() > time_budget:
                break
            # Try flipping h[i]
            old_h_i = h[i]
            new_h_i = -old_h_i
            delta = new_h_i - old_h_i
            
            # Save old ac state for rollback (or just recompute if we don't like it)
            # To avoid copying ac (O(N)), we can apply update, check score, and revert if needed.
            # Reverting is just applying -delta.
            
            _incremental_ac_update(ac, h, i, delta)
            h[i] = new_h_i
            
            # Compute new score
            # max|ac|
            max_abs = np.max(np.abs(ac))
            s = np.sum(h)
            if abs(s) < 1e-9:
                new_score = 1e10
            else:
                new_score = max_abs / (s * s)
                
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
                best_ac = ac.copy()
                improved = True
            else:
                # Revert
                _incremental_ac_update(ac, h, i, -delta)
                h[i] = old_h_i
        
        if not improved:
            break
            
    return best_h, best_score, best_ac

def _anneal_search(h, ac, time_budget, T_start=1.0, T_end=0.01):
    """
    Simulated annealing in the space of sign flips.
    """
    n = len(h)
    cur_h = h.copy()
    cur_ac = ac.copy()
    cur_score = _score(cur_h)
    
    best_h = cur_h.copy()
    best_ac = cur_ac.copy()
    best_score = cur_score
    
    t_start = time.time()
    total_time = time_budget
    steps = 0
    
    # We don't know the number of steps in advance, so we'll just loop until time runs out
    # and interpolate temperature based on elapsed time.
    
    while time.time() < t_start + time_budget:
        steps += 1
        # Linear cooling
        progress = (time.time() - t_start) / time_budget
        T = T_start * (T_end / T_start) ** progress
        
        i = np.random.randint(n)
        old_h_i = cur_h[i]
        new_h_i = -old_h_i
        delta = new_h_i - old_h_i
        
        _incremental_ac_update(cur_ac, cur_h, i, delta)
        cur_h[i] = new_h_i
        
        max_abs = np.max(np.abs(cur_ac))
        s = np.sum(cur_h)
        if abs(s) < 1e-9:
            new_score = 1e10
        else:
            new_score = max_abs / (s * s)
            
        delta_score = new_score - cur_score
        
        if delta_score < 0 or np.random.random() < np.exp(-delta_score / (T + 1e-9)):
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = cur_h.copy()
                best_ac = cur_ac.copy()
        else:
            # Revert
            _incremental_ac_update(cur_ac, cur_h, i, -delta)
            cur_h[i] = old_h_i
            
    return best_h, best_score, best_ac

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    best_ac = None
    
    # Initialize ac for best_h
    def _compute_ac(h):
        n = len(h)
        size = 2 * n - 1
        fsize = 1
        while fsize < size:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)
        return ac[:size]
        
    best_ac = _compute_ac(best_h)
    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()
        best_ac = best_ac.copy()
        
    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Random signs
    rng = np.random.default_rng(42)
    for i in range(5):
        inits.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Chirp
    x = np.arange(N)
    inits.append(np.cos(np.pi * x**2 / N))
    
    # 4. Barker-like
    inits.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 5. Random smooth
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.hanning(11)
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # Optimization loop
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        # Normalize to sum 1? No, the score is scale invariant.
        # But the search works on signs/magnitudes.
        # Let's just use the raw init.
        
        # Compute initial AC
        ac_init = _compute_ac(h_init)
        
        # 1. Greedy local search
        h_opt, val_opt, ac_opt = _local_search_fast(h_init, ac_init, deadline - 20)
        if val_opt < best_score:
            best_score = val_opt
            best_h = h_opt.copy()
            best_ac = ac_opt.copy()
            
        # 2. Simulated annealing from the result of greedy search
        if time.time() < deadline - 10:
            h_sa, val_sa, ac_sa = _anneal_search(h_opt, ac_opt, deadline - 10, T_start=0.5, T_end=0.01)
            if val_sa < best_score:
                best_score = val_sa
                best_h = h_sa.copy()
                best_ac = ac_sa.copy()
                
    # Final polish with greedy search on the best solution found
    if time.time() < deadline - 2:
        h_final, val_final, ac_final = _local_search_fast(best_h, best_ac, deadline - 1)
        if val_final < best_score:
            best_score = val_final
            best_h = h_final.copy()
            best_ac = ac_final.copy()
            
    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N
        best_ac = _compute_ac(best_h)

    return best_h

# EVOLVE-BLOCK-END
