import numpy as np
import time

N = 400

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    return np.max(np.abs(ac[1:N]))

def _update_autocorr_flip(ac, h, idx):
    """
    Update autocorrelation array in-place when h[idx] is flipped.
    h[idx] becomes -h[idx].
    ac[k] = sum_{i} h[i] h[i+k]
    Change in ac[k] = -2 * h[idx] * (h[idx-k] + h[idx+k])
    Note: Indices must be valid.
    """
    old_val = h[idx]
    new_val = -old_val
    h[idx] = new_val
    
    # k ranges from 1 to N-1
    # ac[k] depends on pairs (i, i+k)
    # The index idx participates in pairs where:
    # 1. idx is the first element: i = idx, so i+k = idx+k. Valid if idx+k < N.
    # 2. idx is the second element: i+k = idx, so i = idx-k. Valid if idx-k >= 0.
    
    # We need to update ac[k] for all k.
    # For a given k, the term involving idx is:
    # If idx < N-k: h[idx] * h[idx+k] is in ac[k].
    # If idx >= k: h[idx-k] * h[idx] is in ac[k].
    
    # Actually, it's easier to just loop k from 1 to N-1 and update.
    # This is O(N).
    
    for k in range(1, N):
        delta = 0.0
        # Case 1: idx is the lower index in the pair (idx, idx+k)
        if idx + k < N:
            delta += old_val * h[idx + k]
            # New term is new_val * h[idx+k]
            # Change is (new_val - old_val) * h[idx+k] = -2*old_val * h[idx+k]
            # Wait, the formula for change is:
            # ac_new[k] = ac_old[k] - 2 * old_val * (h[idx-k] + h[idx+k]) 
            # Let's verify.
            # ac[k] = sum h[i]h[i+k].
            # If we flip h[idx], terms involving h[idx] change.
            # Term 1: i = idx. h[idx]h[idx+k]. Change: -2 h[idx] h[idx+k].
            # Term 2: i+k = idx => i = idx-k. h[idx-k]h[idx]. Change: -2 h[idx-k] h[idx].
            # So delta_ac[k] = -2 * old_val * (h[idx-k] (if valid) + h[idx+k] (if valid))
            
            # Let's just compute the new contribution directly to be safe?
            # No, the delta formula is correct.
            pass
        
        # Let's use the delta formula directly in the loop.
        # delta = -2 * old_val * ( (h[idx-k] if idx>=k else 0) + (h[idx+k] if idx+k<N else 0) )
        
    # Re-implementing the update loop correctly
    for k in range(1, N):
        term = 0.0
        if idx >= k:
            term += h[idx - k]
        if idx + k < N:
            term += h[idx + k]
        ac[k] += -2.0 * old_val * term
        
    return ac, h

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_ac = _compute_autocorr(best_h)
    best_val = _max_sidelobe(best_ac)
    
    # Phase 1: Greedy Hill Climbing from Seed (Ones)
    # This might get stuck, but it's fast.
    h = best_h.copy()
    ac = best_ac.copy()
    improved = True
    while improved and (time.time() < deadline - 20.0):
        improved = False
        for i in range(N):
            # Try flip
            old_ac, old_h_i = ac.copy(), h[i]
            new_h, new_ac = _update_autocorr_flip(ac.copy(), h.copy(), i)
            new_val = _max_sidelobe(new_ac)
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
                best_ac = ac.copy()
                improved = True
            else:
                # Revert
                # We modified h and ac in place in _update_autocorr_flip? 
                # No, I passed copies in the check, but for efficiency we should do it in place and revert.
                # Let's optimize the greedy loop to be in-place.
                pass
        
        # The above greedy loop with copies is O(N^2) per pass due to copies.
        # Let's do in-place.
        
    # Redo Phase 1 with In-Place Greedy
    h = best_h.copy()
    ac = best_ac.copy()
    current_val = best_val
    
    improved = True
    while improved and (time.time() < deadline - 30.0):
        improved = False
        for i in range(N):
            old_h_i = h[i]
            # Save old ac? No, we can revert ac if we store the delta?
            # Or just copy ac (N=400, small).
            ac_old = ac.copy()
            _update_autocorr_flip(ac, h, i)
            new_val = _max_sidelobe(ac)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                # Revert
                h[i] = old_h_i
                ac[:] = ac_old
                
    if current_val < best_val:
        best_val = current_val
        best_h = h.copy()
        best_ac = ac.copy()
        
    # Phase 2: Random Restarts with Greedy Polish
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline - 20.0:
            break
            
        h_rand = np.random.choice([-1.0, 1.0], size=N)
        ac_rand = _compute_autocorr(h_rand)
        val_rand = _max_sidelobe(ac_rand)
        
        # Greedy Polish from Random
        h_cur = h_rand.copy()
        ac_cur = ac_rand.copy()
        val_cur = val_rand
        
        # Quick Greedy Pass
        improved = True
        while improved and (time.time() < deadline - 10.0):
            improved = False
            for i in range(N):
                old_h_i = h_cur[i]
                ac_old = ac_cur.copy()
                _update_autocorr_flip(ac_cur, h_cur, i)
                new_val = _max_sidelobe(ac_cur)
                if new_val < val_cur:
                    val_cur = new_val
                    improved = True
                else:
                    h_cur[i] = old_h_i
                    ac_cur[:] = ac_old
                    
        if val_cur < best_val:
            best_val = val_cur
            best_h = h_cur.copy()
            best_ac = ac_cur.copy()
            
    # Phase 3: Simulated Annealing from Best
    if time.time() < deadline - 5.0:
        h_sa = best_h.copy()
        ac_sa = best_ac.copy()
        val_sa = best_val
        
        T0 = max(val_sa * 0.1, 1.0)
        cooling = 0.999
        
        it = 0
        while time.time() < deadline - 2.0:
            i = np.random.randint(N)
            old_h_i = h_sa[i]
            ac_old = ac_sa.copy()
            
            _update_autocorr_flip(ac_sa, h_sa, i)
            new_val = _max_sidelobe(ac_sa)
            
            delta = new_val - val_sa
            T = T0 * (cooling ** it)
            
            if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
                val_sa = new_val
                if val_sa < best_val:
                    best_val = val_sa
                    best_h = h_sa.copy()
                    best_ac = ac_sa.copy()
            else:
                h_sa[i] = old_h_i
                ac_sa[:] = ac_old
                
            it += 1
            if it % 1000 == 0:
                T0 *= 0.9 # Extra cooling?

    return best_h
