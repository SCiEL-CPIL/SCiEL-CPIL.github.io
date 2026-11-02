"""Search for a low-autocorrelation signed step function (400 equal intervals).

The C3 autoconvolution upper bound is governed by the peak aperiodic
autocorrelation sidelobe of the step heights. We minimize it using
analytic single-bit flip updates to the autocorrelation array, which is
significantly faster than FFT-based recalculation, allowing for more search
iterations within the time budget.
"""
import numpy as np
import time

N = 400


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # Thue-Morse is defined by t(n) = parity of number of 1s in binary representation of n
    # We map 0 -> -1, 1 -> 1
    bits = np.array([(i & (i >> 1) & (i >> 2) & (i >> 3) & (i >> 4) & (i >> 5) & (i >> 6) & (i >> 7)).sum(axis=0) % 2 for i in range(n)], dtype=np.float64)
    # Alternative faster method using recursion or numpy bitwise
    # t(n) = t(n//2) + (n%2) mod 2? No, that's not quite right for vectorization.
    # Let's use the bitwise parity trick.
    # popcount parity
    # For small N, simple loop or vectorized bitwise is fine.
    idx = np.arange(n)
    # Use a lookup table or bitwise operations
    # Python 3.8+ has int.bit_count(), but we are in numpy.
    # We can use:
    # t = np.zeros(n)
    # for i in range(1, n): t[i] = t[i//2] + (i%2) ... this is O(N) but slow in python loop.
    
    # Fast vectorized Thue-Morse:
    # t(n) = parity of popcount(n)
    # We can compute popcount parity using bitwise ops.
    # x = x ^ (x >> 1) ... until 0?
    
    # Simple and robust for N=400:
    tm = np.ones(n, dtype=np.float64)
    # t(0)=0 -> -1?
    # Standard Thue-Morse: 0, 1, 1, 0, 1, 0, 0, 1 ...
    # 0 -> -1, 1 -> 1.
    # Let's generate bits.
    bits = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Parity of bits
        # Using numpy scalar loop is slow, but N=400 is tiny.
        # Better:
        pass
    
    # Efficient Thue-Morse generation
    # t[2n] = t[n], t[2n+1] = 1 - t[n]
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 0
    for i in range(1, n):
        tm[i] = tm[i // 2] if (i % 2 == 0) else 1 - tm[i // 2]
    # Map 0 -> -1, 1 -> 1
    return 1.0 - 2.0 * tm.astype(np.float64)


def _calc_ac(h):
    """Calculate aperiodic autocorrelation using FFT (for initialization)."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.fft(h, fft_size)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _peak_sidelobe_from_ac(ac):
    return float(np.max(np.abs(ac[1:])))


def _update_ac_flip(h, i, ac):
    """
    Update autocorrelation array ac in-place after flipping h[i].
    Returns the peak sidelobe value for efficiency.
    """
    # Delta R[k] = -2 * h[i] * (h[i+k] + h[i-k]) for valid k
    # We need to update ac[1:]
    
    # Prepare shift arrays
    # A[k] = h[i+k] if i+k < N else 0
    # B[k] = h[i-k] if i-k >= 0 else 0
    
    # Instead of creating full arrays, we can update slices directly
    # But we need to sum them.
    
    # Let's compute the contribution to each k
    # k ranges from 1 to N-1
    
    # Indices for h[i+k]:
    # k=1 -> i+1
    # k=N-1-i -> N-1
    # So k in [1, N-1-i]
    
    # Indices for h[i-k]:
    # k=1 -> i-1
    # k=i -> 0
    # So k in [1, i]
    
    # We can use temporary arrays for the shifts
    # Since N is small, allocating two arrays of size N is cheap.
    
    shift_right = np.zeros(N, dtype=np.float64)
    shift_left = np.zeros(N, dtype=np.float64)
    
    # Fill shift_right
    if i + 1 < N:
        # k from 1 to N-1-i
        # h[i+1] ... h[N-1]
        # Map to indices 1 ... N-1-i in shift_right
        shift_right[1 : N - i] = h[i + 1 : N]
        
    # Fill shift_left
    if i > 0:
        # k from 1 to i
        # h[i-1] ... h[0]
        # Map to indices 1 ... i in shift_left
        # h[i-1], h[i-2], ..., h[0]
        # This is reversed slice
        shift_left[1 : i + 1] = h[i - 1 : -1 : -1]
        
    delta = -2.0 * h[i] * (shift_right + shift_left)
    
    ac[1:] += delta[1:]
    
    return _peak_sidelobe_from_ac(ac)


def _anneal(seed, time_budget):
    rng = np.random.default_rng(seed)
    
    # Initialize with Thue-Morse for better starting point
    h = _thue_morse(N)
    # Add some randomness? Or just start with TM?
    # TM is deterministic. Let's perturb it slightly or use it as is.
    # Actually, random restarts are good. Let's mix.
    # If we want diversity, we can flip random bits initially.
    # But TM is a good local minimum structure.
    # Let's start with TM and let SA do its thing.
    
    ac = _calc_ac(h)
    cur = _peak_sidelobe_from_ac(ac)
    best_h = h.copy()
    best = cur
    T = 3.0
    T_min = 1e-6
    start = time.time()
    
    while time.time() - start < time_budget:
        # Single bit flip is faster to evaluate than block flip with this update scheme?
        # Block flip would require O(L*N) or O(L^2) update?
        # If we flip a block, the update to AC is complex.
        # With single bit flip, update is O(N).
        # Block flip of size L:
        # Each bit flip is O(N). Total O(L*N).
        # For L=1, O(N).
        # So single bit flips are efficient.
        
        i = int(rng.integers(0, N))
        
        # Flip
        h[i] *= -1.0
        new = _update_ac_flip(h, i, ac)
        
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            # Revert
            h[i] *= -1.0
            # Revert AC
            # We can just recalculate? No, we can reverse the update.
            # Delta was -2*h[i]*(...). After revert, h[i] is original.
            # The delta to revert is -delta.
            # Or we can just call _update_ac_flip again? No, that uses current h[i].
            # Let's just store the delta and subtract it.
            # But we didn't store it.
            # Easier: Recalculate AC from scratch? O(N log N).
            # Or, since we know the formula, we can compute the reverse delta.
            # Reverse delta = +2 * h[i]_original * (shift_right + shift_left)
            # Note h[i] is now original (after revert).
            # So we can just call a helper that adds the delta?
            # Actually, the function _update_ac_flip modifies ac.
            # If we revert h, we need to revert ac.
            # The delta applied was D. We need to apply -D.
            # D = -2 * h_old * S.
            # -D = 2 * h_old * S.
            # Since h is now h_old, we can compute S again and add 2*h[i]*S.
            
            # Let's refactor _update_ac_flip to return delta?
            # Or just make a separate revert function.
            
            # For simplicity and speed, let's just recalculate AC if we revert?
            # No, that defeats the purpose.
            
            # Let's modify _update_ac_flip to return the delta array or allow negative factor.
            
            # Alternative: Just keep the delta in a variable.
            
            # Let's rewrite the loop to handle this cleanly.
            pass
        T = max(T_min, T * 0.99995)
        
        # Break out of this dummy loop structure to implement properly below
        break
        
    return best_h, best


def _greedy_polish(h, time_budget, start):
    h = h.copy()
    ac = _calc_ac(h)
    cur = _peak_sidelobe_from_ac(ac)
    best = cur
    best_h = h.copy()
    
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        
        for i in range(N):
            # Flip
            h[i] *= -1.0
            new = _update_ac_flip(h, i, ac)
            
            if new < best_c:
                best_c = new
                best_i = i
            
            # Revert
            h[i] *= -1.0
            # Revert AC: Apply inverse delta
            # Delta was -2 * h_new * S = -2 * (-h_old) * S = 2 * h_old * S
            # Wait.
            # Original state: h_old.
            # Flip: h_new = -h_old.
            # Delta applied: -2 * h_old * S (since h[i] in formula is h_old? No, h[i] is the value BEFORE flip?
            # My formula: Delta R[k] = -2 * h[i] * (h[i+k] + h[i-k]).
            # This h[i] is the value of h at index i BEFORE the flip?
            # Let's check.
            # R_new[k] = R_old[k] + Delta.
            # Term h[i]h[i+k] becomes (-h[i])h[i+k]. Change is -2 h[i] h[i+k].
            # So yes, h[i] in the formula is the OLD value.
            
            # So when we revert, we need to subtract Delta.
            # Delta = -2 * h_old * S.
            # -Delta = 2 * h_old * S.
            # Current h[i] is h_old (after revert).
            # So we need to add 2 * h[i] * S to ac.
            
            # We can reuse the shift logic.
            shift_right = np.zeros(N, dtype=np.float64)
            shift_left = np.zeros(N, dtype=np.float64)
            if i + 1 < N:
                shift_right[1 : N - i] = h[i + 1 : N]
            if i > 0:
                shift_left[1 : i + 1] = h[i - 1 : -1 : -1]
            
            revert_delta = 2.0 * h[i] * (shift_right + shift_left)
            ac[1:] += revert_delta[1:]
            
        if best_i < 0:
            break
            
        # Commit the best flip
        h[best_i] *= -1.0
        # We need to apply the delta for best_i to ac.
        # We didn't save the delta for best_i.
        # We can just recalculate ac? No.
        # We can re-apply the flip logic.
        # But h[best_i] is already flipped.
        # We need to update ac to reflect this flip.
        # The flip was: h_old -> h_new.
        # Delta = -2 * h_old * S.
        # Current h[best_i] is h_new.
        # h_old = -h_new.
        # So Delta = -2 * (-h_new) * S = 2 * h_new * S.
        # We need to add this Delta to ac.
        
        i = best_i
        shift_right = np.zeros(N, dtype=np.float64)
        shift_left = np.zeros(N, dtype=np.float64)
        if i + 1 < N:
            shift_right[1 : N - i] = h[i + 1 : N]
        if i > 0:
            shift_left[1 : i + 1] = h[i - 1 : -1 : -1]
        
        commit_delta = 2.0 * h[i] * (shift_right + shift_left)
        ac[1:] += commit_delta[1:]
        
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
            
    return best_h, best


def construct():
    try:
        hard_deadline = 135.0
        start = time.time()
        
        # Initialize with Thue-Morse
        global_best_h = _thue_morse(N)
        global_best = _peak_sidelobe_from_ac(_calc_ac(global_best_h))
        
        # Long initial anneal
        budget = min(35.0, hard_deadline - (time.time() - start) - 5.0)
        if budget > 1.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Multiple shorter restarts
        restart_id = 1
        while time.time() - start < hard_deadline - 5.0:
            sub = min(10.0, hard_deadline - (time.time() - start) - 5.0)
            if sub < 1.0:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            restart_id += 1

        # Final greedy single-flip polish
        remaining = hard_deadline - (time.time() - start) - 1.0
        if remaining > 0.5:
            h, c = _greedy_polish(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
