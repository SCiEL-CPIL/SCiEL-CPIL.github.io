"""Rewritten program: fast incremental local search on structured initializations."""

import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s limit

    def _autoconv_max_abs(f):
        """Compute max absolute value of linear autoconvolution using FFT."""
        N = len(f)
        size = 1
        while size < 2 * N - 1:
            size *= 2
        F = np.fft.rfft(f, n=size)
        C = np.fft.irfft(F * F, n=size)
        return np.max(np.abs(C[:2 * N - 1]))

    def _score(f):
        S = np.sum(f)
        if abs(S) < 1e-12:
            return 1e10
        return _autoconv_max_abs(f) / (S * S)

    def _legendre_sequence(p):
        """Generate ±1 sequence from Legendre symbols of length p-1."""
        seq = np.zeros(p - 1, dtype=np.float64)
        for n in range(1, p):
            ls = pow(n, (p - 1) // 2, p)
            seq[n - 1] = 1.0 if ls == 1 else -1.0
        return seq

    def _barker_sequences():
        barker = {
            2: [1, 1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, -1, 1, 1],
            7: [1, 1, -1, -1, 1, -1, 1],
            11: [1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1],
            13: [1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        }
        return [np.array(v, dtype=np.float64) for v in barker.values()]

    def _m_sequence(n):
        """Generate m-sequence of length 2^n - 1 using primitive polynomial."""
        # Primitive polynomials for n=1..8
        polys = {
            1: [0],  # x+1
            2: [0, 0],  # x^2+x+1
            3: [0, 1, 0],  # x^3+x+1
            4: [0, 0, 1, 0],  # x^4+x+1
            5: [0, 0, 0, 1, 0],  # x^5+x^2+1
            6: [0, 0, 1, 1, 0, 0],  # x^6+x^4+x^3+x^2+1
            7: [0, 0, 0, 1, 0, 0, 0],  # x^7+x^4+1
            8: [0, 0, 0, 0, 1, 0, 0, 0],  # x^8+x^4+x^3+x^2+1
        }
        if n not in polys:
            return None
        L = 2**n - 1
        poly = polys[n]  # poly[i] is coefficient of x^i
        # State: n bits, index 0 = x^0, index n-1 = x^(n-1)
        state = [0] * n
        state[0] = 1  # Initial state with one 1
        seq = np.zeros(L, dtype=np.float64)
        for t in range(L):
            seq[t] = 1.0 if state[0] else -1.0
            # Feedback: XOR of bits where poly has 1
            fb = 0
            for i in range(n):
                if poly[i]:
                    fb ^= state[i]
            # Shift right (towards higher powers)
            state = [fb] + state[:n-1]
        return seq

    def _pad_to_length(seq, target_len):
        """Pad or truncate a sequence to target length."""
        if len(seq) >= target_len:
            return seq[:target_len].astype(np.float64)
        reps = (target_len + len(seq) - 1) // len(seq)
        padded = np.tile(seq, reps)
        return padded[:target_len].astype(np.float64)

    def _fast_local_search(f, max_iters=50000, deadline=DEADLINE):
        """
        Fast local search using O(1) incremental autoconvolution updates.
        
        Key insight: when we flip f[i] (delta = -2*f[i]), the change in C[k] is:
        - For k=0: delta_C[0] = 2*f[i]*delta + delta^2 = 2*f[i]*(-2*f[i]) + 4*f[i]^2 = -4+4 = 0
          Wait, f[i] is ±1, so f[i]^2=1. delta = -2*f[i], delta^2=4.
          delta_C[0] = 2*f[i]*(-2*f[i]) + 4 = -4+4 = 0. So C[0] doesn't change!
        
        - For k>0: C[k] = sum_j f[j]*f[j+k]
          f[i] appears as f[i]*f[i+k] (j=i) and f[i-k]*f[i] (j=i-k)
          delta_C[k] = delta*f[i+k] + delta*f[i-k] (where valid)
        
        We can precompute prefix sums of f to get f[i+k] and f[i-k] in O(1),
        but actually we just need individual values, which are O(1) from the array.
        
        The challenge is tracking max|C[k]|. We'll maintain a sorted structure or
        just recompute the max from the changed C values.
        
        Actually, for efficiency, let's track C as an array and only update the
        affected entries. The number of affected C[k] values is O(N) per flip,
        but we can compute delta_C[k] in O(1) for each k.
        
        Better approach: precompute for each position i, the list of k values
        affected and their delta_C values. But that's still O(N) per flip.
        
        Alternative: Use the fact that for ±1 sequences, C[k] is an integer
        between -(N-|k|) and (N-|k|). We can maintain a count array for each
        possible C[k] value and track the max.
        
        Let's use a simpler but effective approach:
        - Maintain C array
        - For each flip, compute new C values for all affected k
        - Track max|C| using a heap or just recompute from scratch periodically
        
        For N up to ~512, O(N) per flip with 50000 flips = 25M operations, feasible.
        """
        N = len(f)
        f = f.copy()
        S = float(np.sum(f))
        
        if abs(S) < 1e-12:
            return f, 1e10
        
        # Compute initial autoconvolution
        size = 1
        while size < 2 * N - 1:
            size *= 2
        F = np.fft.rfft(f, n=size)
        C = np.fft.irfft(F * F, n=size)
        C = C[:2 * N - 1].copy()
        
        cur_max = float(np.max(np.abs(C)))
        cur_score = cur_max / (S * S)
        
        best_f = f.copy()
        best_score = cur_score
        
        # Precompute: for each i, which k values are affected
        # k ranges from 0 to 2N-2
        # C[k] involves f[j]*f[j+k] for j=0..N-1-k
        # f[i] affects C[k] if i is in [0, N-1-k] (as j) or i-k is in [0, N-1-k] (as j+k=i)
        # i.e., k <= N-1-i (i as j) or k >= i (i-k as j, so i-k >= 0 => k <= i, and i-k <= N-1-k => i <= N-1)
        
        # For flip at position i with delta = -2*f[i]:
        # C_new[k] = C[k] + delta * (f[i+k] if 0<=i+k<N else 0) + delta * (f[i-k] if 0<=i-k<N else 0)
        # But for k=0: both terms are f[i], so it's delta*2*f[i], but actually:
        # C[0] = sum f[j]^2 = N (always, for ±1). So C[0] never changes!
        # Let's verify: delta_C[0] = delta*f[i] + delta*f[i] = 2*delta*f[i] = 2*(-2*f[i])*f[i] = -4
        # But C[0] should be N always... Let me recheck.
        # C[0] = sum_j f[j]*f[j] = sum_j f[j]^2 = N. Yes, constant.
        # So the formula must be wrong for k=0.
        
        # The issue: C[k] = sum_{j=0}^{N-1-k} f[j]*f[j+k]
        # For k=0: C[0] = sum_{j=0}^{N-1} f[j]^2 = N
        # When f[i] changes by delta:
        #   Term j=i in C[0]: f[i]^2 -> (f[i]+delta)^2 = f[i]^2 + 2*f[i]*delta + delta^2
        #   So delta_C[0] = 2*f[i]*delta + delta^2
        # With delta = -2*f[i]: delta_C[0] = 2*f[i]*(-2*f[i]) + 4*f[i]^2 = -4+4 = 0. Correct!
        
        # For k != 0:
        #   f[i] appears in C[k] as:
        #     j=i: f[i]*f[i+k] (valid if i+k < N, i.e., k < N-i)
        #     j=i-k: f[i-k]*f[i] (valid if i-k >= 0, i.e., k <= i)
        #   delta_C[k] = delta*f[i+k] (if k < N-i) + delta*f[i-k] (if k <= i)
        
        no_improve = 0
        max_no_improve = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try positions in a shuffled order to escape local minima
            order = np.random.permutation(N)
            
            for i in order:
                if time.time() > deadline:
                    break
                
                v = f[i]
                delta = -2.0 * v  # new value is -v, so delta = -v - v = -2v
                
                new_S = S + delta
                if abs(new_S) < 1e-12:
                    continue
                
                # Compute delta_C for all affected k
                # k ranges from 0 to 2N-2
                # Affected k: those where i+k < N (k < N-i) or i-k >= 0 (k <= i)
                # So k in [0, max(N-1-i, i)]
                
                # For efficiency, compute the change in max|C|
                # We'll track the current max and see if it decreases
                
                # Compute new C values for affected k
                # Number of affected k: min(N, max(N-i, i+1)) roughly
                k_min = 0
                k_max = min(2*N-2, max(N-1-i, i))
                
                # Track the maximum |C[k]| after the change
                # We need to know the current max and whether it's at an affected position
                
                # Simple approach: compute new max from all C values
                # But that's O(N) per flip. For N=256, that's fine.
                
                # Let's just recompute max from the modified C
                # Store old C values for affected k
                old_C_affected = {}
                new_max = cur_max
                max_changed = False
                
                for k in range(k_min, k_max + 1):
                    contrib = 0.0
                    if k < N - i:  # i+k < N
                        contrib += f[i + k]
                    if k <= i:  # i-k >= 0
                        contrib += f[i - k]
                    
                    if k == 0:
                        delta_C = 2.0 * v * delta + delta * delta  # = 0
                    else:
                        delta_C = delta * contrib
                    
                    old_val = C[k]
                    new_val = old_val + delta_C
                    old_C_affected[k] = old_val
                    C[k] = new_val
                    
                    if abs(new_val) > new_max:
                        new_max = abs(new_val)
                        max_changed = True
                
                # If the old max was at an unaffected position, new_max = cur_max
                # If at an affected position, we need to check
                # For simplicity, just use new_max as computed above
                # But we only computed for affected k. If the global max was at an unaffected k,
                # it's still cur_max. So new_max = max(cur_max, max over affected k of |C[k]|)
                # Wait, we replaced C[k] for affected k, so the unaffected k still have their old values.
                # The new global max is max(max over unaffected k, max over affected k)
                # = max(cur_max if old max was unaffected, new_max_from_affected)
                
                # Actually, cur_max is the max over ALL k before the change.
                # After the change, unaffected k have the same values, affected k have new values.
                # new_global_max = max(max over unaffected k, max over affected k)
                # max over unaffected k <= cur_max (could be less if old max was at affected k)
                # So new_global_max = max(cur_max, new_max_from_affected) only if old max was unaffected
                # If old max was at an affected k, we need to recompute.
                
                # For correctness, let's just recompute max over all k. O(N) per flip.
                # For N up to 512, this is acceptable.
                new_global_max = float(np.max(np.abs(C)))
                
                new_score = new_global_max / (new_S * new_S)
                
                if new_score < cur_score - 1e-15:
                    # Accept the flip
                    f[i] = -v
                    S = new_S
                    cur_max = new_global_max
                    cur_score = new_score
                    improved = True
                    break
                else:
                    # Revert C
                    for k, old_val in old_C_affected.items():
                        C[k] = old_val
            
            if improved:
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_f = f.copy()
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
        
        return best_f, best_score

    def _random_restart_search(f_init, n_restarts=5, deadline=DEADLINE):
        """Try multiple random perturbations and local search from each."""
        N = len(f_init)
        best_f = f_init.copy()
        best_score = _score(f_init)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Create a perturbed version
            f_pert = f_init.copy()
            # Flip a random fraction of positions
            n_flips = max(1, N // 8)
            flip_idx = np.random.choice(N, size=n_flips, replace=False)
            f_pert[flip_idx] *= -1
            
            refined, score = _fast_local_search(f_pert, max_iters=10000, deadline=deadline)
            if score < best_score:
                best_score = score
                best_f = refined
        
        return best_f, best_score

    # Generate candidate initial sequences
    candidates = []
    
    # Legendre sequences
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    for p in primes:
        seq = _legendre_sequence(p)
        candidates.append(seq)
    
    # Barker sequences
    for seq in _barker_sequences():
        candidates.append(seq)
    
    # M-sequences
    for n in range(3, 9):
        seq = _m_sequence(n)
        if seq is not None:
            candidates.append(seq)
    
    # Target lengths to try
    target_lengths = [16, 32, 64, 128, 256]
    
    best_overall_score = 1e10
    best_overall_f = None
    
    # First pass: find the best initial candidate
    for target_len in target_lengths:
        if time.time() > DEADLINE:
            break
        
        for cand in candidates:
            if time.time() > DEADLINE:
                break
            
            f = _pad_to_length(cand, target_len)
            s = _score(f)
            if s < best_overall_score:
                best_overall_score = s
                best_overall_f = f.copy()
    
    # Also try some random initializations
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        for N in [64, 128, 256]:
            f = np.random.choice([-1.0, 1.0], size=N)
            s = _score(f)
            if s < best_overall_score:
                best_overall_score = s
                best_overall_f = f.copy()
    
    if best_overall_f is None:
        best_overall_f = np.ones(64, dtype=np.float64)
        best_overall_score = _score(best_overall_f)
    
    # Run fast local search on the best candidate
    if time.time() < DEADLINE - 10:
        refined_f, refined_score = _fast_local_search(best_overall_f, max_iters=80000, deadline=DEADLINE - 5)
        if refined_score < best_overall_score:
            best_overall_score = refined_score
            best_overall_f = refined_f
    
    # Try random restarts from the best solution
    if time.time() < DEADLINE - 10:
        best_f2, best_score2 = _random_restart_search(best_overall_f, n_restarts=3, deadline=DEADLINE - 5)
        if best_score2 < best_overall_score:
            best_overall_score = best_score2
            best_overall_f = best_f2
    
    # If we have time, try local search on other good candidates
    if time.time() < DEADLINE - 15:
        # Try a few more structured initializations
        for p in [11, 13, 17, 23, 31, 47, 71, 127]:
            if time.time() > DEADLINE - 10:
                break
            seq = _legendre_sequence(p)
            for target_len in [64, 128, 256]:
                if time.time() > DEADLINE - 10:
                    break
                f = _pad_to_length(seq, target_len)
                s = _score(f)
                if s < best_overall_score * 1.05:
                    refined_f, refined_score = _fast_local_search(f, max_iters=20000, deadline=DEADLINE - 5)
                    if refined_score < best_overall_score:
                        best_overall_score = refined_score
                        best_overall_f = refined_f
    
    return best_overall_f
