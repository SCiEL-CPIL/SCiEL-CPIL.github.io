import numpy as np
import time

N = 100
W = 2.0 / N
TARGET_SUM = N / 2.0

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    # overlap_k = W * sum_j h[j] * (1 - h[(j+k)%N])
    # = W * (sum_j h[j] - sum_j h[j]*h[(j+k)%N])
    # = W * (N/2 - autocorr[k])
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    return np.max(overlaps)

def _project_to_box_sum(h, target_sum):
    """
    Project a vector h to the set { x in [0,1]^N : sum(x) = target_sum }.
    Uses a bisection on the shift parameter tau for x = clip(h - tau, 0, 1).
    """
    # Initial guess for tau
    # We want sum(clip(h - tau, 0, 1)) = target_sum
    # If h is already in [0,1], sum(h) might be > or < target_sum.
    # Let f(tau) = sum(clip(h - tau, 0, 1)) - target_sum.
    # f is continuous and decreasing.
    
    # Find range for tau
    # If tau is very small, clip(h-tau, 0, 1) ~ h (if h>=0) or 0 (if h<0)
    # If tau is very large, clip is 0.
    
    # Robust bounds
    min_h = np.min(h)
    max_h = np.max(h)
    
    # If target_sum is close to 0 or N, tau might need to be outside [min_h, max_h]
    # But since we want sum=N/2=50, and h is roughly in [0,1], tau should be in [min_h, max_h]
    # However, if h has values > 1 or < 0, we need to be careful.
    
    lo = min_h - 1.0
    hi = max_h + 1.0
    
    # Ensure f(lo) > 0 and f(hi) < 0
    # f(tau) = sum(clip(h - tau, 0, 1)) - target_sum
    
    def f(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0)) - target_sum
    
    # Expand bounds if necessary
    # This is a safety measure, usually not needed if h is reasonable
    while f(lo) < 0:
        lo -= 1.0
    while f(hi) > 0:
        hi += 1.0
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(h - tau, 0.0, 1.0)

def _evaluate_phases(phases, magnitudes):
    """
    Evaluate the objective for given phases and fixed magnitudes.
    Returns (value, h).
    """
    # Construct complex spectrum
    # H[k] = |H[k]| * exp(i * phi[k])
    # Note: For real h, H must satisfy Hermitian symmetry: H[N-k] = conj(H[k])
    # So we only define phases for k=0..N/2 and mirror them.
    
    H = np.zeros(N, dtype=complex)
    
    # DC component (k=0)
    H[0] = magnitudes[0]
    
    # Nyquist (k=N/2) if N is even
    if N % 2 == 0:
        H[N//2] = magnitudes[N//2] * np.exp(1j * phases[N//2])
        # Actually, for real signal, H[N/2] must be real. 
        # So phase must be 0 or pi. 
        # Let's force it to be real by taking real part or constraining phase.
        # To keep it simple, let's just use the magnitude and assume phase 0 or adjust later.
        # Better: Let's define phases for k=1..N/2-1 and handle k=0, N/2 specially.
        
    # Let's refine the spectrum construction to ensure real output.
    # We will generate a full complex spectrum that is Hermitian.
    
    # k=0 is real.
    # k=N/2 is real (if N even).
    # k=1..N/2-1: H[k] = M[k] * exp(i*phi[k]), H[N-k] = conj(H[k])
    
    # Let's re-define the input 'phases' to be of length N//2 + 1 (for k=0..N/2)
    # But k=0 phase is irrelevant (real). k=N/2 phase must be 0 or pi.
    
    # Let's just use the first N//2 phases for the positive frequencies.
    # H[0] = M[0]
    # For k=1 to N//2 - 1:
    #   H[k] = M[k] * exp(i * phi[k])
    #   H[N-k] = conj(H[k])
    # If N even:
    #   H[N//2] = M[N//2] * cos(phi[N//2]) ? No, H[N//2] must be real.
    #   Let's force H[N//2] to be M[N//2] * (1 if phi[N//2] < pi/2 else -1). 
    #   Or simpler: just use M[N//2] and ignore phase for Nyquist, or allow phase to modulate sign.
    
    # Let's simplify: We optimize phases for k=1..N-1 but enforce symmetry in the evaluation.
    # Input: phases (length N//2), magnitudes (length N)
    
    H = np.zeros(N, dtype=complex)
    H[0] = magnitudes[0]
    
    half = N // 2
    for k in range(1, half):
        H[k] = magnitudes[k] * np.exp(1j * phases[k])
        H[N-k] = np.conj(H[k])
        
    if N % 2 == 0:
        # Nyquist frequency
        # H[N//2] must be real.
        # Let's use the phase to determine the sign: cos(phi) is not quite right if we want continuous.
        # Actually, if we treat H[N//2] as a real number, we can just use M[N//2] * cos(phi[N//2])?
        # No, magnitude is fixed. So H[N//2] = M[N//2] * sign(cos(phi[N//2]))?
        # Let's just fix H[N//2] = M[N//2] for simplicity, or allow it to vary?
        # If we fix magnitudes, we can't vary the magnitude.
        # So H[N//2] is either M[N//2] or -M[N//2].
        # Let's use the sign of cos(phases[half]) if we had a phase there.
        # Since we only passed half phases, let's assume phases has length half.
        # If N=100, half=50. phases[1..49] are used. phases[50] is not in the list if length is 50 (indices 0..49).
        # Let's assume phases is length N//2. Index 0 is k=0 (ignored), 1..N//2-1 used.
        # If N even, k=N//2 is the last one.
        # Let's define phases to have length N//2.
        # k=1..N//2-1 use phases[k].
        # k=N//2: Let's just set H[N//2] = magnitudes[N//2]. The phase doesn't matter for magnitude, 
        # but the sign does. Let's flip the sign based on a random bit or just keep it positive.
        # Actually, to allow optimization, we can treat the Nyquist component as having a "phase" of 0 or pi.
        # Let's just keep it positive for now.
        H[N//2] = magnitudes[N//2]
        
    h = np.fft.ifft(H).real
    
    # Project to feasible set [0,1] with sum N/2
    h_proj = _project_to_box_sum(h, TARGET_SUM)
    
    val = _overlap_max(h_proj)
    return val, h_proj

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Define Magnitudes
    # We want to minimize max overlap.
    # overlap_k = W * (N/2 - autocorr_k)
    # To minimize max overlap, we want autocorr_k to be as large as possible (close to N/2) for all k != 0.
    # Max possible autocorr is N/2 (if h is binary and balanced? No, if h is constant 0.5, autocorr is N/4? No.
    # If h is binary 0/1 with 50 ones:
    # autocorr_k = sum h[j]h[j+k].
    # If h is random binary, autocorr_k ~ N/4.
    # Overlap = W * (N/2 - N/4) = W * N/4 = (2/N) * N/4 = 0.5.
    # If we can make autocorr_k closer to N/2, overlap is smaller.
    # Max autocorr_k for k!=0 is limited by the structure.
    # For a "perfect" sequence (like Barker or M-sequence), the sidelobes of autocorrelation are low (close to 0).
    # Wait. If autocorr_k is close to 0, then overlap = W * (N/2 - 0) = W * N/2 = (2/N) * N/2 = 1.0.
    # This is the MAXIMUM overlap.
    # We want to MINIMIZE overlap.
    # So we want to MAXIMIZE autocorr_k.
    # The maximum autocorr_k for k!=0 is when the signal is "self-similar" at shift k.
    # If h is periodic with period k, autocorr_k is high.
    # But h must have sum N/2.
    # If h is constant 0.5, autocorr_k = N * 0.25 = N/4.
    # Overlap = W * (N/2 - N/4) = 0.5.
    # Can we do better?
    # If h is binary, autocorr_k can be as high as N/2 (if the pattern repeats).
    # Example: h = [1, 0, 1, 0, ...]. N=100.
    # autocorr_1 = 0. Overlap = 1.0.
    # autocorr_2 = 50. Overlap = W * (50 - 50) = 0.
    # But we need max over ALL k.
    # For h=[1,0,1,0...], autocorr_k is 0 for odd k, 50 for even k.
    # Max overlap is 1.0 (for odd k).
    
    # So we want to maximize the MINIMUM autocorr_k (for k!=0).
    # This is equivalent to minimizing the maximum deviation of autocorr from the mean?
    # No, we want autocorr_k to be as HIGH as possible for all k.
    # The theoretical maximum for autocorr_k (k!=0) given sum=N/2 and bounds [0,1] is...
    # If we can make autocorr_k = N/2 for all k, overlap is 0.
    # This requires h[j]h[j+k] = 1/2 * h[j] + ...? No.
    # Sum h[j]h[j+k] = N/2.
    # This implies the signal is "uncorrelated" in a specific way?
    # Actually, if h is a "perfect" binary sequence, the autocorrelation sidelobes are 0.
    # We want them to be HIGH.
    # This is the opposite of the usual "low autocorrelation" design.
    # We want a signal that is highly correlated with its shifts.
    # The most correlated signal is a constant signal.
    # h=0.5 gives autocorr = N/4.
    # Can we get higher?
    # Consider h = [1, 1, ..., 1, 0, 0, ...] (50 ones, 50 zeros).
    # autocorr_k depends on k.
    # For k=1, if the block is contiguous, autocorr_1 = 49 (boundary effects).
    # Overlap = W * (50 - 49) = 2/100 = 0.02.
    # For k=50, autocorr_50 = 0 (shifted by half, ones meet zeros).
    # Overlap = W * (50 - 0) = 1.0.
    # So blocky signals are bad for large shifts.
    
    # We need a signal that is "uniform" in its correlation properties.
    # The uniform signal h=0.5 gives autocorr_k = N/4 for all k.
    # Overlap = 0.5 for all k.
    # Is it possible to get autocorr_k > N/4 for all k?
    # Sum of autocorr_k over all k is |FFT|^2 sum? No.
    # Sum_{k=0}^{N-1} autocorr_k = sum_j h[j]^2.
    # If h is binary, sum h[j]^2 = N/2.
    # Mean autocorr = (N/2) / N = 1/2.
    # Wait, autocorr_k is defined as sum_j h[j]h[j+k].
    # For binary, sum h[j]^2 = N/2.
    # So the average autocorr over all shifts is 0.5.
    # If we can make ALL autocorr_k equal to 0.5, then overlap = W * (0.5 - 0.5) = 0?
    # No, the formula is W * (N/2 - autocorr_k).
    # Here N/2 = 50.
    # If autocorr_k = 0.5, overlap = W * 49.5 = 0.99.
    # Wait, my previous calculation for h=0.5:
    # h=0.5 is not binary.
    # sum h[j]^2 = N * 0.25 = N/4.
    # Mean autocorr = (N/4)/N = 0.25.
    # Overlap = W * (50 - 0.25) = 0.995.
    # My previous "0.5" calculation was wrong.
    # Let's re-evaluate h=0.5.
    # autocorr_k = sum 0.5*0.5 = N * 0.25 = N/4.
    # overlap_k = W * (N/2 - N/4) = W * N/4 = (2/N) * N/4 = 0.5.
    # Yes, 0.5 is correct.
    
    # For binary h:
    # Mean autocorr = 0.5.
    # If autocorr_k = 0.5 for all k, overlap = W * (50 - 0.5) = 49.5 * 0.02 = 0.99.
    # This is much worse than 0.5.
    
    # So we want to MINIMIZE the variance of autocorr_k?
    # No, we want to MAXIMIZE autocorr_k.
    # The max possible autocorr_k is constrained by the sum of squares.
    # For h in [0,1], sum h[j]^2 <= sum h[j] = N/2 (if h is 0/1).
    # If h is continuous, sum h[j]^2 can be smaller (if h is spread out).
    # Min sum h[j]^2 is N * (1/N)^2 * N? No.
    # Min sum h[j]^2 subject to sum h[j] = N/2, 0<=h<=1 is achieved when h is as uniform as possible.
    # h=0.5 gives sum h[j]^2 = N/4.
    # This is the MINIMUM possible sum of squares.
    # Therefore, the MEAN autocorr is MINIMIZED at h=0.5.
    # We want to MAXIMIZE autocorr.
    # So we want h to be as "peaky" as possible (binary).
    # But we want the MAX overlap to be small.
    # Max overlap corresponds to MIN autocorr.
    # So we want to MAXIMIZE the MIN autocorr_k.
    
    # This is a "max-min" problem.
    # We want the autocorrelation function to be as flat and high as possible.
    
    # Spectral approach:
    # autocorr_k = IFFT(|H|^2)_k.
    # We want IFFT(|H|^2) to be constant (ideally high).
    # If |H|^2 is constant, then IFFT is a delta function (k=0 only).
    # This gives autocorr_0 = N/2 (for binary), and autocorr_k=0 for k!=0.
    # This gives Max Overlap = W * (N/2 - 0) = 1.0.
    
    # We want the OPPOSITE: |H|^2 to be concentrated at k=0?
    # If |H|^2 is a delta at k=0, then h is constant.
    # h=0.5 -> |H|^2 is delta at 0.
    # autocorr is constant N/4.
    # Overlap is 0.5.
    
    # So the "flat spectrum" (white noise) is BAD.
    # The "delta spectrum" (constant signal) is GOOD.
    
    # So we should NOT use random phases with flat magnitude.
    # We should use a magnitude spectrum that is concentrated at low frequencies.
    
    # Strategy:
    # 1. Define a magnitude spectrum M[k] that is peaked at k=0.
    #    e.g., M[k] = exp(-alpha * k^2).
    # 2. Optimize phases to ensure the resulting h is in [0,1] and has sum N/2.
    #    The projection will handle the sum/bounds, but the phases determine the shape.
    #    Actually, if the spectrum is too concentrated, h will be a smooth bump.
    #    We need h to be in [0,1].
    
    # Let's try a mix:
    # Start with a "smooth" spectrum (low freq dominant) and optimize phases to "shape" the time-domain signal to be more binary-like while maintaining the high autocorrelation.
    
    # Actually, if we fix M[k] to be smooth, the time domain signal is smooth.
    # Smooth signals have high autocorrelation.
    # The main issue is the bounds [0,1] and sum=N/2.
    # A smooth Gaussian-like pulse might exceed 1 or have sum != N/2.
    # We can scale it.
    
    # Let's try:
    # M[k] = exp(-beta * k^2).
    # Optimize phases to minimize the "cost" of projection (i.e., make h close to [0,1] before projection) or just minimize overlap after projection.
    
    # Since we have a genetic algorithm, we can just search for phases that, when combined with a fixed smooth M, yield a good h after projection.
    
    # Let's pick a few different M profiles (different beta) and run GA on phases for each.
    
    def _run_ga_for_spectrum(magnitudes, n_pop=50, n_gen=200, deadline=None):
        """
        Run GA to optimize phases for a fixed magnitude spectrum.
        """
        half = N // 2
        # Phases for k=1..N//2-1 (and maybe N//2)
        # Let's optimize phases for k=1..N//2
        n_phases = half
        
        # Initialize population
        pop = np.random.uniform(0, 2 * np.pi, (n_pop, n_phases))
        best_val = np.inf
        best_h = None
        best_phases = None
        
        for gen in range(n_gen):
            if deadline is not None and time.time() > deadline:
                break
            
            # Evaluate population
            vals = np.zeros(n_pop)
            h_list = [None] * n_pop
            for i in range(n_pop):
                val, h = _evaluate_phases(pop[i], magnitudes)
                vals[i] = val
                h_list[i] = h
                
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    best_phases = pop[i].copy()
            
            # Selection: Keep top 20%
            n_keep = max(5, n_pop // 5)
            sorted_indices = np.argsort(vals)
            parents = pop[sorted_indices[:n_keep]]
            
            # Crossover and Mutation to fill rest
            new_pop = [parents[0].copy()] # Elitism
            
            while len(new_pop) < n_pop:
                # Crossover
                p1 = parents[np.random.randint(n_keep)]
                p2 = parents[np.random.randint(n_keep)]
                
                if np.random.random() < 0.5:
                    # Uniform crossover
                    mask = np.random.randint(0, 2, n_phases)
                    child = np.where(mask == 1, p1, p2)
                else:
                    # Blend crossover
                    alpha = np.random.random()
                    child = alpha * p1 + (1 - alpha) * p2
                    child = np.mod(child, 2 * np.pi)
                
                # Mutation
                if np.random.random() < 0.5:
                    # Gaussian mutation
                    mutation = np.random.normal(0, 0.5, n_phases)
                    child = child + mutation
                    child = np.mod(child, 2 * np.pi)
                
                new_pop.append(child)
            
            pop = np.array(new_pop)
            
            # Adaptive mutation rate based on stagnation?
            # For now, fixed.
            
        return best_val, best_h, best_phases

    best_val = np.inf
    best_h = None
    
    # Try different spectrum shapes
    # beta controls the width. Smaller beta -> wider in time (smoother).
    betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Construct magnitudes
        k = np.arange(N)
        # Circular distance
        dist = np.minimum(k, N - k)
        magnitudes = np.exp(-beta * dist**2)
        
        # Normalize? The absolute scale is handled by projection.
        # But relative scale matters for the shape.
        
        val, h, phases = _run_ga_for_spectrum(magnitudes, n_pop=40, n_gen=100, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
        # If we found a very good solution, break early?
        if best_val < 0.3:
            break

    # If no good solution found, fallback to uniform
    if best_h is None or best_val > 0.5:
        best_h = np.full(N, 0.5)
        best_val = _overlap_max(best_h)
        
    # Final polish with local search on the best spectral solution?
    # The spectral solution might be slightly off due to projection.
    # A few steps of local search might help.
    
    if time.time() < deadline - 10:
        # Simple local search
        h = best_h.copy()
        val = best_val
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(N)
            old_val = h[idx]
            new_val = np.clip(old_val + np.random.normal(0, 0.1), 0, 1)
            
            trial = h.copy()
            trial[idx] = new_val
            
            # Project
            trial = _project_to_box_sum(trial, TARGET_SUM)
            
            new_val_obj = _overlap_max(trial)
            
            if new_val_obj < val:
                h = trial
                val = new_val_obj
                best_val = val
                best_h = h.copy()
    
    return best_h

