"""Erdos minimum overlap - Markov chain parameterization on cycle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    def markov_h(p_stay):
        """Generate step heights from a random walk on a cycle.
        p_stay: probability to stay, (1-p_stay)/2 to move left, (1-p_stay)/2 to move right.
        Stationary distribution is uniform (1/N), so we use the transition structure
        to shape the autocorrelation while keeping sum = N/2 via scaling.
        """
        # For a random walk on cycle, the spectral decomposition gives
        # eigenvalues lambda_k = p_stay + (1-p_stay)*cos(2*pi*k/N)
        # The autocorrelation at lag m is sum_k lambda_k^m * (1/N) * exp(2*pi*i*k*m/N)
        # For real symmetric: autocorr(m) = (1/N) * sum_k lambda_k^m * cos(2*pi*k*m/N)
        
        # We want h to be the "shape" that gives desired autocorrelation.
        # If we set h_i = (1/N) * sum_k a_k * cos(2*pi*k*i/N) with a_0 = N/2 (DC)
        # then autocorr(m) = (1/N) * sum_k a_k * a_k * cos(2*pi*k*m/N) / N ... 
        # 
        # Actually, let's think differently. If h is the stationary distribution of a
        # Markov chain, it's uniform. Instead, let h be a function whose Fourier 
        # coefficients are controlled by the Markov chain eigenvalues.
        
        # Better approach: h_i = (1/N) * (1 + sum_{k=1}^{N/2} b_k * cos(2*pi*k*i/N))
        # where b_k are determined by the Markov chain parameter.
        # 
        # For a two-state-like structure on the cycle:
        # Let's use h_i = (1/N) * (1 + A * cos(2*pi*i/N) + B * cos(4*pi*i/N) + ...)
        # 
        # Simplest: single frequency modulation
        # h_i = (1/N) * (1 + a * cos(2*pi*i/N))
        # Sum = N/2 * (1 + 0) = N/2 (since sum of cos over full period = 0)
        # Autocorr at lag m: (1/N^2) * sum_i h_i * h_{i-m}
        # = (1/N^2) * sum_i [(1/N)(1 + a*cos(2*pi*i/N))] * [(1/N)(1 + a*cos(2*pi*(i-m)/N))]
        # = (1/N^3) * sum_i [1 + a*cos(2*pi*i/N) + a*cos(2*pi*(i-m)/N) + a^2*cos(2*pi*i/N)*cos(2*pi*(i-m)/N)]
        # = (1/N^3) * [N + 0 + 0 + a^2 * N/2 * cos(2*pi*m/N)]
        # = (1/N^2) * [1 + a^2/2 * cos(2*pi*m/N)]
        
        # So autocorr(m) = (1/N^2) * (1 + a^2/2 * cos(2*pi*m/N))
        # Max autocorr at m=0: (1/N^2)(1 + a^2/2)
        # Max non-zero autocorr: at m=1 or N-1: (1/N^2)(1 + a^2/2 * cos(2*pi/N))
        # 
        # Overlap at lag m: N/2 - N * autocorr(m) = N/2 - (1/N)(1 + a^2/2*cos(2*pi*m/N))
        # Wait, I need to be careful with normalization.
        
        # Let me redo: overlap(m) = sum_i h_i * (1 - h_{i-m}) = sum_i h_i - sum_i h_i h_{i-m}
        # = N/2 - (N * autocorr_h(m)) where autocorr_h(m) = (1/N) sum_i h_i h_{i-m}
        
        # With h_i = (1/N)(1 + a*cos(2*pi*i/N)):
        # autocorr_h(m) = (1/N) sum_i h_i h_{i-m} = (1/N^3) sum_i (1+a*cos(2*pi*i/N))(1+a*cos(2*pi*(i-m)/N))
        # = (1/N^3) [N + a^2 * (N/2) cos(2*pi*m/N)]
        # = (1/N^2) + (a^2/(2*N)) cos(2*pi*m/N)
        
        # overlap(m) = N/2 - N * autocorr_h(m) = N/2 - 1/N - (a^2/2) cos(2*pi*m/N)
        
        # Max overlap over m != 0: N/2 - 1/N - (a^2/2) * min_{m!=0} cos(2*pi*m/N)
        # min cos is at m=N/2: cos(pi) = -1
        # So max overlap = N/2 - 1/N + a^2/2
        
        # This INCREASES with a^2! So a=0 (uniform) is best for single frequency?
        # That can't be right. Let me reconsider.
        
        # Actually for a=0: overlap(m) = N/2 - 1/N for all m. Score = N/2 - 1/N.
        # For a != 0: the overlap varies, and the MAX is at m=N/2 where cos=-1,
        # giving overlap = N/2 - 1/N + a^2/2 > N/2 - 1/N.
        
        # So uniform is locally optimal for this family! We need a different approach.
        
        # The issue: we need to MINIMIZE the max overlap, which means we want
        # autocorr(m) to be as UNIFORM as possible (all equal to some c).
        # Uniform h gives autocorr(m) = 1/N for all m, overlap = N/2 - 1.
        
        # To do better, we need autocorr(m) < 1/N for all m != 0, but that's 
        # impossible since autocorr(0) = sum h_i^2 >= (sum h_i)^2/N = N/4, 
        # and by Parseval, sum_m autocorr(m)^2 = autocorr(0)^2 ... 
        
        # Actually the constraint is: autocorr(0) = sum h_i^2, and 
        # sum_m autocorr(m) = (sum h_i)^2 / N = N/4 (for the normalized version)
        # 
        # Let's use the DFT approach directly. Let H_k = sum_i h_i exp(-2*pi*i*k*i/N)
        # Then autocorr(m) = (1/N) sum_k |H_k|^2 exp(2*pi*i*k*m/N)
        # 
        # We want to minimize max_{m!=0} [N/2 - N*autocorr(m)] = maximize min_{m!=0} autocorr(m)
        # subject to h_i in [0,1], sum h_i = N/2.
        
        # H_0 = sum h_i = N/2
        # autocorr(0) = (1/N) sum_k |H_k|^2 = sum h_i^2
        # 
        # The optimal solution likely has |H_k| constant for k != 0, 
        # making autocorr(m) constant for m != 0.
        
        # Let |H_k|^2 = c for k != 0. Then:
        # sum h_i^2 = (1/N)[(N/2)^2 + (N-1)*c]
        # autocorr(m) = (1/N)[(N/2)^2 + c * sum_{k!=0} exp(2*pi*i*k*m/N)]
        # For m != 0: sum_{k!=0} exp(2*pi*i*k*m/N) = -1
        # autocorr(m) = (1/N)[N^2/4 - c]
        
        # overlap(m) = N/2 - N*autocorr(m) = N/2 - N^2/4 + c
        # 
        # We need h_i in [0,1] and sum h_i = N/2.
        # sum h_i^2 = N^2/4N + (N-1)c/N = N/4 + (N-1)c/N
        
        # By Cauchy-Schwarz: sum h_i^2 >= (sum h_i)^2/N = N/4
        # So (N-1)c/N >= 0, i.e., c >= 0.
        
        # Also h_i <= 1: sum h_i^2 <= sum h_i = N/2 (since h_i <= 1 means h_i^2 <= h_i)
        # So N/4 + (N-1)c/N <= N/2 => (N-1)c/N <= N/4 => c <= N^2/(4(N-1))
        
        # To minimize max overlap = N/2 - N^2/4 + c, we want c as small as possible: c=0
        # Then overlap = N/2 - N^2/4 = N(1/2 - N/4) = N(2-N)/4
        # For N=100: 100*(2-100)/4 = 100*(-98)/4 = -2450. Negative? That's wrong.
        
        # I'm making an error in normalization. Let me restart more carefully.
        
        # overlap(m) = sum_{i=0}^{N-1} h_i * (1 - h_{(i-m) mod N})
        # = sum_i h_i - sum_i h_i h_{i-m}
        # = N/2 - S(m) where S(m) = sum_i h_i h_{i-m}
        
        # S(0) = sum h_i^2
        # S(m) for m != 0: by DFT, S(m) = (1/N) sum_k |H_k|^2 exp(2*pi*i*k*m/N)
        # where H_k = sum_i h_i exp(-2*pi*i*k*i/N)
        
        # H_0 = N/2
        # We want to minimize max_{m!=0} [N/2 - S(m)] = maximize min_{m!=0} S(m)
        
        # If |H_k| = d for all k != 0, then:
        # S(m) = (1/N)[(N/2)^2 + d^2 * sum_{k=1}^{N-1} exp(2*pi*i*k*m/N)]
        # For m != 0: sum_{k=1}^{N-1} exp(2*pi*i*k*m/N) = -1
        # S(m) = (1/N)[N^2/4 - d^2]
        
        # max overlap = N/2 - S(m) = N/2 - N/4 + d^2/N = N/4 + d^2/N
        
        # Constraints:
        # 1) sum h_i^2 = S(0) = (1/N)[N^2/4 + (N-1)d^2] = N/4 + (N-1)d^2/N
        # 2) h_i in [0,1]: sum h_i^2 <= sum h_i = N/2 (since h_i^2 <= h_i)
        #    So N/4 + (N-1)d^2/N <= N/2 => (N-1)d^2/N <= N/4 => d^2 <= N^2/(4(N-1))
        # 3) h_i >= 0: need to verify, but for the "flat spectrum" solution, 
        #    h_i = (1/N)[N/2 + d^2 * sum_{k!=0} exp(2*pi*i*k*i/N)] ... no, that's not right.
        #    h_i = (1/N) sum_k H_k exp(2*pi*i*k*i/N). If H_k has magnitude d and 
        #    arbitrary phase, h_i = (1/N)[N/2 + sum_{k!=0} d * exp(i*theta_k) * exp(2*pi*i*k*i/N)]
        #    For h_i to be real and non-negative, we need to choose phases carefully.
        #    Simplest: all H_k = d (real positive). Then:
        #    h_i = (1/N)[N/2 + d * sum_{k=1}^{N-1} exp(2*pi*i*k*i/N)]
        #    = (1/N)[N/2 + d * (-1)]  (since sum_{k=0}^{N-1} exp(...) = 0, so sum_{k=1}^{N-1} = -1)
        #    = (1/N)[N/2 - d]
        #    This is constant! h_i = (N/2 - d)/N for all i. Sum = N * (N/2-d)/N = N/2 - d.
        #    But we need sum = N/2, so d = 0. Contradiction.
        
        # The issue: if all H_k have the same phase, h is constant. We need varying phases.
        # For the "flat magnitude" spectrum with random phases, h_i will vary.
        
        # Let's try: H_0 = N/2, H_k = d * exp(i*phi_k) for k != 0.
        # h_i = (1/N) sum_k H_k exp(2*pi*i*k*i/N) = (1/N)[N/2 + sum_{k=1}^{N-1} d*exp(i*phi_k)*exp(2*pi*i*k*i/N)]
        
        # To make h_i non-negative and in [0,1], we can choose phi_k to make the 
        # sum term non-negative. But this is complex.
        
        # Alternative: use a different parameterization. Let's directly optimize 
        # in the cosine basis with the constraint that h_i in [0,1].
        
        return None  # placeholder, will implement below
    
    # Direct optimization in cosine basis
    # h_i = 1/2 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N)
    # Sum h_i = N/2 + sum a_k * sum cos(...) = N/2 (since sum cos over full period = 0)
    # So sum constraint is automatically satisfied!
    # 
    # h_i in [0,1] => 1/2 + sum a_k cos(2*pi*k*i/N) in [0,1] for all i
    # => -1/2 <= sum a_k cos(2*pi*k*i/N) <= 1/2 for all i
    #
    # Overlap at lag m: N/2 - sum_i h_i h_{i-m}
    # sum_i h_i h_{i-m} = sum_i (1/2 + sum_k a_k cos(2*pi*k*i/N)) * (1/2 + sum_l a_l cos(2*pi*l*(i-m)/N))
    # = sum_i [1/4 + (1/2)sum_k a_k cos(2*pi*k*i/N) + (1/2)sum_l a_l cos(2*pi*l*(i-m)/N) + sum_{k,l} a_k a_l cos(2*pi*k*i/N)cos(2*pi*l*(i-m)/N)]
    # = N/4 + 0 + 0 + sum_k a_k^2 * (N/2) cos(2*pi*k*m/N)  (using orthogonality)
    # = N/4 + (N/2) sum_k a_k^2 cos(2*pi*k*m/N)
    #
    # overlap(m) = N/2 - N/4 - (N/2) sum_k a_k^2 cos(2*pi*k*m/N)
    # = N/4 - (N/2) sum_k a_k^2 cos(2*pi*k*m/N)
    #
    # We want to minimize max_{m=1,...,N-1} overlap(m)
    # = N/4 - (N/2) * min_{m!=0} [sum_k a_k^2 cos(2*pi*k*m/N)]
    #
    # To minimize max overlap, we want to MAXIMIZE min_{m!=0} [sum_k a_k^2 cos(2*pi*k*m/N)]
    
    # Let's use a few cosine modes. Start with K=10 modes.
    K = 20
    
    def overlap_values(a):
        """Compute overlap at all non-zero lags."""
        # a: array of shape (K,)
        # sum_k a_k^2 cos(2*pi*k*m/N) for m = 1,...,N-1
        # Use DFT for efficiency
        # f(m) = sum_{k=1}^{K} a_k^2 cos(2*pi*k*m/N)
        # This is the real part of sum_{k=1}^{K} a_k^2 exp(2*pi*i*k*m/N)
        # Let b_k = a_k^2 for k=1..K, and b_0 = 0, b_k = 0 for k > K
        # f(m) = Re[sum_{k=1}^{K} b_k exp(2*pi*i*k*m/N)]
        
        b = np.zeros(N, dtype=np.complex128)
        b[1:K+1] = a**2
        # f(m) = Re[IFFT(b) * N] ... let me be careful
        # DFT: B_k = sum_m f(m) exp(-2*pi*i*k*m/N)
        # IFFT: f(m) = (1/N) sum_k B_k exp(2*pi*i*k*m/N)
        # So if we set B_k = b_k for k=1..K, 0 otherwise, then
        # f(m) = (1/N) sum_{k=1}^K b_k exp(2*pi*i*k*m/N)
        # But we want f(m) = sum_{k=1}^K b_k cos(2*pi*k*m/N) = Re[sum_{k=1}^K b_k exp(2*pi*i*k*m/N)]
        # So f(m) = N * Re[IFFT(b)[m]]
        
        f = np.real(np.fft.ifft(b) * N)
        # f[0] = sum b_k, f[m] for m != 0
        return f
    
    def max_overlap(a):
        f = overlap_values(a)
        # overlap(m) = N/4 - (N/2) * f(m) for m = 1,...,N-1
        # max overlap = N/4 - (N/2) * min(f[1:])
        return N/4 - (N/2) * np.min(f[1:])
    
    def h_from_a(a):
        """Reconstruct h from cosine coefficients."""
        i = np.arange(N)
        h = np.full(N, 0.5)
        for k in range(1, K+1):
            h += a[k-1] * np.cos(2*np.pi*k*i/N)
        return h
    
    def objective(a):
        """Objective: max overlap, with penalty for h_i outside [0,1]."""
        h = h_from_a(a)
        # Penalty for violation
        viol = np.maximum(0, 1 - h) + np.maximum(0, h - 1)  # should be 0 if in [0,1]
        # Actually: h < 0 => penalty (0 - h), h > 1 => penalty (h - 1)
        viol = np.maximum(0, -h) + np.maximum(0, h - 1)
        penalty = 100.0 * np.sum(viol**2)
        return max_overlap(a) + penalty
    
    # Initial guess: small coefficients
    a0 = np.zeros(K)
    a0[0] = 0.1  # small first harmonic
    
    best_a = a0.copy()
    best_score = objective(a0)
    
    # Optimize
    try:
        result = minimize(objective, a0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < best_score:
            best_a = result.x
            best_score = result.fun
    except:
        pass
    
    # Check if time allows for refinement
    if time.time() < deadline - 30:
        # Try multiple random restarts
        np.random.seed(42)
        for trial in range(20):
            a_init = np.random.randn(K) * 0.05
            try:
                result = minimize(objective, a_init, method='L-BFGS-B',
                                  options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if result.fun < best_score:
                    best_a = result.x
                    best_score = result.fun
            except:
                pass
            if time.time() > deadline - 30:
                break
    
    # Final polish with more iterations
    if time.time() < deadline - 10:
        try:
            result = minimize(objective, best_a, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_score:
                best_a = result.x
                best_score = result.fun
        except:
            pass
    
    h = h_from_a(best_a)
    
    # Clamp to [0,1] and renormalize to ensure sum = N/2
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    diff = N/2 - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute the difference
        h += diff / N
        h = np.clip(h, 0, 1)
        # Second pass
        diff = N/2 - np.sum(h)
        h += diff / N
    
    return h
