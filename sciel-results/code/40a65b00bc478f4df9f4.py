"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    We use a chaotic logistic map (r=4) sampled at a fixed phase to generate
    a deterministic, pseudo-random sign sequence. We evaluate the C3 bound
    (max absolute value of the triple autocorrelation) for multiple chaotic
    trajectories and select the one with the minimum bound. A fast greedy
    local search is then applied to the best candidate to further minimize
    the bound.
    """

    # Precompute the padded FFT size for the triple autocorrelation
    # Convolution of 3 sequences of length N has length 3N-2
    N_pad = 1 << (3 * N - 1).bit_length()
    fft_size = N_pad

    def compute_c3_bound(s):
        """
        Compute the upper bound on C3, defined as the maximum absolute value
        of the third-order autocorrelation coefficients.
        """
        s_pad = np.zeros(fft_size, dtype=np.float64)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S * S * S).real
        return np.max(np.abs(C3))

    def generate_signs_from_logistic(seed, r=4.0):
        """
        Generate a sign sequence by iterating the logistic map x_{n+1} = r*x*(1-x)
        and thresholding at 0.5.
        """
        x = seed
        signs = np.empty(N, dtype=np.float64)
        for i in range(N):
            x = r * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs

    # Generate a diverse set of initial conditions for the logistic map
    # We use irrational-like numbers to ensure ergodic, aperiodic trajectories
    seeds = []
    # Add several irrational-like constants
    seeds.append(np.sqrt(2.0) - 1.0)  # ~0.4142
    seeds.append(np.sqrt(3.0) - 1.0)  # ~0.7321
    seeds.append(np.pi / 4.0)        # ~0.7854
    seeds.append(np.e / 4.0)         # ~0.6796
    seeds.append((1.0 + np.sqrt(5.0)) / 2.0 - 1.0)  # golden ratio - 1 ~0.6180
    seeds.append(np.sqrt(2.0) / 2.0)  # ~0.7071
    seeds.append(np.sqrt(3.0) / 3.0)  # ~0.5774
    seeds.append(1.0 / np.sqrt(2.0))  # ~0.7071
    # Add some random-looking but deterministic values
    seeds.append(0.123456789)
    seeds.append(0.987654321)
    seeds.append(0.314159265)
    seeds.append(0.271828183)
    seeds.append(0.618033989)
    seeds.append(0.732050808)

    best_signs = None
    best_bound = float('inf')

    # Evaluate each seed
    for seed in seeds:
        # Ensure seed is in (0,1)
        x0 = seed % 1.0
        if x0 == 0.0:
            x0 = 0.5
        signs = generate_signs_from_logistic(x0)
        bound = compute_c3_bound(signs)

        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()

    # Fast greedy local search on the best candidate
    # We flip individual bits if it reduces the bound
    signs = best_signs
    current_bound = best_bound

    # Set a time limit to ensure we finish within the 120s constraint
    # Allow about 60 seconds for the local search
    start_time = time.time()
    time_limit = 60.0

    # Maximum number of evaluations as a safety cap
    max_evals = 50000
    evals = 0
    improved = True

    while improved and evals < max_evals:
        improved = False
        for i in range(N):
            # Check time limit
            if time.time() - start_time > time_limit:
                break

            if evals >= max_evals:
                break

            # Try flipping bit i
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            evals += 1

            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                # Revert the flip
                signs[i] *= -1.0

        if time.time() - start_time > time_limit:
            break

    return signs
# EVOLVE-BLOCK-END
