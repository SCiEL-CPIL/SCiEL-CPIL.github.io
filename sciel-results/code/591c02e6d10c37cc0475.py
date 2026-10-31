"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin before 150s limit

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

    def _legendre_symbol(p, n):
        """Legendre symbol (p/n) for prime p."""
        if n % p == 0:
            return 0
        return pow(n, (p - 1) // 2, p) if pow(n, (p - 1) // 2, p) == 1 else -1

    def _generate_legendre_sequence(p):
        """Generate a ±1 sequence from Legendre symbols of length p-1."""
        seq = np.array([_legendre_symbol(p, n) for n in range(1, p)], dtype=np.float64)
        return seq

    def _generate_barker_sequence():
        """Return known Barker sequences (lengths 2, 3, 4, 5, 7, 11, 13)."""
        barker = {
            2: [1, 1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, -1, 1, 1],
            7: [1, 1, -1, -1, 1, -1, 1],
            11: [1, -1, -1, 1, -1, 1, 1, 1, -1, 1, -1],
            13: [1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
        }
        return barker

    def _pad_to_length(seq, target_len):
        """Pad or truncate a sequence to target length by repeating."""
        if len(seq) >= target_len:
            return seq[:target_len].astype(np.float64)
        # Repeat and truncate
        reps = (target_len + len(seq) - 1) // len(seq)
        padded = np.tile(seq, reps)
        return padded[:target_len].astype(np.float64)

    def _local_sign_flip_incremental(f, max_iters=2000):
        """Local search via sign flips with incremental convolution update."""
        N = len(f)
        S = np.sum(f)
        if abs(S) < 1e-12:
            return f, 1e10

        # Compute full autoconvolution
        size = 1
        while size < 2 * N - 1:
            size *= 2
        F = np.fft.rfft(f, n=size)
        C = np.fft.irfft(F * F, n=size)
        C = C[:2 * N - 1].copy()
        cur_score = np.max(np.abs(C)) / (S * S)

        best_f = f.copy()
        best_score = cur_score

        no_improve = 0
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break

            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > DEADLINE:
                    break

                v = f[i]
                delta = -2.0 * v  # new_v - v = -v - v = -2v

                # Update S
                new_S = S + delta
                if abs(new_S) < 1e-12:
                    continue

                # Incrementally update C
                # C[k] = sum_j f[j]*f[j+k]
                # When f[i] changes by delta:
                #   C[k] += delta * f[i+k] (term where j=i)
                #   C[k] += delta * f[i-k] (term where j+k=i => j=i-k)
                # But we need to be careful about double counting when i+k == i-k => k=0
                # Actually for k=0: C[0] = sum f[j]^2, change is delta*(f[i]+new_v) = delta*(2*f[i]+delta)
                # For k != 0: C[k] changes by delta*f[i+k] + delta*f[i-k] (if both in range)
                # But wait, this is for the contribution of f[i] to C[k].
                # C[k] = sum_j f[j] * f[j+k]
                # f[i] appears in C[k] as:
                #   j=i: f[i]*f[i+k]
                #   j+k=i => j=i-k: f[i-k]*f[i]
                # So delta_C[k] = delta*f[i+k] + delta*f[i-k] (when indices valid)
                # For k=0: both terms are f[i]*f[i], so delta_C[0] = delta*(f[i] + f[i]) = 2*delta*f[i]
                # But actually new f[i] = f[i]+delta, so f[i]^2 changes to (f[i]+delta)^2 = f[i]^2 + 2*f[i]*delta + delta^2
                # So delta_C[0] = 2*f[i]*delta + delta^2

                # We'll recompute C from scratch for accuracy but use a smarter approach:
                # Just compute the change in max|C| approximately

                # For speed, let's compute the actual change in C values
                # C_new[k] = C[k] + delta*(f[i+k] if 0<=i+k<N else 0) + delta*(f[i-k] if 0<=i-k<N else 0)
                # But for k=0, it's delta*(2*f[i] + delta) instead of delta*2*f[i]

                # Let's just do a quick estimate: compute new max
                # For efficiency, only check if this flip could improve things
                # The maximum |C[k]| can change by at most |delta| * (|f[i+k]| + |f[i-k]|) <= 2*|delta|
                # So if current max is much larger than 2*|delta|, this flip won't help much

                # Simple approach: compute the affected C values
                # Actually, let's just recompute C for the candidate and check
                # This is O(N) per flip which is too slow for large N
                # Let's use the incremental update properly

                # Incremental update of C:
                C_new = C.copy()
                for k in range(2 * N - 1):
                    contrib = 0.0
                    j1 = i + k
                    j2 = i - k
                    if 0 <= j1 < N:
                        contrib += f[j1]
                    if 0 <= j2 < N:
                        contrib += f[j2]
                    if k == 0:
                        # Special case: C[0] = sum f[j]^2
                        # Change: (f[i]+delta)^2 - f[i]^2 = 2*f[i]*delta + delta^2
                        C_new[0] += 2.0 * f[i] * delta + delta * delta
                    else:
                        C_new[k] += delta * contrib

                new_max = np.max(np.abs(C_new))
                new_score = new_max / (new_S * new_S)

                if new_score < cur_score - 1e-12:
                    # Accept the flip
                    f[i] = -v
                    S = new_S
                    C = C_new
                    cur_score = new_score
                    improved = True
                    break

            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 50:
                    break

            # Update best
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()

        return best_f, best_score

    # Generate candidate initial sequences
    candidates = []

    # Legendre sequences for various primes
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        seq = _generate_legendre_sequence(p)
        candidates.append(seq)

    # Barker sequences
    barker_dict = _generate_barker_sequence()
    for L, seq in barker_dict.items():
        candidates.append(np.array(seq, dtype=np.float64))

    # Try different target lengths
    target_lengths = [32, 64, 128, 256, 512]

    best_overall_score = 1e10
    best_overall_f = None

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

    # Also try random ±1 sequences
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        N = 128
        f = np.random.choice([-1.0, 1.0], size=N)
        s = _score(f)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_f = f.copy()

    # Run local search on the best candidate
    if best_overall_f is not None:
        if time.time() < DEADLINE - 20:
            refined_f, refined_score = _local_sign_flip_incremental(best_overall_f, max_iters=3000)
            if refined_score < best_overall_score:
                best_overall_score = refined_score
                best_overall_f = refined_f

    # If we still have time, try local search on other good candidates
    if best_overall_f is not None and time.time() < DEADLINE - 30:
        # Try a few more initializations
        for p in [11, 13, 17, 23, 31]:
            if time.time() > DEADLINE - 25:
                break
            seq = _generate_legendre_sequence(p)
            for target_len in [64, 128]:
                if time.time() > DEADLINE - 25:
                    break
                f = _pad_to_length(seq, target_len)
                s = _score(f)
                if s < best_overall_score * 1.1:  # Only if reasonably close
                    refined_f, refined_score = _local_sign_flip_incremental(f, max_iters=1000)
                    if refined_score < best_overall_score:
                        best_overall_score = refined_score
                        best_overall_f = refined_f

    if best_overall_f is None:
        # Fallback: all ones
        best_overall_f = np.ones(64)

    return best_overall_f
# EVOLVE-BLOCK-END
