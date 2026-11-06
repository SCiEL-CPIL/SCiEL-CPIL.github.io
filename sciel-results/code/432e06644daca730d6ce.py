import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    N = 24
    SQRT_N = math.sqrt(N)
    TARGET = 1.2583758  # Known optimum

    # Dense sampling points for Parseval constraint
    M = 1024
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    powers = np.arange(N)
    # Precompute z^k for each sample point: shape (M, N)
    ZK = np.exp(1j * thetas[:, None] * powers[None, :])

    def eval_poly(c, theta):
        """Evaluate |P(e^{i*theta})| for scalar theta."""
        z = np.exp(1j * theta)
        zk = np.power(z, powers)
        return np.abs(np.dot(zk, c))

    def sup_norm_precise(c, M_dense=8192):
        """Compute sup norm with golden-section refinement."""
        theta = np.linspace(0, 2 * np.pi, M_dense, endpoint=False)
        z = np.exp(1j * theta)
        zk = np.power(z[:, None], powers[None, :])
        mags = np.abs(zk @ c)

        # Find local maxima
        peak_indices = []
        for i in range(M_dense):
            prev_i = (i - 1) % M_dense
            next_i = (i + 1) % M_dense
            if mags[i] >= mags[prev_i] and mags[i] >= mags[next_i]:
                peak_indices.append(i)

        if not peak_indices:
            peak_indices = [int(np.argmax(mags))]

        peak_indices.sort(key=lambda i: mags[i], reverse=True)
        top_peaks = peak_indices[:15]

        best = 0.0
        gr = (np.sqrt(5) - 1) / 2

        for idx in top_peaks:
            t_left = theta[(idx - 1) % M_dense]
            t_right = theta[(idx + 1) % M_dense]
            if t_left > t_right:
                t_right += 2 * np.pi

            a, b = t_left, t_right
            for _ in range(40):
                c1 = b - gr * (b - a)
                c2 = a + gr * (b - a)
                v1 = eval_poly(c, c1)
                v2 = eval_poly(c, c2)
                if v1 < v2:
                    a = c1
                else:
                    b = c2
            mid = (a + b) / 2
            vmid = eval_poly(c, mid)
            best = max(best, vmid)

        return best

    def parseval_residual(c_partial, k):
        """
        Given partial assignment c[0..k-1], compute the minimum possible
        Parseval residual at each sample point.
        Returns array of shape (M,) with min |sum_{j=0}^{k-1} c_j z_j^m|^2
        for each sample point m, where remaining coefficients can be +1 or -1.
        """
        # Current partial sum at each sample point
        partial = ZK[:, :k] @ c_partial  # shape (M,)
        remaining = N - k

        # For each sample point, the remaining sum can range in magnitude
        # The minimum |partial + remaining_sum|^2 where remaining_sum has
        # each term in {+1, -1} * z^j
        # Lower bound: (|partial| - remaining)^2 if |partial| >= remaining, else 0
        # But we need a tighter bound. Use: the remaining terms have fixed magnitudes
        # but we can choose signs. The minimum |partial + S| where S is sum of
        # remaining terms with ±1 coefficients.
        # Simple lower bound: max(0, |partial| - remaining)^2
        lb = np.maximum(0.0, np.abs(partial) - remaining) ** 2
        return lb

    def backtrack(c, k, target_sq, best_c, best_val):
        """
        Backtracking search. c is array of ±1 (or 0 for unassigned).
        k = number of assigned coefficients.
        target_sq = target^2 (sup norm squared bound)
        Returns (best_c, best_val) where best_val is the sup norm found.
        """
        if time.time() > deadline - 5.0:
            return best_c, best_val

        if k == N:
            # Full assignment: compute precise sup norm
            val = sup_norm_precise(c, M_dense=4096)
            if val < best_val:
                best_val = val
                best_c = c.copy()
            return best_c, best_val

        # Try both +1 and -1 for position k
        for sign in [1.0, -1.0]:
            if time.time() > deadline - 5.0:
                return best_c, best_val

            c[k] = sign

            # Arc consistency: check if this partial assignment can possibly
            # achieve sup norm <= target
            # Use Parseval: sum |P(z_m)|^2 over m = N (discrete Parseval)
            # So average |P(z_m)|^2 = 1, meaning at least one point has |P| >= 1
            # More usefully: if at EVERY sample point the minimum possible
            # |P|^2 > target_sq, we can prune.
            residual = parseval_residual(c, k + 1)
            if np.all(residual > target_sq):
                c[k] = 0
                continue

            # Additional pruning: check current partial sum at sample points
            # If |partial_sum| already exceeds target at some point AND remaining
            # terms all have the same sign contribution, we can't fix it.
            # Simpler: just check if current partial magnitude is too large
            partial = ZK[:, :k+1] @ c[:k+1]
            current_max = np.max(np.abs(partial))
            # The remaining N-k-1 terms can at best reduce this by (N-k-1)
            # (if they all oppose the current sum at that point)
            # But this is a loose bound. Use: if current_max - (N-k-1) > target, prune
            if current_max - (N - k - 1) > target_sq ** 0.5:
                c[k] = 0
                continue

            best_c, best_val = backtrack(c, k + 1, target_sq, best_c, best_val)
            c[k] = 0

            # If we found the optimal, stop
            if best_val <= TARGET + 1e-6:
                return best_c, best_val

        return best_c, best_val

    # Strategy: Binary search on target sup norm, using backtracking
    # Start with a reasonable target and refine

    # First, get a good initial solution using Rudin-Shapiro + local search
    def rs_poly():
        p, q = [1.0], [1.0]
        while len(p) < N:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:N], dtype=np.float64)

    def local_search(c, current_val, M=4096):
        """Local search with single and double flips."""
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Single flips
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                trial = c.copy()
                trial[i] = -trial[i]
                val = sup_norm_precise(trial, M_dense=2048)
                if val < current_val - 1e-10:
                    c = trial
                    current_val = val
                    improved = True
            # Double flips
            if time.time() > deadline - 10.0:
                break
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 10.0:
                        break
                    trial = c.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    val = sup_norm_precise(trial, M_dense=2048)
                    if val < current_val - 1e-10:
                        c = trial
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
        return c, current_val

    # Get initial solution
    c0 = rs_poly()
    val0 = sup_norm_precise(c0, M_dense=4096)
    c0, val0 = local_search(c0, val0)

    best_c = c0.copy()
    best_val = val0

    # Now use backtracking with decreasing target
    # Start target slightly above current best
    target = best_val + 0.01
    min_target = TARGET - 0.001

    while target > min_target and time.time() < deadline - 15.0:
        target_sq = target ** 2
        c_partial = np.zeros(N, dtype=np.float64)

        # Use backtracking with the target
        # But full backtracking over 24 bits is 2^24 = 16M, too slow without good pruning
        # Instead, use a smarter approach: fix first few coefficients from best_c,
        # and backtrack on the rest

        # Fix first 8 coefficients from current best
        c_fixed = best_c[:8].copy()
        c_bt = np.zeros(N, dtype=np.float64)
        c_bt[:8] = c_fixed

        # Backtrack on positions 8..23
        def backtrack_tail(c, k, target_sq, best_c, best_val):
            if time.time() > deadline - 10.0:
                return best_c, best_val

            if k == N:
                val = sup_norm_precise(c, M_dense=4096)
                if val < best_val:
                    best_val = val
                    best_c = c.copy()
                return best_c, best_val

            for sign in [1.0, -1.0]:
                if time.time() > deadline - 10.0:
                    return best_c, best_val

                c[k] = sign

                # Pruning: check if partial sum can achieve target
                partial = ZK[:, :k+1] @ c[:k+1]
                remaining = N - k - 1
                # Lower bound on max |P|: at least max(|partial_m| - remaining) over m
                # But we need: is there ANY completion with sup norm <= target?
                # Lower bound on sup norm given partial: max over m of (|partial_m| - remaining)
                # This is very loose. Better: use the fact that sum |P_m|^2 = N
                # So avg |P_m|^2 = 1, and we need max |P_m| <= target
                # This is always satisfiable if target >= 1, so not useful for pruning
                # when target is close to 1.258.

                # Tighter: if at some sample point, |partial_m| - remaining > target, prune
                if np.max(np.abs(partial) - remaining) > target:
                    c[k] = 0
                    continue

                best_c, best_val = backtrack_tail(c, k + 1, target_sq, best_c, best_val)
                c[k] = 0

                if best_val <= TARGET + 1e-6:
                    return best_c, best_val

            return best_c, best_val

        best_c, best_val = backtrack_tail(c_bt, 8, target_sq, best_c, best_val)

        # Update target
        if best_val < target - 0.005:
            target = best_val + 0.005
        else:
            # Try different fixed prefix
            # Fix last 8 instead
            c_fixed = best_c[-8:].copy()
            c_bt = np.zeros(N, dtype=np.float64)
            c_bt[-8:] = c_fixed

            def backtrack_head(c, k, target_sq, best_c, best_val):
                if time.time() > deadline - 10.0:
                    return best_c, best_val

                if k == N:
                    val = sup_norm_precise(c, M_dense=4096)
                    if val < best_val:
                        best_val = val
                        best_c = c.copy()
                    return best_c, best_val

                for sign in [1.0, -1.0]:
                    if time.time() > deadline - 10.0:
                        return best_c, best_val

                    c[k] = sign

                    partial = ZK[:, :k+1] @ c[:k+1]
                    remaining = N - k - 1
                    if np.max(np.abs(partial) - remaining) > target:
                        c[k] = 0
                        continue

                    best_c, best_val = backtrack_head(c, k + 1, target_sq, best_c, best_val)
                    c[k] = 0

                    if best_val <= TARGET + 1e-6:
                        return best_c, best_val

                return best_c, best_val

            best_c, best_val = backtrack_head(c_bt, 0, target_sq, best_c, best_val)

            if best_val < target - 0.005:
                target = best_val + 0.005
            else:
                break

    # Final polish
    if time.time() < deadline - 8.0:
        best_c, best_val = local_search(best_c, best_val)

    # Convert to int list
    result = np.where(best_c > 0, 1, -1).astype(int)
    return result.tolist()
