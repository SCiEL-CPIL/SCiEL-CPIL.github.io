"""Rewritten program: critical-AP focused local search with multi-restart."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import floor

N = 256
DEADLINE_MARGIN = 2.0  # seconds of safety margin before the 90s kill

def _build_incidence():
    """Build the binary incidence matrix for all APs in [1..256].
    Rows = APs, columns = positions 0..255.
    Entry (r, c) = 1 if position c+1 is in AP r.
    """
    rows = []
    for start in range(1, N + 1):
        max_step = N - start + 1
        for step in range(1, max_step + 1):
            length = (N - start) // step + 1
            if length < 1:
                continue
            row = np.zeros(N, dtype=np.float64)
            for k in range(length):
                pos = start + k * step - 1
                if pos < N:
                    row[pos] = 1.0
            rows.append(row)
    return np.array(rows, dtype=np.float64)

# Precompute the incidence matrix at module load time
_INC = _build_incidence()
_NUM_APS = _INC.shape[0]

# Precompute list of AP indices for each position for faster updates
_APS_PER_POS = [np.nonzero(_INC[:, j] == 1.0)[0] for j in range(N)]

def _max_abs_sum(signs):
    """Compute max absolute signed sum over all APs.
    signs: array of shape (256,) with values -1 or +1.
    """
    sums = _INC @ signs
    return float(np.max(np.abs(sums)))

def _identify_critical_aps(sums, threshold):
    """Return indices of APs with |sum| >= threshold."""
    return np.nonzero(np.abs(sums) >= threshold)[0]

def _eval_critical(sums, critical_idx):
    """Evaluate max abs sum only on critical APs."""
    if len(critical_idx) == 0:
        return 0.0
    return float(np.max(np.abs(sums[critical_idx])))

def construct():
    t_start = time.time()
    deadline = 90.0 - DEADLINE_MARGIN

    # Default fallback: alternating signs
    default_signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])

    rng = np.random.default_rng(42)

    # Multi-restart local search
    best_signs = default_signs.copy()
    best_score = _max_abs_sum(best_signs)

    max_restarts = 30

    try:
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Initialize signs
            if restart == 0:
                signs = rng.choice([-1.0, 1.0], size=N)
            elif restart % 3 == 0:
                # Fresh random
                signs = rng.choice([-1.0, 1.0], size=N)
            else:
                # Perturb best known
                signs = best_signs.copy()
                num_flips = int(rng.integers(3, 15))
                flip_idx = rng.choice(N, size=num_flips, replace=False)
                signs[flip_idx] *= -1.0

            # Compute initial sums
            sums = _INC @ signs
            current_score = float(np.max(np.abs(sums)))

            # Identify critical APs (those near the max)
            # Use a threshold slightly below the current max to capture near-critical APs
            critical_threshold = current_score - 2.0
            critical_idx = _identify_critical_aps(sums, critical_threshold)

            # Local search: greedily flip positions that reduce the max abs sum on critical APs
            best_local_signs = signs.copy()
            best_local_score = current_score

            # Track which positions have been tried recently to avoid cycling
            stall_counter = 0
            max_stall = 5000

            while time.time() < deadline:
                if stall_counter > max_stall:
                    break

                # Periodically refresh critical APs (every ~2000 iterations)
                if stall_counter % 2000 == 0 and stall_counter > 0:
                    critical_threshold = best_local_score - 2.0
                    critical_idx = _identify_critical_aps(sums, critical_threshold)

                # Try flipping each position; pick the best improvement
                best_flip_j = -1
                best_flip_delta = 0.0

                # Sample a subset of positions to try (for speed)
                # Try all 256 positions but only evaluate on critical APs
                for j in range(N):
                    old_val = signs[j]
                    idxs = _APS_PER_POS[j]
                    if len(idxs) == 0:
                        continue

                    # Compute change in sums for critical APs that include j
                    # For each critical AP r that includes j, sum changes by -2*old_val
                    # We need to check which critical APs include position j
                    # Efficient: precompute for each j, which critical APs include j
                    # But critical_idx changes, so we do it on the fly
                    # For each j, find intersection of _APS_PER_POS[j] with critical_idx
                    j_aps = _APS_PER_POS[j]
                    # Find which of j_aps are in critical_idx
                    # Use set intersection for small sets
                    if len(critical_idx) < len(j_aps):
                        crit_set = set(critical_idx.tolist())
                        relevant = [r for r in j_aps if r in crit_set]
                    else:
                        crit_set = set(critical_idx.tolist())
                        relevant = [r for r in j_aps if r in crit_set]

                    if len(relevant) == 0:
                        continue

                    # Apply flip to relevant critical APs
                    old_sums = sums[relevant].copy()
                    sums[relevant] -= 2.0 * old_val

                    # Evaluate new max on critical APs
                    new_crit_max = float(np.max(np.abs(sums[critical_idx])))

                    # Undo flip
                    sums[relevant] = old_sums

                    delta = new_crit_max - best_local_score
                    if delta < best_flip_delta - 1e-10:
                        best_flip_delta = delta
                        best_flip_j = j

                if best_flip_j == -1:
                    # No improving flip found; try a random perturbation to escape
                    if rng.random() < 0.1:
                        # Random flip
                        j = int(rng.integers(0, N))
                        old_val = signs[j]
                        idxs = _APS_PER_POS[j]
                        if len(idxs) > 0:
                            sums[idxs] -= 2.0 * old_val
                        signs[j] = -old_val
                        # Re-evaluate
                        current_score = float(np.max(np.abs(sums)))
                        if current_score < best_local_score:
                            best_local_score = current_score
                            best_local_signs = signs.copy()
                            stall_counter = 0
                        else:
                            # Undo
                            signs[j] = old_val
                            if len(idxs) > 0:
                                sums[idxs] += 2.0 * old_val
                            stall_counter += 1
                    else:
                        stall_counter += 1
                        continue
                else:
                    # Apply the best flip
                    j = best_flip_j
                    old_val = signs[j]
                    idxs = _APS_PER_POS[j]
                    sums[idxs] -= 2.0 * old_val
                    signs[j] = -old_val

                    # Re-evaluate on critical APs
                    current_score = float(np.max(np.abs(sums)))
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_signs = signs.copy()
                        stall_counter = 0
                    else:
                        # This shouldn't happen if we computed correctly, but be safe
                        stall_counter += 1

            # End of local search for this restart
            if best_local_score < best_score:
                best_score = best_local_score
                best_signs = best_local_signs.copy()

    except Exception:
        pass

    # Convert to list of Python ints -1 or +1
    result = [int(s) for s in best_signs]
    return result

# EVOLVE-BLOCK-END
