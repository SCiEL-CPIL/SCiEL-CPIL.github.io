"""Optimized low-discrepancy on arithmetic progressions via SA with efficient deltas."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

# Precompute all APs as (start, step, length) with start in 1..N, step>=1, length>=1
# and start + (length-1)*step <= N
# Represent each AP as a list of 0-based indices for fast delta computation.
def _build_aps():
    aps = []
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            # max length: start + (L-1)*step <= N  => L <= (N - start)//step + 1
            max_len = (N - start) // step + 1
            for length in range(1, max_len + 1):
                # indices 0-based: start-1, start-1+step, ..., start-1+(length-1)*step
                idx = [start - 1 + k * step for k in range(length)]
                aps.append(idx)
    return aps

_APS = _build_aps()
_NUM_APS = len(_APS)

# Precompute, for each position i, the list of AP indices that contain i.
# pos_to_aps[i] = list of ap_index
def _build_pos_to_aps():
    m = _NUM_APS
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, idx in enumerate(_APS):
        for i in idx:
            pos_to_aps[i].append(ap_idx)
    return pos_to_aps

_POS_TO_APS = _build_pos_to_aps()

# Precompute AP sums as a matrix? That's _NUM_APS x N. _NUM_APS is large.
# Count: sum over step of sum over start of max_len.
# step=1: for start 1..N, max_len = N-start+1 => sum = N(N+1)/2 = 32896
# step=2: start 1..N, max_len = (N-start)//2+1
# This could be ~ O(N^2 log N) or more. Let's estimate total.
# For step s, number of APs = sum_{start=1}^{N} (floor((N-start)/s)+1)
# = sum_{k=0}^{N-1} (floor(k/s)+1) where k=N-start
# = (N) + sum_{k=0}^{N-1} floor(k/s)
# sum_{k=0}^{N-1} floor(k/s) ~ (N^2)/(2s)
# So total ~ sum_{s=1}^{N} [N + N^2/(2s)] ~ N^2 + (N^2/2) * H_N ~ 65536 + 32768 * ~6 = ~262k
# That's about 260k APs. A 260k x 256 float matrix = 66M floats = 528 MB. Too big.
# So we won't build a full matrix. Instead we compute sums on the fly but use
# pos_to_aps for efficient delta updates.

def _initial_sums(signs):
    """Compute sum for each AP given signs array (0-based, values -1/1)."""
    sums = np.zeros(_NUM_APS, dtype=np.int32)
    # For each position, add sign to all APs containing it.
    for i in range(N):
        s = signs[i]
        for ap_idx in _POS_TO_APS[i]:
            sums[ap_idx] += s
    return sums

def _max_abs(sums):
    return int(np.max(np.abs(sums)))

def _objective(signs):
    sums = _initial_sums(signs)
    return _max_abs(sums)

def _delta_of_flip(signs, i, sums):
    """Return (new_max, changed_sums_list) when flipping position i.
    We flip signs[i] -> -signs[i]. For each AP containing i, its sum changes by -2*signs[i].
    We update sums in place and compute new max abs.
    Returns list of (ap_idx, delta) that were applied, so we can undo.
    """
    old_sign = signs[i]
    delta = -2 * old_sign
    affected = _POS_TO_APS[i]
    # Track previous max abs quickly: we'll recompute max after update.
    # To be efficient, we update and compute max over affected + keep global? 
    # Simplest: update affected, then recompute max over all sums (260k) each eval is too slow.
    # Instead maintain max_abs incrementally: but max can decrease, so we need to recompute
    # when the current max element is affected and decreased.
    # Approach: keep current max_abs. After update, if any affected ap had |sum| == old_max
    # and its new abs < old_max, we may need to recompute. Otherwise new_max = max(old_max, max abs of affected).
    # We'll implement: compute new_abs for affected; candidate = max of new_abs over affected and old_max.
    # But if old_max came from an unaffected AP, it's still there. If it came from an affected
    # AP that decreased, we might overestimate. Overestimate is okay for SA acceptance? 
    # Actually we need exact objective for final, but for delta we can accept slight overestimate
    # as long as we recompute exactly when needed. Simpler and safe: recompute max over affected
    # and also keep a count of how many APs have abs == current max. If that count becomes 0,
    # recompute global max.
    # Given time constraints, let's just recompute global max each flip but in a vectorized way:
    # sums is int32 array of 260k; np.max(np.abs(sums)) each flip. 260k abs+max is ~ fast in numpy.
    # Let's test feasibility: 260k elements, abs+max ~ maybe 20-50 us? With 10k flips that's 0.2-0.5s. OK.
    for ap_idx in affected:
        sums[ap_idx] += delta
    new_max = int(np.max(np.abs(sums)))
    signs[i] = -old_sign
    return new_max

def _undo_flip(signs, i, sums, affected):
    """Undo a flip: flip signs[i] back and revert sums."""
    # signs[i] was already flipped; flip back
    signs[i] = -signs[i]
    # The delta applied was -2*old_sign where old_sign is the sign BEFORE the flip.
    # After flip, signs[i] = -old_sign. To undo, we add 2*signs[i] to affected? 
    # Let's recompute: original sign = s0. Flip: new sign = -s0. delta to sums = -2*s0.
    # To undo, we need +2*s0 = -2*(-s0) = -2*(new sign). new sign = signs[i] currently.
    # So add -2*signs[i] to each affected.
    undo_delta = -2 * signs[i]
    for ap_idx in affected:
        sums[ap_idx] += undo_delta

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    rng = random.Random(171)
    np_rng = np.random.default_rng(171)

    best_signs = None
    best_obj = 10**9

    # Multi-restart SA
    n_restarts = 200  # will be limited by time
    for restart in range(n_restarts):
        # Initialize: random signs
        signs = np.array([1 if np_rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums = _initial_sums(signs)
        cur_obj = _max_abs(sums)

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = signs.copy()

        # SA parameters
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        # Number of steps per restart: try to do as many as fit, but cap
        max_steps = 5000
        for step in range(max_steps):
            if time.time() >= deadline:
                break
            i = int(np_rng.integers(N))
            old_sign = signs[i]
            affected = _POS_TO_APS[i]
            delta = -2 * old_sign
            for ap_idx in affected:
                sums[ap_idx] += delta
            signs[i] = -old_sign
            new_obj = int(np.max(np.abs(sums)))

            d = new_obj - cur_obj
            if d <= 0 or (T > T_min and rng.random() < np.exp(-d / T)):
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = signs.copy()
            else:
                # revert
                signs[i] = old_sign
                undo_delta = -2 * old_sign
                for ap_idx in affected:
                    sums[ap_idx] += undo_delta

            T *= cooling
            if T < T_min:
                break

        if time.time() >= deadline:
            break

    if best_signs is None:
        # fallback
        best_signs = np.array([1 if np_rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)

    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
