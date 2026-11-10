"""Rewritten program: algebraic initialization + vectorized local search."""
import time
import numpy as np

N = 256

def _legendre_init():
    """Use Legendre symbol mod 257 (prime) as initial signs."""
    p = 257
    # For i in 0..256, compute Legendre symbol (i/p)
    # For i=0, define as 0 but we need +/-1, so map 0 to +1
    signs = np.zeros(N, dtype=np.int64)
    for i in range(N):
        # i corresponds to position i (0-indexed), value i+1 in 1..256
        x = (i + 1) % p
        if x == 0:
            signs[i] = 1
        else:
            # Euler's criterion: x^((p-1)/2) mod p
            ls = pow(x, (p - 1) // 2, p)
            if ls == 1:
                signs[i] = 1
            else:  # ls == p-1
                signs[i] = -1
    return signs

def _max_abs_ap_sum(signs):
    """Compute max |signed sum| over all arithmetic progressions in [0, n-1]."""
    n = len(signs)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            idx = np.arange(r, n, d)
            if len(idx) < 2:
                continue
            csum = np.cumsum(signs[idx])
            m = int(np.max(np.abs(csum)))
            if m > max_disc:
                max_disc = m
    return max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s

    # Precompute all APs: (s, d, t) where t >= 2
    all_aps = []
    for d in range(1, N):
        for s in range(d):
            L = (N - 1 - s) // d + 1
            for t in range(2, L + 1):
                all_aps.append((s, d, t))
    
    num_aps = len(all_aps)
    
    # Build reverse map: for each position f, which AP indices contain it
    ap_by_pos = [[] for _ in range(N)]
    for i, (s, d, t) in enumerate(all_aps):
        for j in range(t):
            pos = s + j * d
            ap_by_pos[pos].append(i)
    
    # Convert ap_by_pos to numpy arrays for faster access
    ap_by_pos_np = [np.array(lst, dtype=np.int64) if lst else np.array([], dtype=np.int64) 
                    for lst in ap_by_pos]
    
    # Precompute for each AP: start index s, step d, length t
    ap_starts = np.array([ap[0] for ap in all_aps], dtype=np.int64)
    ap_steps = np.array([ap[1] for ap in all_aps], dtype=np.int64)
    ap_lengths = np.array([ap[2] for ap in all_aps], dtype=np.int64)
    
    def _init_ap_sums(signs):
        """Compute sum for each AP."""
        sums = np.zeros(num_aps, dtype=np.int64)
        for i in range(num_aps):
            s, d, t = all_aps[i]
            seq = signs[s:s + t * d: d]
            sums[i] = int(np.sum(seq))
        return sums
    
    def _get_max(ap_sums):
        return int(np.max(np.abs(ap_sums)))
    
    best_disc = N  # worst case
    best_signs = None
    
    # Multi-start local search
    rng = np.random.default_rng(42)
    
    # Start 1: Legendre symbol initialization
    signs = _legendre_init()
    ap_sums = _init_ap_sums(signs)
    cur_max = _get_max(ap_sums)
    
    if cur_max < best_disc:
        best_disc = cur_max
        best_signs = signs.copy()
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with perturbation
            for _ in range(16):
                f = int(rng.integers(0, N))
                old_sign = signs[f]
                signs[f] = -old_sign
                ap_indices = ap_by_pos_np[f]
                if len(ap_indices) > 0:
                    ap_sums[ap_indices] -= 2 * old_sign
            no_improve = 0
            cur_max = _get_max(ap_sums)
            if cur_max < best_disc:
                best_disc = cur_max
                best_signs = signs.copy()
            continue
        
        # Try flipping a random position
        f = int(rng.integers(0, N))
        old_sign = signs[f]
        ap_indices = ap_by_pos_np[f]
        
        if len(ap_indices) == 0:
            no_improve += 1
            continue
        
        # Apply flip
        signs[f] = -old_sign
        ap_sums[ap_indices] -= 2 * old_sign
        new_max = _get_max(ap_sums)
        
        if new_max <= cur_max:
            cur_max = new_max
            no_improve = 0
            if new_max < best_disc:
                best_disc = new_max
                best_signs = signs.copy()
        else:
            # Revert
            signs[f] = old_sign
            ap_sums[ap_indices] += 2 * old_sign
            no_improve += 1
    
    # Return best_signs as list of Python integers
    result = []
    for v in best_signs:
        result.append(int(v))
    return result
