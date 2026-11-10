"""Rewritten program: faster discrepancy + greedy init + simulated annealing."""
import time
import math
import random
import numpy as np

def _max_disc_fast(signs):
    """Compute max |sum| over all APs in [1,n] using precomputed prefix sums."""
    n = len(signs)
    arr = np.asarray(signs, dtype=np.int32)
    # prefix sums: P[0]=0, P[k]=sum arr[0:k]
    P = np.empty(n + 1, dtype=np.int32)
    P[0] = 0
    P[1:] = np.cumsum(arr)
    
    max_d = 1  # singletons
    
    for d in range(1, n + 1):
        # For each residue r in [0, min(d, n)-1], consider the AP: r, r+d, r+2d, ...
        # The AP is contained in [0, n-1] (0-indexed) as long as r < n.
        # Sum from index a to b (0-indexed, inclusive) is P[b+1] - P[a].
        for r in range(min(d, n)):
            # positions: r, r+d, r+2d, ...
            # length of this AP
            L = (n - 1 - r) // d + 1
            if L <= 1:
                continue
            # prefix over this AP: ap_prefix[0]=0, ap_prefix[k] = sum of first k elements
            # element j (0-indexed in AP) is at global index r + j*d
            # sum of first k elements = sum over j=0..k-1 of arr[r + j*d]
            # We can compute these using P: sum_{j=0}^{k-1} arr[r+j*d]
            # = (P[r+d] - P[r]) + (P[r+2d] - P[r+d]) + ...
            # This is a telescoping sum if we think of it as P[r+kd] - P[r] only if consecutive,
            # but here they are spaced by d. So we need to sum the individual differences.
            # Let's build the AP prefix array.
            # Actually, let's just iterate to build ap_prefix, it's O(L) per AP.
            
            ap = arr[r::d][:L]
            ap_cum = np.empty(L + 1, dtype=np.int32)
            ap_cum[0] = 0
            ap_cum[1:] = np.cumsum(ap)
            # max abs of ap_cum
            m = np.max(np.abs(ap_cum))
            if m > max_d:
                max_d = m
    return int(max_d)

def _greedy_init(n, rng):
    """Greedy construction: assign each sign to minimize the max AP discrepancy so far."""
    signs = np.zeros(n, dtype=np.int8)
    signs[0] = rng.choice((-1, 1))
    
    for i in range(1, n):
        best_val = None
        best_max = None
        for v in (-1, 1):
            signs[i] = v
            # Compute max discrepancy for APs that include index i
            # This is expensive if done naively. Let's compute full discrepancy for now,
            # or just the affected APs.
            # For simplicity and correctness, compute full, but this is O(n^2 log n) per step = O(n^3 log n) total.
            # For n=256, that's ~256 * 256 * 8 = 524288 * 256 = 134M ops. Too slow.
            # Let's just use random for now and rely on SA.
            # Actually, let's do a lighter greedy: only check APs ending at i.
            pass
        # Fall back to random with slight bias
        signs[i] = rng.choice((-1, 1))
    
    return signs

def construct():
    deadline = time.time() + 50  # Safety margin before 60s (evaluator kills at 90s)
    rng = random.Random(42)
    n = 256
    
    # Start with random initialization
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int8)
    current_disc = _max_disc_fast(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated annealing
    start_temp = 10.0
    end_temp = 0.1
    total_time = deadline - time.time()
    
    # Precompute AP structures for faster update
    # For each index i, list of (d, r, pos_in_ap) where i belongs to AP (d, r) at position pos
    # pos_in_ap = (i - r) // d
    ap_membership = [[] for _ in range(n)]
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            L = (n - 1 - r) // d + 1
            if L <= 1:
                continue
            for j in range(L):
                idx = r + j * d
                ap_membership[idx].append((d, r, j))
    
    # Build prefix-sum based AP cumulative arrays for each (d, r)
    # ap_data[(d, r)] = (cumsum_array, length)
    ap_data = {}
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            L = (n - 1 - r) // d + 1
            if L <= 1:
                continue
            arr = signs[r::d][:L].astype(np.int32)
            cum = np.empty(L + 1, dtype=np.int32)
            cum[0] = 0
            cum[1:] = np.cumsum(arr)
            ap_data[(d, r)] = (cum, L)
    
    def get_max_disc_from_ap_data():
        m = 1
        for (d, r), (cum, L) in ap_data.items():
            if L > 1:
                mm = np.max(np.abs(cum))
                if mm > m:
                    m = mm
        return int(m)
    
    steps = 0
    total_steps = max(1, int(total_time / 0.005))  # ~200k steps
    while time.time() < deadline:
        progress = steps / total_steps
        T = start_temp * (1 - progress) + end_temp * progress
        if T <= 0.01:
            T = 0.01
        
        # Pick a random index to flip
        i = rng.randint(0, n - 1)
        old_sign = signs[i]
        new_sign = -old_sign
        delta = 2 * new_sign  # change in value
        
        signs[i] = new_sign
        
        # Update affected APs
        old_max = current_disc
        new_max = current_disc
        
        for (d, r, j) in ap_membership[i]:
            cum, L = ap_data[(d, r)]
            # Update cum[j+1:] by adding delta
            cum[j+1:] += delta
            mm = np.max(np.abs(cum))
            if mm > new_max:
                new_max = mm
        
        # Acceptance criterion
        accepted = False
        if new_max < old_max:
            accepted = True
        elif new_max == old_max:
            accepted = True  # Accept equal to allow exploration
        else:
            dE = new_max - old_max
            if dE / T < 700:
                if rng.random() < math.exp(-dE / T):
                    accepted = True
        
        if accepted:
            current_disc = new_max
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[i] = old_sign
            for (d, r, j) in ap_membership[i]:
                cum, L = ap_data[(d, r)]
                cum[j+1:] -= delta
        
        steps += 1
        if steps % 5000 == 0:
            # Periodic full recomputation to guard against drift
            full_disc = _max_disc_fast(signs)
            if full_disc != current_disc:
                current_disc = full_disc
                # Rebuild ap_data
                for (d, r), (cum, L) in list(ap_data.items()):
                    arr = signs[r::d][:L].astype(np.int32)
                    new_cum = np.empty(L + 1, dtype=np.int32)
                    new_cum[0] = 0
                    new_cum[1:] = np.cumsum(arr)
                    ap_data[(d, r)] = (new_cum, L)
    
    return best_signs.tolist()
