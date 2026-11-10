"""
Low discrepancy signs on all arithmetic progressions in [1, 256].

Structural change: Simulated annealing with a fast incremental AP-sum update.
Instead of pure hill-climbing (which gets stuck in local minima), we use
simulated annealing to accept worsening moves with a temperature-dependent
probability, allowing escape from local minima. The AP-sum table is updated
incrementally on each flip (O(num_APs_containing_p) per flip), and we track
the max absolute sum efficiently.

Key improvements:
1. Simulated annealing instead of pure hill-climbing - accepts bad moves early
2. Multiple restarts with different initializations (Thue-Morse, random, greedy)
3. Efficient max tracking: maintain a sorted structure or just recompute max
   (num_APs ~ 16K, so max over 16K ints is fast in numpy)
4. Greedy initialization that builds signs one at a time minimizing worst-case
"""
import time
import random
import numpy as np

_N = 256


def _build_ap_table():
    """
    Precompute all arithmetic progressions and their position lists.
    Also build pos_aps: for each position, which APs contain it.
    """
    n = _N
    ap_positions = []
    pos_aps = [[] for _ in range(n)]
    
    for s in range(n):
        for d in range(1, n):
            positions = []
            x = s
            while x < n:
                positions.append(x)
                x += d
            if len(positions) >= 2:
                ap_idx = len(ap_positions)
                ap_positions.append(positions)
                for p in positions:
                    pos_aps[p].append(ap_idx)
    
    return ap_positions, pos_aps


def _thue_morse_init(n):
    """Thue-Morse sequence: low discrepancy on intervals."""
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        bits = bin(i).count('1')
        signs[i] = 1 if bits % 2 == 0 else -1
    return signs


def _greedy_init(ap_positions, pos_aps, rng):
    """
    Greedy construction: assign signs one position at a time.
    For each position, try both +1 and -1, pick the one that minimizes
    the maximum AP sum considering already-assigned positions.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int8)
    assigned = np.zeros(n, dtype=bool)
    
    # We'll assign in random order to break symmetry
    order = list(range(n))
    rng.shuffle(order)
    
    # Track current AP sums (only counting assigned positions)
    ap_sums = np.zeros(len(ap_positions), dtype=np.int32)
    
    for p in order:
        # Try +1
        delta_plus = np.zeros(len(ap_positions), dtype=np.int32)
        for a in pos_aps[p]:
            delta_plus[a] = 1
        candidate_plus = ap_sums + delta_plus
        max_plus = int(np.max(np.abs(candidate_plus)))
        
        # Try -1
        delta_minus = np.zeros(len(ap_positions), dtype=np.int32)
        for a in pos_aps[p]:
            delta_minus[a] = -1
        candidate_minus = ap_sums + delta_minus
        max_minus = int(np.max(np.abs(candidate_minus)))
        
        if max_plus <= max_minus:
            signs[p] = 1
            ap_sums += delta_plus
        else:
            signs[p] = -1
            ap_sums += delta_minus
        
        assigned[p] = True
    
    return signs


def _simulated_annealing(signs, ap_positions, pos_aps, max_time, rng, T0=10.0, T1=0.1, cooling=0.9995):
    """
    Simulated annealing with incremental AP-sum updates.
    """
    n = len(signs)
    signs = np.array(signs, dtype=np.int8)
    
    # Initialize AP_sums
    ap_sums = np.zeros(len(ap_positions), dtype=np.int32)
    for a, positions in enumerate(ap_positions):
        ap_sums[a] = int(np.sum(signs[positions]))
    
    current_max = int(np.max(np.abs(ap_sums)))
    current_max = max(current_max, 1)
    
    best_max = current_max
    best_signs = signs.copy()
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = T0
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        p = rng.randint(0, n - 1)
        old_sign = signs[p]
        new_sign = -old_sign
        
        # Apply flip
        signs[p] = new_sign
        for a in pos_aps[p]:
            ap_sums[a] -= 2 * old_sign
        
        new_max = int(np.max(np.abs(ap_sums)))
        new_max = max(new_max, 1)
        
        delta = new_max - current_max
        
        if delta <= 0:
            # Accept (improvement or equal)
            current_max = new_max
            no_improve = 0
            if current_max < best_max:
                best_max = current_max
                best_signs = signs.copy()
        else:
            # Worsening: accept with probability exp(-delta / T)
            if T > 1e-10:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_max = new_max
                    no_improve += 1
                else:
                    # Revert
                    signs[p] = old_sign
                    for a in pos_aps[p]:
                        ap_sums[a] += 2 * old_sign
                    no_improve += 1
            else:
                # Revert
                signs[p] = old_sign
                for a in pos_aps[p]:
                    ap_sums[a] += 2 * old_sign
                no_improve += 1
        
        T *= cooling
        
        if no_improve >= max_no_improve:
            # Reset temperature occasionally to escape deep local minima
            if rng.random() < 0.3:
                T = max(T, 5.0)
                no_improve = 0
            else:
                # Restart from best
                signs = best_signs.copy()
                for a, positions in enumerate(ap_positions):
                    ap_sums[a] = int(np.sum(signs[positions]))
                current_max = int(np.max(np.abs(ap_sums)))
                current_max = max(current_max, 1)
                T = T0 * 0.5
                no_improve = 0
    
    return best_signs, best_max


def construct():
    start_time = time.time()
    total_budget = 55  # safety margin for 60s limit
    
    rng = random.Random(42)
    
    # Precompute AP tables
    ap_positions, pos_aps = _build_ap_table()
    num_aps = len(ap_positions)
    
    elapsed = time.time() - start_time
    
    best_signs = None
    best_max = float('inf')
    
    # Phase 1: Try multiple initializations
    inits = []
    
    # Thue-Morse
    signs_tm = _thue_morse_init(_N)
    inits.append(signs_tm)
    
    # Random
    for _ in range(3):
        signs_r = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
        inits.append(signs_r)
    
    # Greedy
    signs_g = _greedy_init(ap_positions, pos_aps, rng)
    inits.append(signs_g)
    
    # Evaluate all initializations
    for signs in inits:
        ap_sums = np.zeros(num_aps, dtype=np.int32)
        for a, positions in enumerate(ap_positions):
            ap_sums[a] = int(np.sum(signs[positions]))
        m = int(np.max(np.abs(ap_sums)))
        m = max(m, 1)
        if m < best_max:
            best_max = m
            best_signs = signs.copy()
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    if remaining > 10 and best_signs is not None:
        # Phase 2: Simulated annealing from best init
        result_signs, result_max = _simulated_annealing(
            best_signs, ap_positions, pos_aps, remaining - 8, rng,
            T0=8.0, T1=0.1, cooling=0.9997
        )
        if result_max < best_max:
            best_max = result_max
            best_signs = result_signs
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    if remaining > 15:
        # Phase 3: SA from Thue-Morse specifically (different trajectory)
        signs_tm2 = _thue_morse_init(_N)
        # Small perturbation
        for _ in range(rng.randint(1, 4)):
            p = rng.randint(0, _N - 1)
            signs_tm2[p] = -signs_tm2[p]
        
        result_signs2, result_max2 = _simulated_annealing(
            signs_tm2, ap_positions, pos_aps, remaining - 8, rng,
            T0=6.0, T1=0.1, cooling=0.9998
        )
        if result_max2 < best_max:
            best_max = result_max2
            best_signs = result_signs2
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    if remaining > 12:
        # Phase 4: SA from greedy init
        signs_g2 = _greedy_init(ap_positions, pos_aps, rng)
        result_signs3, result_max3 = _simulated_annealing(
            signs_g2, ap_positions, pos_aps, remaining - 8, rng,
            T0=5.0, T1=0.1, cooling=0.9998
        )
        if result_max3 < best_max:
            best_max = result_max3
            best_signs = result_signs3
    
    return [int(x) for x in best_signs]
