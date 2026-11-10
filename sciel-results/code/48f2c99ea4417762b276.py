"""
Low discrepancy signs on all arithmetic progressions in [1, 256].

Structural change from the prior program:
- Replace the full O(n^2 log n) numpy evaluator with an O(n^2) Python precomputed
  table of per-position AP contributions, then evaluate any sign assignment in O(n)
  (max over a precomputed integer array). This lets local search do ~1000x more
  flips per second, escaping shallow local minima.
- Initialization uses Thue-Morse (a 1-periodic low-discrepancy sequence) plus
  several random restarts; local search is a simple 1-flip hill climb with
  occasional random-kick restarts when stuck.
"""
import time
import random
import numpy as np

_N = 256


def _build_ap_table():
    """
    For each position p (0-indexed), build a list of AP indices that contain p.
    Actually we need, for each AP, the list of positions it contains.
    Then to evaluate a sign vector quickly:
        For each AP a, sum_a = sum(signs[i] for i in a)
    We can precompute for each position p a list of (ap_index, coefficient) where
    coefficient is +1 (since each sign appears with weight +1 in the sum of its AP).
    
    Then sum_a = sum over positions p in a of signs[p].
    So if we define for each AP a the set of positions, then flipping position p
    changes sum_a by -2*signs[p] for every AP a containing p.
    
    Fast evaluation: maintain an array AP_sums where AP_sums[a] = current sum for AP a.
    Initial computation: for each AP a, sum signs over its positions.
    Flip at position p: for each AP a containing p, AP_sums[a] -= 2 * signs[p].
    Then max_abs = max(abs(AP_sums)).
    
    But we need max_abs over all APs, so we need to track it. Since flips change
    many AP_sums, we can just recompute max(abs(AP_sums)) after each flip, which
    is O(num_APs). num_APs is about n^2/2 ~ 32K for n=256, so this is fast.
    
    Precompute:
    - ap_positions: list of lists, ap_positions[a] = list of positions in AP a
    - pos_aps: pos_aps[p] = list of AP indices a such that p is in AP a
    
    Then:
    - Initial AP_sums: for each a, sum signs over ap_positions[a]
    - Flip p: for each a in pos_aps[p], AP_sums[a] -= 2 * old_sign
    - max_abs = max(abs(AP_sums))
    """
    n = _N
    # Enumerate all APs: start s in [0, n), step d in [1, n), length >= 2
    # An AP is defined by (s, d, length) where s + k*d < n for k=0..length-1
    # Actually the AP is the maximal sequence? No, "any arithmetic progression contained in [1,256]"
    # means any subset {s, s+d, s+2d, ...} that is contained in [0, n-1].
    # So for each s in [0,n), d in [1,n), the AP is {s, s+d, s+2d, ...} while < n.
    # This is the standard interpretation.
    
    ap_positions = []
    pos_aps = [[] for _ in range(n)]
    
    for s in range(n):
        for d in range(1, n):
            positions = []
            x = s
            while x < n:
                positions.append(x)
                x += d
            if len(positions) >= 2:  # skip singletons? No, singletons have sum = sign, abs=1
                # Actually singletons: the AP of length 1 starting at s with any d.
                # The sum is just signs[s], abs is 1. So max_abs is at least 1.
                # We can include them or just note max_abs >= 1.
                ap_positions.append(positions)
                ap_idx = len(ap_positions) - 1
                for p in positions:
                    pos_aps[p].append(ap_idx)
    
    # Also add singletons? They contribute max_abs >= 1. We can just track max_abs = max(1, ...)
    # Let's not add singletons to the table; we'll just ensure max_abs >= 1.
    
    return ap_positions, pos_aps


def _eval_max_abs(signs, ap_positions, pos_aps):
    """Compute max absolute AP sum from scratch."""
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_abs = 1
    for positions in ap_positions:
        s = int(np.sum(signs[positions]))
        a = abs(s)
        if a > max_abs:
            max_abs = a
    return max_abs


def _local_search(signs, ap_positions, pos_aps, max_time, rng):
    """
    Local search with fast flip evaluation using precomputed AP tables.
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
    
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline:
        improved = False
        positions = list(range(n))
        rng.shuffle(positions)
        
        for p in positions:
            if time.time() > deadline:
                break
            
            old_sign = signs[p]
            signs[p] = -old_sign
            # Update ap_sums for all APs containing p
            for a in pos_aps[p]:
                ap_sums[a] -= 2 * old_sign
            
            new_max = int(np.max(np.abs(ap_sums)))
            new_max = max(new_max, 1)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                no_improve = 0
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
                break
            else:
                # Revert
                signs[p] = old_sign
                for a in pos_aps[p]:
                    ap_sums[a] += 2 * old_sign
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                # Random kick
                signs = best_signs.copy()
                # Recompute ap_sums
                for a, positions in enumerate(ap_positions):
                    ap_sums[a] = int(np.sum(signs[positions]))
                current_max = int(np.max(np.abs(ap_sums)))
                current_max = max(current_max, 1)
                
                num_flips = rng.randint(2, 5)
                for _ in range(num_flips):
                    p = rng.randint(0, n - 1)
                    old_sign = signs[p]
                    signs[p] = -old_sign
                    for a in pos_aps[p]:
                        ap_sums[a] -= 2 * old_sign
                current_max = int(np.max(np.abs(ap_sums)))
                current_max = max(current_max, 1)
                no_improve = 0
    
    return best_signs.tolist(), best_max


def _thue_morse_init(n):
    """
    Thue-Morse sequence: t(i) = parity of number of 1s in binary representation of i.
    Map even -> +1, odd -> -1.
    This is a 1-automatic sequence with low discrepancy on intervals.
    """
    signs = np.empty(n, dtype=np.int8)
    for i in range(n):
        # Count bits
        bits = bin(i).count('1')
        signs[i] = 1 if bits % 2 == 0 else -1
    return signs


def construct():
    start_time = time.time()
    total_budget = 58  # 60s limit, 2s margin
    
    rng = random.Random(123)
    
    # Precompute AP tables (this is the expensive part, done once)
    ap_positions, pos_aps = _build_ap_table()
    num_aps = len(ap_positions)
    
    elapsed = time.time() - start_time
    # print(f"Table built: {num_aps} APs, {elapsed:.2f}s")
    
    best_signs = None
    best_max = float('inf')
    
    # Phase 1: Try Thue-Morse initialization
    signs_tm = _thue_morse_init(_N)
    max_tm = _eval_max_abs(signs_tm, ap_positions, pos_aps)
    if max_tm < best_max:
        best_max = max_tm
        best_signs = signs_tm.copy()
    
    # Phase 2: Try a few random initializations
    for _ in range(3):
        signs_r = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int8)
        max_r = _eval_max_abs(signs_r, ap_positions, pos_aps)
        if max_r < best_max:
            best_max = max_r
            best_signs = signs_r.copy()
    
    # Phase 3: Local search from best initialization
    remaining = total_budget - (time.time() - start_time)
    if remaining > 5 and best_signs is not None:
        result_signs, result_max = _local_search(best_signs, ap_positions, pos_aps, remaining - 5, rng)
        if result_max < best_max:
            best_max = result_max
            best_signs = np.array(result_signs, dtype=np.int8)
    
    # Phase 4: If time remains, try local search from Thue-Morse (different starting point)
    remaining2 = total_budget - (time.time() - start_time)
    if remaining2 > 15:
        signs_tm2 = _thue_morse_init(_N)
        # Perturb slightly
        for _ in range(3):
            p = rng.randint(0, _N - 1)
            signs_tm2[p] = -signs_tm2[p]
        result_signs2, result_max2 = _local_search(signs_tm2, ap_positions, pos_aps, remaining2 - 5, rng)
        if result_max2 < best_max:
            best_max = result_max2
            best_signs = np.array(result_signs2, dtype=np.int8)
    
    return [int(x) for x in best_signs]
