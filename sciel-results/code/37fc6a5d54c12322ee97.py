"""Low discrepancy on all arithmetic progressions - local search design."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
# Precompute all (start, step) pairs; index is 0-based, value is 1-based position
PAIRS = [(s, d) for s in range(1, N + 1) for d in range(1, N + 1) if s + d - 1 <= N]
PAIRS = np.array(PAIRS, dtype=np.int64)

def _max_abs_ap_sum(x):
    """Return the maximum absolute signed sum over any arithmetic progression."""
    # x: int8 array of length 256 with values in {-1, 1}
    starts = PAIRS[:, 0]
    steps = PAIRS[:, 1]
    # For each (s,d): positions are s, s+d, s+2d, ... (1-based)
    # max k such that s + k*d <= N  ->  k <= (N - s)//d
    maxk = (N - starts) // steps
    # We need sum of x[s-1 + k*d] for k=0..maxk-1
    # Use numpy: for each pair, take slice x[s-1 : s-1 + maxk*d : d]
    # Vectorized approach: build indices
    # For efficiency, process in chunks by step size
    best = 0
    for d in range(1, N + 1):
        # starts where s + d - 1 <= N  -> s <= N - d + 1
        s_arr = np.arange(1, N - d + 2)
        # length of progression from s: (N - s)//d + 1
        lens = (N - s_arr) // d + 1
        # For each start s, sum x[s-1 : s-1 + lens*d : d]
        # Vectorize: create index matrix
        # idx[i, j] = s_arr[i]-1 + j*d  for j in range(lens[i])
        # This is variable length; use a padded approach
        max_len = lens.max()
        idx = (s_arr[:, None] - 1) + np.arange(max_len)[None, :] * d
        # Mask where j < lens[i]
        mask = np.arange(max_len)[None, :] < lens[:, None]
        vals = x[idx]
        vals = vals * mask
        sums = vals.sum(axis=1)
        abs_sums = np.abs(sums)
        m = abs_sums.max()
        if m > best:
            best = int(m)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin below 90s
    rng = random.Random(20240521)

    # Initial: a few random restarts, pick best
    best_x = None
    best_val = 10**9
    for r in range(8):
        x = np.array([rng.choice((-1, 1)) for _ in range(N)], dtype=np.int8)
        v = _max_abs_ap_sum(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        if time.time() > deadline - 30.0:
            break

    if best_x is None:
        # Fallback: all +1 (won't happen given loop above)
        best_x = np.ones(N, dtype=np.int8)
        best_val = _max_abs_ap_sum(best_x)

    # Local search: 1-flip and 2-flip moves
    cur_x = best_x
    cur_val = best_val

    # Precompute for fast single-flip evaluation:
    # For a flip at position p, the change in AP sum for progression (s,d)
    # is -2*x[p] if p is in that progression, else 0.
    # We want to minimize max abs sum. A good heuristic: target the worst APs.

    def _get_worst_aps(x, k=40):
        """Return up to k (abs_sum, s, d) for the worst APs, sorted by abs_sum desc."""
        results = []
        for d in range(1, N + 1):
            s_arr = np.arange(1, N - d + 2)
            lens = (N - s_arr) // d + 1
            max_len = lens.max()
            idx = (s_arr[:, None] - 1) + np.arange(max_len)[None, :] * d
            mask = np.arange(max_len)[None, :] < lens[:, None]
            vals = x[idx] * mask
            sums = vals.sum(axis=1)
            abs_sums = np.abs(sums)
            # top few for this d
            top = abs_sums.argsort()[::-1][:max(1, k // (N // d + 1))]
            for i in top:
                results.append((int(abs_sums[i]), int(s_arr[i]), d))
        results.sort(key=lambda t: -t[0])
        return results[:k]

    stall = 0
    max_stall = 2000
    while time.time() < deadline - 5.0 and stall < max_stall:
        # Get worst APs to guide the search
        worst = _get_worst_aps(cur_x, k=30)

        # 1-flip moves: try flipping positions that appear in many worst APs
        # Count how many worst APs contain each position
        pos_count = np.zeros(N, dtype=np.int32)
        for _, s, d in worst:
            for p in range(s - 1, N, d):
                pos_count[p] += 1

        # Try flipping top candidate positions
        candidates = np.argsort(pos_count)[::-1][:30]
        improved = False
        for p in candidates:
            if time.time() > deadline - 5.0:
                break
            new_x = cur_x.copy()
            new_x[p] = -new_x[p]
            v = _max_abs_ap_sum(new_x)
            if v < cur_val:
                cur_x = new_x
                cur_val = v
                improved = True
                break
            # Also accept equal (plateau walk) with some probability
            elif v == cur_val and rng.random() < 0.1:
                cur_x = new_x
                improved = True
                break

        # 2-flip moves
        if not improved and time.time() < deadline - 5.0:
            # Pick two positions from top candidates
            cands = list(candidates[:15])
            rng.shuffle(cands)
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    if time.time() > deadline - 5.0:
                        break
                    p1, p2 = cands[i], cands[j]
                    new_x = cur_x.copy()
                    new_x[p1] = -new_x[p1]
                    new_x[p2] = -new_x[p2]
                    v = _max_abs_ap_sum(new_x)
                    if v < cur_val:
                        cur_x = new_x
                        cur_val = v
                        improved = True
                        break
                if improved:
                    break

        if improved:
            stall = 0
            # Update global best
            if cur_val < best_val:
                best_val = cur_val
                best_x = cur_x.copy()
        else:
            stall += 1

        # Periodic random restart if stuck too long
        if stall > 500 and time.time() < deadline - 20.0:
            # Try a random restart seeded near current
            new_x = cur_x.copy()
            for _ in range(rng.randint(2, 8)):
                p = rng.randint(0, N - 1)
                new_x[p] = -new_x[p]
            v = _max_abs_ap_sum(new_x)
            if v <= cur_val:
                cur_x = new_x
                cur_val = v
            stall = 0

    # Final verification
    final_val = _max_abs_ap_sum(best_x)
    return [int(v) for v in best_x]
# EVOLVE-BLOCK-END
