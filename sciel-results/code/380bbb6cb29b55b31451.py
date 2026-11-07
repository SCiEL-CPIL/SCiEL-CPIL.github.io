"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco(demands, capacity, H, n_ants=20, n_iter=30, seed=0):
    """Run the exact ACO protocol; return fewest bins used by any ant."""
    rng = np.random.default_rng(seed)
    n = len(demands)
    W = np.asarray(demands, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    # Clamp non-positive to 1e-9 (protocol)
    H = np.where(H > 0, H, 1e-9)
    tau = np.ones((n, n), dtype=np.float64)
    best_fewest = n  # worst case: each item its own bin
    for it in range(n_iter):
        iter_fewest = n
        for _ in range(n_ants):
            remaining = np.ones(n, dtype=bool)
            bins = 0
            cur = None
            while remaining.any():
                # fitting items: W[j] <= capacity if cur is None else W[j] <= capacity - W[cur]
                if cur is None:
                    fit = remaining & (W <= capacity)
                else:
                    rem_cap = capacity - W[cur]
                    fit = remaining & (W <= rem_cap)
                if not fit.any():
                    # open a new bin
                    bins += 1
                    cur = None
                    continue
                if cur is None:
                    # place a random fitting item
                    idx = np.nonzero(fit)[0]
                    pick = idx[rng.integers(len(idx))]
                    remaining[pick] = False
                    cur = pick
                    bins += 1
                else:
                    # prob proportional to tau[cur, j] * H[cur, j]
                    probs = tau[cur, :] * H[cur, :]
                    probs = np.where(fit, probs, 0.0)
                    total = probs.sum()
                    if total <= 0 or not np.isfinite(total):
                        idx = np.nonzero(fit)[0]
                        pick = idx[rng.integers(len(idx))]
                    else:
                        p = probs / total
                        pick = rng.choice(n, p=p)
                    remaining[pick] = False
                    cur = pick
            iter_fewest = min(iter_fewest, bins)
        best_fewest = min(best_fewest, iter_fewest)
        # pheromone update: multiply by 0.9, add 1/(fewest bins of that iteration)
        tau = tau * 0.9 + 1.0 / iter_fewest
    return best_fewest


def heuristics(demands, capacity):
    start = time.time()
    deadline = start + 290  # safety margin before 330s kill

    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    pair = W[:, None] + W[None, :]
    fit = pair <= capacity

    # Build candidate heuristic matrices
    cands = []

    # 1. Seed: pair/capacity where fits
    cands.append(np.where(fit, pair / capacity, 1e-9))

    # 2. Smallest gap: 1/(capacity - pair)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair
        h2 = np.where(fit, 1.0 / (gap + 1e-9), 1e-9)
    cands.append(h2)

    # 3. Best fit: target size
    cands.append(np.where(fit, W[None, :], 1e-9))

    # 4. Complement: capacity - pair (bigger when tighter fit)
    cands.append(np.where(fit, capacity - pair, 1e-9))

    # 5. Product of both sizes
    cands.append(np.where(fit, W[:, None] * W[None, :], 1e-9))

    # 6. Geometric mean of sizes
    cands.append(np.where(fit, np.sqrt(W[:, None] * W[None, :]), 1e-9))

    # 7. Inverse pair (prefer smaller combined)
    with np.errstate(divide='ignore', invalid='ignore'):
        h7 = np.where(fit, 1.0 / (pair + 1e-9), 1e-9)
    cands.append(h7)

    # 8. Combined: size * (1/gap)
    with np.errstate(divide='ignore', invalid='ignore'):
        h8 = np.where(fit, W[None, :] / (gap + 1e-9), 1e-9)
    cands.append(h8)

    # 9. Both sizes / gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h9 = np.where(fit, (W[:, None] * W[None, :]) / (gap + 1e-9), 1e-9)
    cands.append(h9)

    # 10. Log of pair
    with np.errstate(divide='ignore', invalid='ignore'):
        h10 = np.where(fit, np.log(pair + 1.0), 1e-9)
    cands.append(h10)

    # 11. Square of complement
    cands.append(np.where(fit, (capacity - pair) ** 2, 1e-9))

    # 12. Sqrt of complement
    cands.append(np.where(fit, np.sqrt(capacity - pair), 1e-9))

    # 13. Ratio: target / source
    with np.errstate(divide='ignore', invalid='ignore'):
        h13 = np.where(fit, W[None, :] / (W[:, None] + 1e-9), 1e-9)
    cands.append(h13)

    # 14. 1 - pair/capacity (normalized complement)
    cands.append(np.where(fit, 1.0 - pair / capacity, 1e-9))

    # 15. pair^2 / capacity^2
    cands.append(np.where(fit, (pair / capacity) ** 2, 1e-9))

    # Evaluate each candidate with the actual ACO on this instance
    best_H = cands[0]
    best_score = float('inf')

    # Use a fixed seed for evaluation (deterministic tuning)
    eval_seed = 42

    for idx, H in enumerate(cands):
        if time.time() > deadline - 5:
            break
        try:
            score = _run_aco(W, capacity, H, n_ants=20, n_iter=30, seed=eval_seed)
            if score < best_score:
                best_score = score
                best_H = H
        except Exception:
            continue

    # If we have spare time, try a few more randomized combinations
    if time.time() < deadline - 30:
        rng = np.random.default_rng(123)
        base = [cands[0], cands[1], cands[2], cands[3]]
        for trial in range(15):
            if time.time() > deadline - 5:
                break
            # Random weighted combination of a few bases
            k = rng.integers(2, 4)
            chosen = rng.choice(len(base), size=k, replace=False)
            weights = rng.dirichlet(np.ones(k))
            Hcomb = np.zeros((n, n), dtype=np.float64)
            for c, w in zip(chosen, weights):
                Hcomb += w * base[c]
            Hcomb = np.where(fit, Hcomb, 1e-9)
            try:
                score = _run_aco(W, capacity, Hcomb, n_ants=20, n_iter=30, seed=eval_seed)
                if score < best_score:
                    best_score = score
                    best_H = Hcomb
            except Exception:
                continue

    # Final safety: ensure shape and finiteness
    best_H = np.asarray(best_H, dtype=np.float64)
    if best_H.shape != (n, n):
        best_H = cands[0]
    if not np.all(np.isfinite(best_H)):
        best_H = cands[0]
    best_H = np.where(best_H > 0, best_H, 1e-9)

    return best_H
# EVOLVE-BLOCK-END
