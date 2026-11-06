import random
import time
import numpy as np

def _sum_counts_vec(mask, N):
    """Number of representations a+b (a<=b) for each sum, using a numpy matrix."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.zeros(2 * N + 1, dtype=np.int32)
    A = idx[:, None]
    B = idx[None, :]
    sums = A + B
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    np.add.at(cnt, sums, 1)
    return cnt

def _can_add_vec(mask, cnt, N):
    """Vectorized check of which candidate elements can be added without creating a count>2."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.ones(N + 1, dtype=bool)
    cand = np.nonzero(~mask)[0]
    if cand.size == 0:
        return np.zeros(N + 1, dtype=bool)
    C = cand[:, None]
    sums = C + idx[None, :]
    new = cnt[sums] + 1
    bad = (new > 2).any(axis=1)
    out = np.zeros(N + 1, dtype=bool)
    out[cand[~bad]] = True
    return out

def _add_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] += 1
    if idx.size:
        cnt[idx + x] += 1
    mask[x] = True

def _remove_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] -= 1
    if idx.size:
        other = idx[idx != x]
        cnt[other + x] -= 1
    mask[x] = False

def _saturating_greedy(N, rng, skip_p=0.0):
    mask = np.zeros(N + 1, dtype=bool)
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if mask[x]:
            continue
        # Vectorized feasibility for this x
        idx = np.nonzero(mask)[0]
        ok = True
        if cnt[2 * x] == 2:
            ok = False
        else:
            if idx.size:
                s = idx + x
                if (cnt[s] == 2).any():
                    ok = False
        if ok:
            if rng.random() < skip_p:
                continue  # skip-noise to diversify seeds
            idx = np.nonzero(mask)[0]
            cnt[2 * x] += 1
            if idx.size:
                cnt[idx + x] += 1
            mask[x] = True
    return mask, cnt

def _conflict_score_bitset(mask, cnt, N, cand):
    """Score candidates by how many sums they push from count 1 to count 2 (blocking).
    Uses bitset-style vectorized ops for speed."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.zeros(cand.size, dtype=np.int32)
    sums = cand[:, None] + idx[None, :]
    # Count how many sums have current count exactly 1 (will become 2 after adding)
    blocking = (cnt[sums] == 1).sum(axis=1)
    # Also penalize sums that are already at count 2 (would create violations)
    overfull = (cnt[sums] == 2).sum(axis=1)
    return blocking.astype(np.int32) + 10 * overfull.astype(np.int32)

def _local_search(mask, cnt, N, rng, time_limit, best_mask, best_size, lambda_pen=10.0):
    """Local search with conflict-graph-aware scoring for insert/swap/double-insert moves."""
    cur_size = int(mask.sum())
    best_size_local = best_size
    best_mask_local = best_mask.copy()

    T = 3.0
    T_min = 0.1
    cooling = 0.9999
    stuck_count = 0
    max_stuck = 400

    def violation_of(cnt):
        return int(np.maximum(0, cnt - 2).sum())

    cur_viol = violation_of(cnt)
    iterations = 0

    while time.time() < time_limit:
        r = rng.random()
        improved = False
        iterations += 1

        if r < 0.55:
            # INSERT: pick feasible candidate with minimum conflict score
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if feas.any():
                cand = np.nonzero(feas)[0]
                if cand.size > 1:
                    # Use conflict-graph scoring: minimize blocking + overfull
                    scores = _conflict_score_bitset(mask, cnt, N, cand)
                    min_score = scores.min()
                    best_cands = cand[scores == min_score]
                    # Among tied, prefer smaller values (denser packing near bottom)
                    x = int(best_cands.min()) if rng.random() < 0.3 else int(rng.choice(best_cands))
                else:
                    x = int(cand[0])
                
                _add_vec(mask, cnt, x, N)
                cur_size += 1
                cur_viol = violation_of(cnt)
                if cur_viol == 0 and cur_size > best_size_local:
                    best_size_local = cur_size
                    best_mask_local = mask.copy()
                    improved = True
                elif cur_viol > 0:
                    _remove_vec(mask, cnt, x, N)
                    cur_size -= 1
                    cur_viol = violation_of(cnt)
        elif r < 0.75:
            # SMART SWAP: remove element with highest "blocking degree", add best candidate
            in_idx = np.nonzero(mask)[0]
            if in_idx.size == 0:
                continue
            # Score each in-set element by how much it blocks others
            # An element x blocks candidate y if x+y has cnt >= 1
            if in_idx.size > 1:
                # Approximate: elements that participate in many count-2 sums are more "valuable"
                # to keep; elements in few sums are better removal candidates
                elem_sum_counts = np.zeros(N + 1, dtype=np.int32)
                for x in in_idx:
                    others = in_idx[in_idx != x]
                    if others.size:
                        ss = others + x
                        elem_sum_counts[x] += (cnt[ss] >= 1).sum()
                # Remove element with lowest participation
                x = int(in_idx[np.argmin(elem_sum_counts[in_idx])])
            else:
                x = int(in_idx[0])
            
            out_idx = np.nonzero(~mask)[0]
            if out_idx.size == 0:
                continue
            _remove_vec(mask, cnt, x, N)
            # Now find best candidate to add
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if feas.any():
                cand = np.nonzero(feas)[0]
                if cand.size > 1:
                    scores = _conflict_score_bitset(mask, cnt, N, cand)
                    min_score = scores.min()
                    best_cands = cand[scores == min_score]
                    y = int(best_cands.min()) if rng.random() < 0.3 else int(rng.choice(best_cands))
                else:
                    y = int(cand[0])
            else:
                # No feasible add, revert removal
                _add_vec(mask, cnt, x, N)
                continue
            
            _add_vec(mask, cnt, y, N)
            new_viol = violation_of(cnt)
            delta_E = lambda_pen * (new_viol - cur_viol)
            accept = (delta_E < 0) or (rng.random() < np.exp(-delta_E / max(T, 1e-9)))
            if accept:
                cur_viol = new_viol
                if cur_viol == 0 and cur_size > best_size_local:
                    best_size_local = cur_size
                    best_mask_local = mask.copy()
                    improved = True
            else:
                _remove_vec(mask, cnt, y, N)
                _add_vec(mask, cnt, x, N)
                cur_viol = violation_of(cnt)
        elif r < 0.90:
            # DOUBLE INSERT with conflict-graph-aware pairing
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if np.sum(feas) >= 2:
                cand = np.nonzero(feas)[0]
                scores = _conflict_score_bitset(mask, cnt, N, cand)
                # Pick top-2 by score
                order = np.argsort(scores)
                x = int(cand[order[0]])
                _add_vec(mask, cnt, x, N)
                feas2 = _can_add_vec(mask, cnt, N)
                feas2 = feas2 & ~mask
                if feas2.any():
                    cand2 = np.nonzero(feas2)[0]
                    # Exclude x
                    cand2 = cand2[cand2 != x]
                    if cand2.size >= 1:
                        scores2 = _conflict_score_bitset(mask, cnt, N, cand2)
                        y = int(cand2[np.argmin(scores2)])
                        _add_vec(mask, cnt, y, N)
                        cur_size += 2
                        cur_viol = violation_of(cnt)
                        if cur_viol == 0 and cur_size > best_size_local:
                            best_size_local = cur_size
                            best_mask_local = mask.copy()
                            improved = True
                        else:
                            _remove_vec(mask, cnt, y, N)
                            _remove_vec(mask, cnt, x, N)
                            cur_size -= 2
                            cur_viol = violation_of(cnt)
                    else:
                        _remove_vec(mask, cnt, x, N)
                        cur_size -= 1
                        cur_viol = violation_of(cnt)
                else:
                    _remove_vec(mask, cnt, x, N)
                    cur_size -= 1
                    cur_viol = violation_of(cnt)
        else:
            # REMOVE: remove element with lowest marginal contribution
            in_idx = np.nonzero(mask)[0]
            if in_idx.size == 0:
                continue
            if in_idx.size > 1:
                # Find element whose removal frees the most capacity
                # i.e., element involved in most count-2 sums
                best_x = in_idx[0]
                best_freed = -1
                for x in in_idx:
                    others = in_idx[in_idx != x]
                    if others.size:
                        ss = others + x
                        freed = (cnt[ss] == 2).sum()
                    else:
                        freed = 0
                    if freed > best_freed:
                        best_freed = freed
                        best_x = x
                x = int(best_x)
            else:
                x = int(in_idx[0])
            _remove_vec(mask, cnt, x, N)
            new_viol = violation_of(cnt)
            delta_E = -1 + lambda_pen * (new_viol - cur_viol)
            accept = (delta_E < 0) or (rng.random() < np.exp(-delta_E / max(T, 1e-9)))
            if accept:
                cur_size -= 1
                cur_viol = new_viol
            else:
                _add_vec(mask, cnt, x, N)
                cur_viol = violation_of(cnt)
                cur_size += 1

        if improved:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                mask = best_mask_local.copy()
                cnt = _sum_counts_vec(mask, N)
                cur_size = int(mask.sum())
                cur_viol = violation_of(cnt)
                T = 3.0
                stuck_count = 0

        T *= cooling
        if T < T_min:
            T = T_min
    return best_mask_local, best_size_local

def _polish_pass(mask, N, rng, time_limit):
    """Final high-precision polish: try every possible single-element swap and keep improvements."""
    cnt = _sum_counts_vec(mask, N)
    cur_size = int(mask.sum())
    best_size = cur_size
    best_mask = mask.copy()

    def violation_of(cnt):
        return int(np.maximum(0, cnt - 2).sum())

    while time.time() < time_limit:
        improved = False
        in_idx = np.nonzero(mask)[0]
        out_idx = np.nonzero(~mask)[0]
        if in_idx.size == 0 or out_idx.size == 0:
            break

        # Try each in-set element for removal, then best add
        for x in in_idx:
            if time.time() >= time_limit:
                break
            _remove_vec(mask, cnt, x, N)
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if feas.any():
                cand = np.nonzero(feas)[0]
                scores = _conflict_score_bitset(mask, cnt, N, cand)
                y = int(cand[np.argmin(scores)])
                _add_vec(mask, cnt, y, N)
                new_viol = violation_of(cnt)
                if new_viol == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_mask = mask.copy()
                    improved = True
                else:
                    _remove_vec(mask, cnt, y, N)
            _add_vec(mask, cnt, x, N)
            cur_viol = violation_of(cnt)

        if not improved:
            # Also try pure insertions
            feas = _can_add_vec(mask, cnt, N)
            feas = feas & ~mask
            if feas.any():
                cand = np.nonzero(feas)[0]
                scores = _conflict_score_bitset(mask, cnt, N, cand)
                y = int(cand[np.argmin(scores)])
                _add_vec(mask, cnt, y, N)
                cur_size += 1
                new_viol = violation_of(cnt)
                if new_viol == 0 and cur_size > best_size:
                    best_size = cur_size
                    best_mask = mask.copy()
                    improved = True
                else:
                    _remove_vec(mask, cnt, y, N)
                    cur_size -= 1

        if not improved:
            break

    return best_mask, best_size

def construct():
    N = 1000
    start_time = time.time()
    time_limit = 145.0  # safety margin before 150s kill

    rng = random.Random(12345)

    # Multiple randomized greedy seeds (skip-noise) to diversify
    best_mask = None
    best_size = 0
    seed_time = 20.0
    while time.time() < start_time + seed_time:
        skip_p = rng.random() * 0.25
        mask, cnt = _saturating_greedy(N, rng, skip_p=skip_p)
        s = int(mask.sum())
        if s > best_size:
            best_size = s
            best_mask = mask
            best_cnt = cnt

    # Local search to grow beyond greedy
    if best_mask is None:
        best_mask = np.zeros(N + 1, dtype=bool)
        best_cnt = np.zeros(2 * N + 1, dtype=np.int32)
    else:
        best_cnt = _sum_counts_vec(best_mask, N)

    # First local search pass
    best_mask, best_size = _local_search(
        best_mask.copy(), best_cnt.copy(), N, rng,
        time_limit - 15.0, best_mask, best_size, lambda_pen=10.0
    )

    # Second local search pass with different parameters
    if time.time() < time_limit - 8.0:
        cnt2 = _sum_counts_vec(best_mask, N)
        best_mask, best_size = _local_search(
            best_mask.copy(), cnt2, N, rng,
            time_limit - 5.0, best_mask, best_size, lambda_pen=12.0
        )

    # Final high-precision polish pass
    if time.time() < time_limit - 2.0:
        best_mask, best_size = _polish_pass(
            best_mask.copy(), N, rng, time_limit - 0.5
        )

    result = [int(i) for i in np.nonzero(best_mask)[0]]
    return result
