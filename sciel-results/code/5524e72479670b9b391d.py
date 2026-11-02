"""Difference triangle set (7,5) - find 7 rows of 6 marks minimizing max entry.

Combined strategy:
- Structural: SA-style stochastic local search with diverse move types
  (row re-solve, multi-position moves, row swaps) from the inspiration,
  plus a learned-priority-guided constructive search from the current program.
- Refinement: incremental used-diff maintenance for O(1) conflict checks,
  right-to-left position ordering, and a tight deadline guard.
- Representation: rows are lists of ints; diffs are maintained as a global
  set plus per-row sets for fast conflict testing.
"""
import time
import random
import copy
import math


def _validate(rows):
    """Check all positive differences within rows are globally distinct."""
    diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """Compute set of all positive differences in a row."""
    dset = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            dset.add(row[j] - row[i])
    return dset


def _is_sidon(row):
    """Check if all pairwise differences in the row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True


def _greedy_row(used, k, max_x=None, rng=None, bias=None):
    """Build one row greedily: start at 0, pick next mark with no diff conflicts.
    bias: optional list of dicts mapping value->weight for learned priorities."""
    row = [0]
    limit = (max_x if max_x is not None else 500)
    while len(row) < k + 1:
        base = row[-1]
        pos = len(row) - 1
        candidates = []
        for x in range(base + 1, limit + 1):
            new_diffs = {x - a for a in row}
            if len(new_diffs) == len(row) and not (new_diffs & used):
                candidates.append(x)
            if len(candidates) > 200:
                break
        if not candidates:
            return None
        if bias is not None and pos < len(bias) and bias[pos] and rng is not None:
            weighted = []
            for x in candidates:
                w = bias[pos].get(x, 0.01)
                w *= (1.0 / (1.0 + x * 0.01))
                weighted.append((x, w))
            total_w = sum(w for _, w in weighted)
            if total_w > 0:
                r = rng.random() * total_w
                cum = 0
                for x, w in weighted:
                    cum += w
                    if r <= cum:
                        row.append(x)
                        used |= {x - a for a in row[:-1]}
                        break
                else:
                    row.append(weighted[-1][0])
                    used |= {weighted[-1][0] - a for a in row[:-1]}
            else:
                row.append(candidates[0])
                used |= {candidates[0] - a for a in row[:-1]}
        else:
            if rng is not None and rng.random() < 0.3:
                pick = rng.choice(candidates[:min(10, len(candidates))])
            else:
                pick = candidates[0]
            row.append(pick)
            used |= {pick - a for a in row[:-1]}
    return row


def _greedy_all(n_rows, k, rng=None, max_x=None, bias=None):
    """Build all rows greedily, each after the previous."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, max_x=max_x, rng=rng, bias=bias)
        if row is None:
            return None
        rows.append(row)
    return rows


def _learn_priorities(population, k):
    """
    Extract statistical patterns from a population of good solutions.
    Returns a list of dicts: for each position (1..k), a mapping from value -> frequency.
    """
    pos_values = [{} for _ in range(k + 1)]
    for sol in population:
        for row in sol:
            for pi in range(1, k + 1):
                val = row[pi]
                if val not in pos_values[pi]:
                    pos_values[pi][val] = 0
                pos_values[pi][val] += 1
    biases = [None] * (k + 1)
    for pi in range(1, k + 1):
        if pos_values[pi]:
            total = sum(pos_values[pi].values())
            biases[pi] = {v: cnt / total for v, cnt in pos_values[pi].items()}
        else:
            biases[pi] = {}
    return biases


def _learned_constructive_search(n_rows, k, biases, rng, deadline, max_x=None):
    """
    Constructive search guided by learned value priorities.
    """
    if max_x is None:
        max_x = 120
    used_diffs = set()
    rows = []
    for row_idx in range(n_rows):
        row = [0]
        success = False
        for attempt in range(50):
            row = [0]
            row_ok = True
            for pos in range(1, k + 1):
                base = row[-1]
                candidates = []
                for val in range(base + 1, max_x + 1):
                    new_diffs = {val - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                        candidates.append(val)
                    if len(candidates) > 300:
                        break
                if not candidates:
                    row_ok = False
                    break
                if biases is not None and pos < len(biases) and biases[pos]:
                    weighted = []
                    for val in candidates:
                        w = biases[pos].get(val, 0.01)
                        w *= (1.0 / (1.0 + val * 0.01))
                        weighted.append((val, w))
                    total_w = sum(w for _, w in weighted)
                    if total_w > 0:
                        r = rng.random() * total_w
                        cum = 0
                        selected = None
                        for val, w in weighted:
                            cum += w
                            if r <= cum:
                                selected = val
                                break
                        if selected is None:
                            selected = weighted[-1][0]
                    else:
                        selected = candidates[0]
                else:
                    selected = candidates[0]
                row.append(selected)
                used_diffs |= {selected - a for a in row[:-1]}
            if row_ok and len(row) == k + 1:
                success = True
                break
        if not success:
            return None
        rows.append(row)
    return rows


def _sa_local_search(rows, n_rows, k, rng, deadline, max_x=None):
    """
    SA-style stochastic local search with diverse moves:
    - Single-position move (right-to-left bias)
    - Row re-solve (rebuild a row from scratch)
    - Row swap (exchange two rows)
    Incremental diff maintenance for O(1) conflict checks.
    """
    if max_x is None:
        max_x = _max_entry(rows)

    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)
    current_score = best_score

    # Precompute per-row diffs and global used
    row_diffs = [_row_diffs(r) for r in current_rows]
    used_all = set()
    for rd in row_diffs:
        used_all |= rd

    T = 10.0
    T_min = 0.1
    cooling = 0.998

    step = 0
    while time.time() < deadline:
        step += 1
        T = max(T_min, T * cooling)

        move_type = rng.random()
        if move_type < 0.45:
            # Single-position move (right-to-left bias for efficiency)
            ri = rng.randrange(n_rows)
            # Bias toward higher positions (which tend to have larger values)
            if rng.random() < 0.6:
                pi = rng.randrange(k // 2 + 1, k + 1)
            else:
                pi = rng.randrange(1, k + 1)

            row = current_rows[ri]
            old_val = row[pi]
            lo = row[pi - 1] + 1
            hi = row[pi + 1] - 1 if pi < k else best_score

            if lo > hi:
                continue

            # Try a few random new values
            found = False
            for _ in range(8):
                span = hi - lo + 1
                if span <= 0:
                    break
                nv = rng.randint(lo, hi)
                if nv == old_val:
                    continue

                cand_row = row[:]
                cand_row[pi] = nv

                if not _is_sidon(cand_row):
                    continue

                new_diffs = _row_diffs(cand_row)
                # Check conflicts with other rows using incremental approach
                other_diffs = used_all - row_diffs[ri]
                if new_diffs & other_diffs:
                    continue

                # Valid move - compute delta
                new_row_max = max(cand_row)
                other_max = max(max(current_rows[oi]) for oi in range(n_rows) if oi != ri)
                new_score = max(new_row_max, other_max)
                delta = new_score - current_score

                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    # Accept
                    used_all = used_all - row_diffs[ri]
                    used_all |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    current_score = new_score

                    if new_score < best_score:
                        best_score = new_score
                        best_rows = copy.deepcopy(current_rows)
                    found = True
                    break

        elif move_type < 0.80:
            # Row re-solve: rebuild a row from scratch
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs[ri]
            old_max = max(old_row)

            other_used = used_all - old_diffs
            target = min(old_max + 5, best_score)
            new_row = _greedy_row(other_used, k, max_x=target, rng=rng)
            if new_row is None:
                # Try with slightly larger target
                new_row = _greedy_row(other_used, k, max_x=best_score + 5, rng=rng)
            if new_row is None:
                continue

            new_diffs = _row_diffs(new_row)
            if new_diffs & other_used:
                continue

            new_row_max = max(new_row)
            other_max = max(max(current_rows[oi]) for oi in range(n_rows) if oi != ri)
            new_score = max(new_row_max, other_max)
            delta = new_score - current_score

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                used_all = used_all - old_diffs
                used_all |= new_diffs
                current_rows[ri] = new_row
                row_diffs[ri] = new_diffs
                current_score = new_score

                if new_score < best_score:
                    best_score = new_score
                    best_rows = copy.deepcopy(current_rows)

        else:
            # Row swap
            ri, rj = rng.sample(range(n_rows), 2)
            old_ri = current_rows[ri][:]
            old_rj = current_rows[rj][:]
            # Swapping rows doesn't change the diff set, so validity is preserved
            # But max entry might change (it shouldn't since we're swapping whole rows)
            # Actually max entry is the same. Skip this move as it doesn't help.
            # Instead, do a "partial swap": take the last element of one row and try to
            # re-insert it into the other. This is complex; skip for now.
            pass

    return best_rows, best_score


def _local_search_polish(rows, n_rows, k, rng, max_iter=2000, deadline=None):
    """
    Deterministic local search to reduce max entry by repositioning individual marks.
    Uses incremental diff maintenance.
    """
    if deadline is None:
        deadline = time.time() + 30

    best_rows = copy.deepcopy(rows)
    best_score = _max_entry(best_rows)
    current_rows = copy.deepcopy(rows)

    row_diffs = [_row_diffs(row) for row in current_rows]

    for it in range(max_iter):
        if time.time() > deadline:
            break

        ri = rng.randrange(n_rows)
        row = current_rows[ri]

        # Right-to-left ordering: try positions from high to low
        if rng.random() < 0.6:
            pi = rng.randrange(k // 2 + 1, k + 1)
        else:
            pi = rng.randrange(1, k + 1)

        old_val = row[pi]
        lo = row[pi - 1] + 1
        hi = row[pi + 1] - 1 if pi < k else best_score

        if lo > hi:
            continue

        # Try to reduce the value
        candidates = list(range(lo, min(hi + 1, old_val)))
        rng.shuffle(candidates)

        for new_val in candidates[:15]:
            if new_val == old_val:
                continue

            new_row = row[:]
            new_row[pi] = new_val

            if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                continue

            new_diffs = _row_diffs(new_row)
            expected = (len(new_row) * (len(new_row) - 1)) // 2
            if len(new_diffs) != expected:
                continue

            other_diffs = set()
            for oi in range(n_rows):
                if oi != ri:
                    other_diffs |= row_diffs[oi]

            if new_diffs & other_diffs:
                continue

            # Valid move
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            new_score = _max_entry(current_rows)
            if new_score < best_score:
                best_score = new_score
                best_rows = copy.deepcopy(current_rows)
            break

    return best_rows, best_score


def _diverse_population(n_rows, k, rng, deadline, max_x=None):
    """
    Generate a diverse population of valid solutions using different random strategies.
    Returns list of (score, rows) pairs sorted by score.
    """
    if max_x is None:
        max_x = 150

    population = []
    t_start = time.time()

    # Strategy 1: Pure greedy (deterministic)
    rows = _greedy_all(n_rows, k, rng=None, max_x=max_x)
    if rows is not None and _validate(rows):
        population.append((_max_entry(rows), rows))

    # Strategy 2-8: Randomized greedy with different seeds
    for trial in range(8):
        if time.time() > deadline:
            break
        rng_local = random.Random(trial * 1000 + 7)
        rows = _greedy_all(n_rows, k, rng=rng_local, max_x=max_x)
        if rows is not None and _validate(rows):
            population.append((_max_entry(rows), rows))

    # Strategy 9+: SA from each top initial solution
    for i, (score, rows) in enumerate(population[:3]):
        if time.time() > deadline:
            break
        sa_rows, sa_score = _sa_local_search(
            rows, n_rows, k, rng, deadline - 10, max_x=score
        )
        if _validate(sa_rows):
            population.append((sa_score, sa_rows))

    population.sort(key=lambda x: x[0])
    return population


def construct(n_rows, k):
    deadline = time.time() + 140  # Safety margin from 150s limit
    rng = random.Random(42)

    # Phase 1: Generate diverse population
    population = _diverse_population(n_rows, k, rng, deadline - 30, max_x=150)

    if not population:
        # Emergency fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                for x in range(row[-1] + 1, 10000):
                    new_diffs = {x - a for a in row}
                    if len(new_diffs) == len(row) and not (new_diffs & used):
                        row.append(x)
                        used |= new_diffs
                        break
            best_rows.append(row)
        return best_rows

    best_score, best_rows = population[0]

    # Phase 2: Learn value priorities
    biases = _learn_priorities([r for _, r in population[:10]], k)

    # Phase 3: Learned constructive searches
    if time.time() < deadline - 20:
        for trial in range(15):
            if time.time() > deadline - 15:
                break
            rng_local = random.Random(trial * 777 + 13)
            rows = _learned_constructive_search(
                n_rows, k, biases, rng_local, deadline - 10, max_x=best_score
            )
            if rows is not None and _validate(rows):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best_rows = copy.deepcopy(rows)
                    if len(population) < 20:
                        population.append((score, rows))
                        population.sort(key=lambda x: x[0])
                        biases = _learn_priorities([r for _, r in population[:10]], k)

    # Phase 4: SA local search from best solution
    if time.time() < deadline - 10:
        sa_rows, sa_score = _sa_local_search(
            best_rows, n_rows, k, rng, deadline - 5, max_x=best_score
        )
        if _validate(sa_rows) and sa_score < best_score:
            best_score = sa_score
            best_rows = copy.deepcopy(sa_rows)

    # Phase 5: Deterministic local search polish
    if time.time() < deadline - 3:
        ls_rows, ls_score = _local_search_polish(
            best_rows, n_rows, k, rng, max_iter=5000, deadline=deadline - 1
        )
        if _validate(ls_rows) and ls_score < best_score:
            best_score = ls_score
            best_rows = copy.deepcopy(ls_rows)

    # Final validation
    if not _validate(best_rows):
        for score, rows in population:
            if _validate(rows):
                best_rows = rows
                break

    return best_rows
