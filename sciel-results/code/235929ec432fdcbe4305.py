import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    N = 1 << n  # 4096

    # Precompute comparability matrix
    comp = [bytearray(N) for _ in range(N)]
    for a in range(N):
        for b in range(a + 1, N):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if comp[family[i]][family[j]]:
                    cnt += 1
        return cnt

    def get_degrees(family):
        degrees = [0] * len(family)
        for i in range(len(family)):
            fi = family[i]
            for j in range(i + 1, len(family)):
                if comp[fi][family[j]]:
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees

    # Generate masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(N):
        masks_by_size[bin(mask).count('1')].append(mask)

    # Greedy chain-based initialization
    family = []
    seen = set()

    # Build a primary chain: 0, {0}, {0,1}, {0,1,2}, ..., {0,...,11}
    for sz in range(n + 1):
        mask = (1 << sz) - 1
        family.append(mask)
        seen.add(mask)

    # Now we have 13 sets. Need 27 more.
    # Greedily add sets that maximize comparability with existing family
    remaining = [mask for mask in range(N) if mask not in seen]

    for _ in range(m - len(family)):
        best_mask = -1
        best_deg = -1
        for c in remaining:
            deg = 0
            for s in family:
                if comp[c][s]:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_mask = c
        if best_mask == -1:
            break
        family.append(best_mask)
        seen.add(best_mask)
        remaining.remove(best_mask)

    # Ensure exactly 40
    while len(family) < m:
        for mask in range(N):
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
                break
    if len(family) > m:
        family = family[:m]
        seen = set(family)

    # Local Search: exhaustive hill climbing over all 4096 candidates per slot
    current_score = count_comparables(family)
    best_family = list(family)
    best_score = current_score

    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        degrees = get_degrees(family)

        for i in range(len(family)):
            if time.time() > deadline - 5:
                break
            old_set = family[i]
            old_deg = degrees[i]

            best_new = None
            best_gain = 0

            for new_set in range(N):
                if new_set == old_set or new_set in seen:
                    continue

                new_deg = 0
                for j in range(len(family)):
                    if j == i:
                        continue
                    if comp[new_set][family[j]]:
                        new_deg += 1

                gain = new_deg - old_deg
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_set

            if best_new is not None:
                family[i] = best_new
                seen.remove(old_set)
                seen.add(best_new)
                current_score += best_gain
                improved = True

                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)

                break

    # Final exhaustive polish: multiple passes over all slots
    if time.time() < deadline - 8:
        polish_family = list(best_family)
        polish_seen = set(polish_family)
        polish_score = best_score

        for _ in range(10):
            if time.time() > deadline - 5:
                break

            any_improvement = False
            for i in range(len(polish_family)):
                if time.time() > deadline - 5:
                    break
                old_set = polish_family[i]

                best_new = None
                best_gain = 0

                for new_set in range(N):
                    if new_set == old_set or new_set in polish_seen:
                        continue

                    new_deg = 0
                    for j in range(len(polish_family)):
                        if j == i:
                            continue
                        if comp[new_set][polish_family[j]]:
                            new_deg += 1

                    old_deg = 0
                    for j in range(len(polish_family)):
                        if j == i:
                            continue
                        if comp[old_set][polish_family[j]]:
                            old_deg += 1

                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None and best_gain > 0:
                    polish_family[i] = best_new
                    polish_seen.remove(old_set)
                    polish_seen.add(best_new)
                    polish_score += best_gain
                    any_improvement = True

                    if polish_score > best_score:
                        best_score = polish_score
                        best_family = list(polish_family)

            if not any_improvement:
                break

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
