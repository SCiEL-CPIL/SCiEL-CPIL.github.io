import time
import random

_start_time = time.time()
DEADLINE = _start_time + 85

N = 12
M = 40
TOTAL = 1 << N  # 4096

def construct():
    # Precompute comparability matrix as bytearrays
    comp = [bytearray(TOTAL) for _ in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(i, TOTAL):
            if (i | j) == i or (i | j) == j:
                comp[i][j] = 1
                if i != j:
                    comp[j][i] = 1

    # Precompute subset and superset lists for faster greedy
    subsets = [[] for _ in range(TOTAL)]
    supersets = [[] for _ in range(TOTAL)]
    for mask in range(TOTAL):
        for sub in range(TOTAL):
            if sub & mask == sub and sub != mask:
                subsets[mask].append(sub)
            if (sub | mask) == sub and sub != mask:
                supersets[mask].append(sub)

    def count_pairs(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                cnt += comp[family[i]][family[j]]
        return cnt

    def get_degrees(family):
        deg = [0] * len(family)
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                c = comp[family[i]][family[j]]
                deg[i] += c
                deg[j] += c
        return deg

    def greedy_init(seed=None):
        """Greedy construction with optional randomization for diversity."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        fam = [0, TOTAL - 1]
        seen = set(fam)
        
        for _ in range(M - 2):
            best_m = -1
            best_s = -1
            for m in range(TOTAL):
                if m in seen:
                    continue
                s = 0
                for f in fam:
                    s += comp[m][f]
                if s > best_s or (s == best_s and rng.random() < 0.3):
                    best_s = s
                    best_m = m
            fam.append(best_m)
            seen.add(best_m)
        
        return fam, count_pairs(fam)

    # Initial greedy construction
    family, current_score = greedy_init()
    seen = set(family)
    best_family = list(family)
    best_score = current_score

    # Local search: try swapping each set with a better candidate
    def local_search(fam, score, seen_set, max_time=None):
        imp = True
        while imp:
            if max_time and time.time() > max_time:
                break
            imp = False
            deg = get_degrees(fam)

            for i in range(M):
                if max_time and time.time() > max_time:
                    break
                old = fam[i]
                old_deg = deg[i]

                best_new = -1
                best_gain = 0

                # Try targeted candidates: subsets and supersets of old, plus random
                candidates = set()
                candidates.update(subsets[old][:50])
                candidates.update(supersets[old][:50])
                # Add some random candidates
                for _ in range(100):
                    candidates.add(random.randrange(TOTAL))
                
                for new_m in candidates:
                    if new_m == old or new_m in seen_set:
                        continue
                    new_deg = 0
                    for j in range(M):
                        if j == i:
                            continue
                        new_deg += comp[new_m][fam[j]]
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_m

                if best_new != -1:
                    seen_set.remove(old)
                    seen_set.add(best_new)
                    fam[i] = best_new
                    score += best_gain
                    imp = True
                    break
        
        return fam, score, seen_set

    # Run local search on initial family
    family, current_score, seen = local_search(family, current_score, seen, max_time=DEADLINE - 15)
    if current_score > best_score:
        best_score = current_score
        best_family = list(family)

    # Multi-restart with different seeds if time allows
    if time.time() < DEADLINE - 20:
        for restart_seed in range(5):
            if time.time() > DEADLINE - 10:
                break
            fam, sc = greedy_init(seed=restart_seed * 12345 + 67890)
            seenset = set(fam)
            fam, sc, seenset = local_search(fam, sc, seenset, max_time=DEADLINE - 5)
            if sc > best_score:
                best_score = sc
                best_family = list(fam)

    # Final polish: try all pairs of swaps within a small neighborhood
    if time.time() < DEADLINE - 3:
        fam = list(best_family)
        seen = set(fam)
        sc = best_score
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            deg = get_degrees(fam)
            for i in range(M):
                if time.time() > DEADLINE - 2:
                    break
                old = fam[i]
                old_deg = deg[i]
                best_new = -1
                best_gain = 0
                # Try all subsets and supersets
                for new_m in subsets[old] + supersets[old]:
                    if new_m in seen:
                        continue
                    new_deg = 0
                    for j in range(M):
                        if j == i:
                            continue
                        new_deg += comp[new_m][fam[j]]
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_m
                if best_new != -1:
                    seen.remove(old)
                    seen.add(best_new)
                    fam[i] = best_new
                    sc += best_gain
                    improved = True
                    if sc > best_score:
                        best_score = sc
                        best_family = list(fam)
                    break

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(N) if (mask >> i) & 1]
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
        for j in range(i + 1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1 - t0:.2f}s, Comparables: {cnt}")
