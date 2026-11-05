import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 78.0  # safety margin under the 90s kill

    # All C(16,4)=1820 four-subsets as 16-bit masks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)

    def mask_to_list(m):
        return [i + 1 for i in range(16) if (m >> i) & 1]

    def build(seed):
        rng = random.Random(seed)
        family = []
        # inter[A][v] = list of B in family with A & B == v
        inter = {}
        cand = all_sets[:]
        rng.shuffle(cand)
        cand_pos = {m: i for i, m in enumerate(cand)}

        def check(C):
            # True iff adding C creates no 3-sunflower with two existing sets
            for A in family:
                v = C & A
                lst = inter[A].get(v)
                if lst:
                    for B in lst:
                        if (B & C) == v:
                            return False
            return True

        def add(C):
            family.append(C)
            # remove C from candidate pool (swap-pop)
            i = cand_pos.pop(C)
            last = cand[-1]
            cand.pop()
            if i < len(cand):
                cand[i] = last
                cand_pos[last] = i
            # update intersection index
            inter[C] = {}
            for A in family[:-1]:
                v = A & C
                if v not in inter[A]:
                    inter[A][v] = []
                inter[A][v].append(C)
                if v not in inter[C]:
                    inter[C][v] = []
                inter[C][v].append(A)

        while cand and len(family) < 250:
            if time.time() > DEADLINE:
                break
            found = None
            # try a batch of random candidates (fast path)
            K = 150
            tried = set()
            for _ in range(K):
                if not cand:
                    break
                C = cand[rng.randrange(len(cand))]
                if C in tried:
                    continue
                tried.add(C)
                if check(C):
                    found = C
                    break
            # fallback: full scan
            if found is None:
                for C in cand:
                    if check(C):
                        found = C
                        break
            if found is None:
                break
            add(found)
        return family

    best = []
    s = 0
    while time.time() < DEADLINE:
        s += 1
        fam = build(s)
        if len(fam) > len(best):
            best = fam
        if len(best) >= 250:
            break

    if not best:
        best = [all_sets[0]]

    return [mask_to_list(m) for m in best]
