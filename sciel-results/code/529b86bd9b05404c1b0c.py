import random
import time

N = 128
DEADLINE = 80.0  # safety margin before the 90s kill

# Precompute, for each unordered pair (i,j), the set of k != i,j such that
# the triple {i,j,k} is "bad": at least one of i+j, i+k, j+k is in [1..N].
# (If all three pair sums exceed N, the triple can never be bad, so it's omitted.)
def _build_bad():
    bad = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            ij = i + j
            if ij > N:
                # Only k with i+k<=N or j+k<=N can make a bad triple.
                # k must satisfy k<=N-i or k<=N-j, i.e. k <= max(N-i, N-j) = N - min(i,j)
                lim = N - i if i < j else N - j  # N - min(i,j)
                lim = min(lim, N)
            else:
                lim = N
            s = set()
            for k in range(1, lim + 1):
                if k == i or k == j:
                    continue
                ik = i + k
                jk = j + k
                if ij <= N or ik <= N or jk <= N:
                    s.add(k)
            if s:
                bad[(i, j)] = s
    return bad

_BAD = _build_bad()

def _conflict(x, chosen):
    # Does adding x create a bad triple with two existing members?
    for y in chosen:
        if y < x:
            key = (y, x)
        else:
            key = (x, y)
        s = _BAD.get(key)
        if s is not None:
            for z in s:
                if z in chosen:
                    return True
    return False

def _seed_set():
    return set(range(65, 129))  # valid 64

def _greedy_random(chosen, order):
    chosen = set(chosen)
    for x in order:
        if x in chosen:
            continue
        if not _conflict(x, chosen):
            chosen.add(x)
    return chosen

def construct():
    start = time.time()
    best = _seed_set()
    best_size = len(best)
    rng = random.Random(1234567)

    # Deterministic descending greedy first (fast, often large)
    cand = set()
    for x in range(N, 0, -1):
        if not _conflict(x, cand):
            cand.add(x)
    if len(cand) > best_size:
        best, best_size = cand, len(cand)

    # Randomized restarts with varying orderings
    while time.time() - start < DEADLINE:
        if rng.random() < 0.5:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        else:
            order = list(range(N, 0, -1))
            rng.shuffle(order)
        cand = _greedy_random(best, order)
        if len(cand) > best_size:
            best, best_size = cand, len(cand)
            if best_size >= 70:
                # good enough; keep polishing a bit
                pass
    return [int(v) for v in sorted(best)]

if __name__ == "__main__":
    print(len(construct()))
