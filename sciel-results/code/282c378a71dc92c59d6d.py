"""Size-efficient sorting networks on 18 wires: hill-climbing search with
deadline guard and a fast bit-parallel verifier over all 2**18 Boolean inputs."""
import time
import random

N = 18
ALL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 78.0  # safety margin under the 90 s kill
_start = time.time()


def _remaining():
    return DEADLINE_SECONDS - (time.time() - _start)


def _verify_fast(comparators):
    """Verify the network sorts every 0/1 input using a bit-parallel trick.

    For each wire w we keep a bitset S_w of inputs where wire w is 1.
    A comparator (i, j) with min->i, max->j transforms:
        S_i' = S_i & S_j
        S_j' = S_i | S_j
    The network sorts iff for every k, S_0 = S_1 = ... = S_k  (the k smallest
    inputs are exactly the 2**k inputs with k ones, i.e. mask (1<<k)-1).
    """
    S = [0] * N
    for m in range(1, 1 << N):
        S[0] |= m
    for i in range(1, N):
        S[i] = S[i - 1] | (1 << i)
    for (i, j) in comparators:
        si = S[i]
        sj = S[j]
        S[i] = si & sj
        S[j] = si | sj
    for k in range(N):
        if S[0] != ((1 << k) - 1):
            return False
    return True


def _odd_even_transposition(n):
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _try_insert(comps, idx, c):
    cand = comps[:idx] + [c] + comps[idx:]
    return cand if _verify_fast(cand) else None


def _try_replace(comps, idx, c):
    cand = comps[:idx] + [c] + comps[idx + 1:]
    return cand if _verify_fast(cand) else None


def _try_delete(comps, idx):
    cand = comps[:idx] + comps[idx + 1:]
    return cand if _verify_fast(cand) else None


def _search(comps, rng):
    cur = comps
    L = len(cur)
    while _remaining() > 1.5:
        op = rng.random()
        if op < 0.45 and L > 1:
            idx = rng.randrange(L)
            if idx == 0:
                i, j = 0, rng.randrange(1, N)
            elif idx == L - 1:
                i, j = rng.randrange(0, N - 1), N - 1
            else:
                i, j = cur[idx - 1]
                i2, j2 = cur[idx]
                a = min(i, i2)
                b = max(j, j2)
                if a == b:
                    continue
                i, j = a, b
            cand = _try_insert(cur, idx, (i, j))
            if cand is not None:
                cur = cand
                L = len(cur)
        elif op < 0.85:
            idx = rng.randrange(L)
            if idx == 0:
                i, j = 0, rng.randrange(1, N)
            elif idx == L - 1:
                i, j = rng.randrange(0, N - 1), N - 1
            else:
                i, j = cur[idx - 1]
                i2, j2 = cur[idx + 1]
                a = min(i, i2)
                b = max(j, j2)
                if a == b:
                    continue
                i, j = a, b
            cand = _try_replace(cur, idx, (i, j))
            if cand is not None:
                cur = cand
                L = len(cur)
        else:
            idx = rng.randrange(L)
            cand = _try_delete(cur, idx)
            if cand is not None:
                cur = cand
                L = len(cur)
    return cur


# EVOLVE-BLOCK-START
def construct():
    global _start
    _start = time.time()
    rng = random.Random(20240517)
    cur = _odd_even_transposition(N)
    if not _verify_fast(cur):
        raise RuntimeError("seed network failed verification")
    best = list(cur)
    # Hill-climb with periodic restarts from the best-so-far.
    while _remaining() > 8.0:
        cur = _search(best, rng)
        if len(cur) < len(best):
            best = list(cur)
    return best
# EVOLVE-BLOCK-END
