"""Size-efficient sorting network on 18 wires via randomized local search."""
import time
import random
import numpy as np

N = 18
NUM_BOOL = 1 << N
TIME_LIMIT = 80.0  # safety margin below the 90s kill

# ---------------- correctness (fast, NumPy) ----------------
def _apply_network(net, bools):
    """bools: (NUM_BOOL, N) uint8. Returns sorted bools after applying net."""
    vals = bools.copy()
    for (i, j) in net:
        a = vals[:, i]
        b = vals[:, j]
        m = np.minimum(a, b)
        x = np.maximum(a, b)
        vals[:, i] = m
        vals[:, j] = x
    return vals

def _is_correct(net):
    """Check the network sorts every 0/1 input (the 0-1 principle)."""
    if not net:
        return N == 1
    bools = np.arange(NUM_BOOL, dtype=np.uint32).reshape(NUM_BOOL, 1)
    for k in range(N):
        bools = np.concatenate((bools, (bools & (1 << k)) >> k), axis=1)
    vals = _apply_network(net, bools)
    for k in range(N):
        col = vals[:, k]
        if not np.all(col[:-1] <= col[1:]):
            return False
    return True

# ---------------- seed network ----------------
def _odd_even():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

# ---------------- local search moves ----------------
def _try_remove(net, idx):
    if len(net) <= 1:
        return None
    cand = net[:idx] + net[idx + 1:]
    if _is_correct(cand):
        return cand
    return None

def _try_move(net, idx, pos):
    if idx == pos:
        return None
    c = net[idx]
    if c in net[:pos] + net[pos + 1:]:
        # moving to a spot that duplicates an existing comparator is redundant
        # (still allow, but skip to save time)
        return None
    cand = net[:idx] + net[idx + 1:]
    if pos > idx:
        pos -= 1
    cand.insert(pos, c)
    if _is_correct(cand):
        return cand
    return None

def _try_swap(net, i, j):
    if i == j:
        return None
    cand = net[:]
    cand[i], cand[j] = cand[j], cand[i]
    if _is_correct(cand):
        return cand
    return None

def _try_insert(net, pos, pair):
    # insert a comparator at pos; skip if it already exists at pos-1 (redundant)
    cand = net[:pos] + [pair] + net[pos:]
    if _is_correct(cand):
        return cand
    return None

# ---------------- search ----------------
def _search(initial, time_limit):
    deadline = time.time() + time_limit
    best = initial
    best_len = len(initial)
    cur = initial[:]
    cur_len = len(cur)
    rng = random.Random(12345)

    while time.time() < deadline:
        made = False
        # 1) removals (full scan, keep first that works)
        for idx in range(len(cur)):
            if time.time() > deadline:
                break
            cand = _try_remove(cur, idx)
            if cand is not None:
                cur = cand
                cur_len = len(cur)
                made = True
                break
        # 2) moves (a few random attempts)
        for _ in range(4):
            if time.time() > deadline:
                break
            if len(cur) < 2:
                break
            idx = rng.randrange(len(cur))
            pos = rng.randrange(len(cur))
            cand = _try_move(cur, idx, pos)
            if cand is not None:
                cur = cand
                cur_len = len(cur)
                made = True
                break
        # 3) swaps (a few random attempts)
        for _ in range(4):
            if time.time() > deadline:
                break
            if len(cur) < 2:
                break
            i = rng.randrange(len(cur))
            j = rng.randrange(len(cur))
            cand = _try_swap(cur, i, j)
            if cand is not None:
                cur = cand
                cur_len = len(cur)
                made = True
                break
        # 4) inserts (a few random attempts)
        for _ in range(4):
            if time.time() > deadline:
                break
            if len(cur) >= 4096:
                break
            i = rng.randrange(N)
            j = rng.randrange(N)
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            pos = rng.randrange(len(cur) + 1)
            cand = _try_insert(cur, pos, (a, b))
            if cand is not None:
                cur = cand
                cur_len = len(cur)
                made = True
                break

        # update best
        if cur_len < best_len:
            best = cur[:]
            best_len = cur_len

        # if stuck (no improvement this round), restart from best with fresh shuffle
        if not made:
            cur = best[:]
            cur_len = len(cur)
            # occasionally perturb: insert a random comparator then try to fix
            if rng.random() < 0.5 and cur_len < 4096:
                i = rng.randrange(N)
                j = rng.randrange(N)
                if i != j:
                    a, b = (i, j) if i < j else (j, i)
                    pos = rng.randrange(len(cur) + 1)
                    cur = cur[:pos] + [(a, b)] + cur[pos:]
                    cur_len = len(cur)

    return best

# ---------------- entry point ----------------
def construct():
    t0 = time.time()
    seed = _odd_even()
    # verify seed is correct; if not, fall back to a simple valid construction
    if not _is_correct(seed):
        # fallback: bubble sort network (valid but larger)
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net
    remaining = max(1.0, TIME_LIMIT - (time.time() - t0) - 1.0)
    best = _search(seed, remaining)
    # final safety: ensure correctness
    if not _is_correct(best):
        return seed
    return best
