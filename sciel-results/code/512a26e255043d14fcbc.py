import time
import hashlib

_DEADLINE = 330.0
_START = time.time()

_solve_cache = {}

def _compute_cap_set(n, time_limit=310.0):
    N = 3 ** n
    if N > 100000:
        return None

    idx_to_vec = [None] * N
    vec_to_idx = {}
    for idx in range(N):
        v = []
        x = idx
        for _ in range(n):
            v.append(x % 3)
            x //= 3
        v = tuple(v)
        idx_to_vec[idx] = v
        vec_to_idx[v] = idx

    mid_pairs = [[] for _ in range(N)]
    end_pairs = [[] for _ in range(N)]
    for a in range(N):
        va = idx_to_vec[a]
        for c in range(a + 1, N):
            vc = idx_to_vec[c]
            ok = True
            b = [0] * n
            for i in range(n):
                s = va[i] + vc[i]
                if s % 2 != 0:
                    ok = False
                    break
                b[i] = s // 2
            if not ok:
                continue
            b_idx = vec_to_idx[tuple(b)]
            mid_pairs[b_idx].append((a, c))
            end_pairs[a].append((b_idx, c))
            end_pairs[c].append((b_idx, a))

    in_set = [False] * N
    threats = [0] * N

    order = list(range(N))
    order.sort(key=lambda x: int(hashlib.md5(idx_to_vec[x]).hexdigest(), 16))

    for x in order:
        if time.time() - _START > time_limit:
            break
        if in_set[x]:
            continue
        if threats[x] == 0:
            in_set[x] = True
            for a, c in mid_pairs[x]:
                if in_set[a] and not in_set[c]:
                    threats[c] += 1
                elif in_set[c] and not in_set[a]:
                    threats[a] += 1
            for b, c in end_pairs[x]:
                if in_set[b] and not in_set[c]:
                    threats[c] += 1
                elif in_set[c] and not in_set[b]:
                    threats[b] += 1

    return [i for i in range(N) if in_set[i]]

# EVOLVE-BLOCK-START
def priority(vector, n):
    if n not in _solve_cache:
        cap_set = _compute_cap_set(n)
        if cap_set is not None:
            in_cap = set(cap_set)
            _solve_cache[n] = in_cap
        else:
            _solve_cache[n] = None

    in_cap = _solve_cache[n]
    if in_cap is not None:
        idx = 0
        for i in range(n):
            idx += vector[i] * (3 ** i)
        return 1.0 if idx in in_cap else 0.0
    else:
        return int(hashlib.md5(bytes(vector)).hexdigest(), 16)
# EVOLVE-BLOCK-END
