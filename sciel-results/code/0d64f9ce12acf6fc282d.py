"""Shortest-path over squarefree kernels to build a square-product extension."""
import time
import math
from collections import deque

# EVOLVE-BLOCK-START
N = 210
MAXV = 10210
MAX_SIZE = 100
DEADLINE = 85.0  # safety margin under 90s wall-clock


def _primes_up_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def _build_kernel_maps():
    primes = _primes_up_to(MAXV)
    idx = {p: i for i, p in enumerate(primes)}
    n = len(primes)
    # smallest prime factor
    spf = [0] * (MAXV + 1)
    for p in primes:
        for m in range(p, MAXV + 1, p):
            if spf[m] == 0:
                spf[m] = p
    # squarefree kernel of each integer 1..MAXV
    kern = [0] * (MAXV + 1)
    kern[1] = 0
    for v in range(2, MAXV + 1):
        x = v
        bits = 0
        while x > 1:
            p = spf[x]
            c = 0
            while x % p == 0:
                x //= p
                c += 1
            if c & 1:
                bits |= (1 << idx[p])
        kern[v] = bits
    return kern, idx, primes


def _isqrt(x):
    r = math.isqrt(x)
    return r


def construct():
    t0 = time.time()
    kern, idx, primes = _build_kernel_maps()

    target = kern[N]  # kernel of 210 = 2*3*5*7

    # candidate integers (211..10210) with their kernels; keep the smallest
    # integer for each kernel (cost = a - 210, so smaller a is always better)
    kernel_to_min_a = {}
    for a in range(N + 1, MAXV + 1):
        k = kern[a]
        cur = kernel_to_min_a.get(k)
        if cur is None or a < cur:
            kernel_to_min_a[k] = a

    # candidate kernel list with (kernel, a)
    cand = [(k, a) for k, a in kernel_to_min_a.items()]

    # Dijkstra over kernel states. dist[state] = min cost (sum of a-210) to reach state.
    # start = target, goal = 0.
    INF = float('inf')
    dist = {target: 0}
    prev = {}  # state -> (prev_state, a)
    import heapq
    heap = [(0, target)]
    found = False
    goal_state = 0

    while heap:
        if time.time() - t0 > DEADLINE:
            break
        d, s = heapq.heappop(heap)
        if d > dist.get(s, INF):
            continue
        if s == goal_state:
            found = True
            break
        for k, a in cand:
            ns = s ^ k
            nd = d + (a - N)
            if nd < dist.get(ns, INF):
                dist[ns] = nd
                prev[ns] = (s, a)
                heapq.heappush(heap, (nd, ns))

    if not found:
        # Fallback: return a trivial valid answer (single element 4*210)
        return [4 * N]

    # Reconstruct
    sel = []
    s = goal_state
    while s != target:
        ps, a = prev[s]
        sel.append(a)
        s = ps
    sel.reverse()

    # Validate
    if len(sel) > MAX_SIZE:
        return [4 * N]
    if len(set(sel)) != len(sel):
        return [4 * N]
    prod = N
    for a in sel:
        prod *= a
    r = math.isqrt(prod)
    if r * r != prod:
        return [4 * N]
    for a in sel:
        if not (isinstance(a, int) and not isinstance(a, bool) and a > N and a <= MAXV):
            return [4 * N]

    return [int(a) for a in sel]
# EVOLVE-BLOCK-END
