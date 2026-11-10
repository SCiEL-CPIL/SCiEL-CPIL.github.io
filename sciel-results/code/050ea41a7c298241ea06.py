"""Linear triple systems on 16 points with small independence number.

Strategy (structural change from the seed program):
  * Represent the system as a set of triples (edges of a 3-uniform hypergraph).
  * Maintain the "pair graph" G: vertices 0..15, edge (u,v) iff some triple
    contains both u and v.  A vertex set S contains NO triple  <=>  S is a
    CLIQUE of G.  Therefore the independence number alpha(H) equals the clique
    number of G, which we compute exactly and FAST with a 16-vertex
    branch-and-bound (McGilvray style).  This is the key speedup over the seed
    program's 2^16-subset brute force, and it lets us run thousands of local
    search steps within the wall-clock budget.
  * Build a maximal linear system by random greedy (every non-edge of G is
    covered by some triple).  Maximality minimises uncovered pairs -> lower alpha.
  * Improve with simulated-annealing local search: swap one triple for another
    (preserving linearity), and grow the system whenever a valid triple exists.
"""
import time
import random
from itertools import combinations

N = 16
DEADLINE_MARGIN = 12.0          # seconds before the 150 s wall-clock kill
ALL_TRIPLES = list(combinations(range(N), 3))
ALL_TRIPLES_SET = set(ALL_TRIPLES)


def _clique_number(adj):
    """Exact clique number of an N-vertex graph given by N bitmasks (int)."""
    best = 0

    def rec(cand, size):
        nonlocal best
        if size + cand.bit_count() <= best:
            return
        if cand == 0:
            if size > best:
                best = size
            return
        # branch on the lowest set bit
        v = (cand & -cand).bit_length() - 1
        rec(cand & ~adj[v], size + 1)          # include v
        rec(cand & ~(1 << v), size)            # exclude v

    rec((1 << N) - 1, 0)
    return best


def _alpha_from_pairs(pairs):
    """Independence number = clique number of the pair graph."""
    adj = [0] * N
    for u, v in pairs:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return _clique_number(adj)


def _maximal_greedy(rnd):
    """Random-greedy maximal linear system.

    Returns (triple_set, pair_mask, pairs_list).  Maximality: every non-edge of
    the pair graph is contained in some triple.
    """
    triples = set()
    pair_mask = 0
    order = ALL_TRIPLES[:]
    rnd.shuffle(order)
    for t in order:
        a, b, c = t
        pm = (1 << (a * N + b)) | (1 << (b * N + a)) | (1 << (a * N + c)) | (1 << (c * N + a)) | (1 << (b * N + c)) | (1 << (c * N + b))
        if pm & pair_mask == 0:
            triples.add(t)
            pair_mask |= pm
    pairs = []
    for idx in range(N * N):
        if (pair_mask >> idx) & 1:
            pairs.append((idx // N, idx % N))
    return triples, pair_mask, pairs


def _all_swaps(triples, pair_mask):
    """Precompute, for every triple in the system, every valid replacement.

    A replacement t' (not already present) is valid iff it shares at most one
    vertex with every other triple.  Using the pair graph this is equivalent to:
    all 3 pairs of t' are covered by triples OTHER than the one being removed.
    Returns dict: triple -> list of replacement triples.
    """
    # count how many triples cover each pair
    pair_count = {}
    for t in triples:
        a, b, c = t
        for p in ((a, b), (a, c), (b, c)):
            if p[0] > p[1]:
                p = (p[1], p[0])
            pair_count[p] = pair_count.get(p, 0) + 1

    swaps = {}
    for t in triples:
        a, b, c = t
        own = [(a, b) if a < b else (b, a),
               (a, c) if a < c else (c, a),
               (b, c) if b < c else (c, b)]
        good = []
        for t2 in ALL_TRIPLES:
            if t2 == t or t2 in triples:
                continue
            x, y, z = t2
            p2 = [(x, y) if x < y else (y, x),
                  (x, z) if x < z else (z, x),
                  (y, z) if y < z else (z, y)]
            ok = True
            for p in p2:
                # pair p must be covered by some triple other than t
                if pair_count.get(p, 0) - (1 if p in own else 0) <= 0:
                    ok = False
                    break
            if ok:
                good.append(t2)
        swaps[t] = good
    return swaps


def _local_search(triples, pair_mask, pairs, rnd, deadline):
    """Simulated-annealing local search over the linear-system edge set."""
    cur_alpha = _alpha_from_pairs(pairs)
    best_alpha = cur_alpha
    best_triples = set(triples)
    best_pairs = list(pairs)

    swaps = _all_swaps(triples, pair_mask, pairs)
    # grow first: add any valid triple until maximal
    def grow(triples, pair_mask, pairs):
        changed = True
        while changed:
            changed = False
            for t in ALL_TRIPLES:
                if t in triples:
                    continue
                a, b, c = t
                pm = (1 << (a * N + b)) | (1 << (b * N + a)) | (1 << (a * N + c)) | (1 << (c * N + a)) | (1 << (b * N + c)) | (1 << (c * N + b))
                if pm & pair_mask == 0:
                    triples.add(t)
                    pair_mask |= pm
                    pairs.extend([(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)])
                    swaps[t] = []          # will be recomputed if needed
                    changed = True
                    break
        return pair_mask, pairs

    pair_mask, pairs = grow(triples, pair_mask, pairs)
    cur_alpha = _alpha_from_pairs(pairs)
    best_alpha = cur_alpha
    best_triples = set(triples)
    best_pairs = list(pairs)
    swaps = _all_swaps(triples, pair_mask, pairs)

    T = 1.0
    T_min = 0.02
    cool = 0.999
    step = 0
    while time.time() < deadline:
        # pick a random triple with at least one swap
        options = [t for t in triples if swaps.get(t)]
        if not options:
            # try to grow again
            pair_mask, pairs = grow(triples, pair_mask, pairs)
            a2 = _alpha_from_pairs(pairs)
            if a2 < best_alpha:
                best_alpha = a2
                best_triples = set(triples)
                best_pairs = list(pairs)
            swaps = _all_swaps(triples, pair_mask, pairs)
            options = [t for t in triples if swaps.get(t)]
            if not options:
                break
            cur_alpha = a2
        t = options[rnd.randrange(len(options))]
        repl_list = swaps[t]
        if not repl_list:
            continue
        t2 = repl_list[rnd.randrange(len(repl_list))]

        # apply swap
        triples.discard(t)
        triples.add(t2)
        # rebuild pairs quickly from scratch (cheap for small systems)
        new_pairs = []
        for x in triples:
            a, b, c = x
            new_pairs.extend([(a, b), (b, a), (a, c), (c, a), (b, c), (c, b)])
        new_alpha = _alpha_from_pairs(new_pairs)

        # decide accept
        if new_alpha < cur_alpha or rnd.random() < min(1.0, ((cur_alpha - new_alpha) + 0.0) / max(T, 1e-9) * 0.5 + (0.3 if new_alpha == cur_alpha else 0.0)):
            cur_alpha = new_alpha
            pairs = new_pairs
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = set(triples)
                best_pairs = list(pairs)
        else:
            # revert
            triples.discard(t2)
            triples.add(t)

        # periodically refresh swap table (cheap enough)
        step += 1
        if step % 25 == 0:
            swaps = _all_swaps(triples, pair_mask, pairs)
            # grow if possible
            pair_mask2, pairs2 = grow(triples, pair_mask, pairs)
            if pairs2 is not pairs:
                pairs = pairs2
                cur_alpha = _alpha_from_pairs(pairs)
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = set(triples)
                    best_pairs = list(pairs)
                swaps = _all_swaps(triples, pair_mask, pairs)
        T = max(T_min, T * cool)

    return best_triples, best_alpha


def construct():
    """Construct a linear triple system on 16 points minimising alpha.

    Returns a list of triples [a, b, c] on points 0..15.
    """
    start = time.time()
    deadline = start + 150.0 - DEADLINE_MARGIN

    best_triples = None
    best_alpha = 16

    # deterministic seed for reproducibility of the search schedule
    master = random.Random(12345)

    # 1) a handful of quick greedy systems to get a strong initial incumbent
    for seed in range(40):
        if time.time() > deadline:
            break
        rnd = random.Random(seed)
        triples, pair_mask, pairs = _maximal_greedy(rnd)
        a = _alpha_from_pairs(pairs)
        if a < best_alpha:
            best_alpha = a
            best_triples = set(triples)

    # 2) local search from the best greedy start, then from random starts
    if time.time() < deadline:
        rnd = random.Random(999)
        bt, ba = _local_search(set(best_triples), None, None, rnd, deadline)
        if ba < best_alpha:
            best_alpha = ba
            best_triples = bt

    # 3) keep trying new random starts + local search until the deadline
    k = 0
    while time.time() < deadline - 1.0:
        k += 1
        rnd = random.Random(100000 + k)
        triples, pair_mask, pairs = _maximal_greedy(rnd)
        a = _alpha_from_pairs(pairs)
        if a < best_alpha:
            best_alpha = a
            best_triples = set(triples)
        # budget a local-search burst; stop if we are close to the deadline
        if time.time() < deadline - 1.0:
            bt, ba = _local_search(set(triples), None, None, rnd, deadline)
            if ba < best_alpha:
                best_alpha = ba
                best_triples = bt
        if best_alpha <= 6:
            break

    if best_triples is None:
        # absolute fallback: 5 disjoint triples (valid, alpha = 16)
        best_triples = {(3 * i, 3 * i + 1, 3 * i + 2) for i in range(5)}

    # sanity: verify linearity and return a plain list of triples
    result = []
    seen = set()
    for t in best_triples:
        if t in seen:
            continue
        seen.add(t)
        result.append(list(t))
    # final linearity check
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            inter = set(result[i]) & set(result[j])
            if len(inter) > 1:
                # should not happen; drop the later triple to stay valid
                result.pop()
                break
    return result
