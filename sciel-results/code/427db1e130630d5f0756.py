"""Maximum independent set on the 'pair-sum divides pair-product' conflict graph over [1,256]."""
# EVOLVE-BLOCK-START
import time

_N = 256
_DEADLINE = 85.0  # safety margin under the 90 s wall-clock kill


def _build_graph():
    """Return adjacency as a list of ints (bit i set => conflict with i)."""
    adj = [0] * _N
    for i in range(_N):
        a = i + 1
        row = 0
        for j in range(i + 1, _N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                row |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = row
    return adj


def _greedy_lower_bound(adj):
    """Greedy independent set (max-degree ordering), returns bitset."""
    cand = (1 << _N) - 1
    selected = 0
    while cand:
        # pick vertex with fewest remaining conflicts (min-degree) among candidates
        best = -1
        best_deg = 1 << 30
        tmp = cand
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            d = (adj[v] & cand).bit_count()
            if d < best_deg:
                best_deg = d
                best = v
            tmp ^= lsb
        selected |= (1 << best)
        cand &= ~(adj[best] | (1 << best))
    return selected


def _color_bound(cand, adj):
    """Greedy coloring upper bound on the size of an independent set in cand."""
    colors = 0
    remaining = cand
    while remaining:
        colors += 1
        # one color class: maximal independent set of remaining (greedy)
        color_class = 0
        tmp = remaining
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            color_class |= lsb
            tmp &= ~(adj[v] | lsb)
        remaining &= ~color_class
    return colors


def _branch_and_bound(adj, init_selected):
    """Exact-ish B&B maximizing |selected ∪ I| over independent I ⊆ cand."""
    best = [init_selected]
    best_size = [init_selected.bit_count()]
    deadline = time.time() + _DEADLINE

    # candidate pool: vertices not in init and not adjacent to any init vertex
    forbidden = 0
    s = init_selected
    while s:
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        forbidden |= adj[v]
        s ^= lsb
    cand0 = ((1 << _N) - 1) & ~init_selected & ~forbidden

    # order candidates by degree desc for a tighter search
    order = []
    c = cand0
    while c:
        lsb = c & (-c)
        v = lsb.bit_length() - 1
        order.append(v)
        c ^= lsb
    order.sort(key=lambda v: (adj[v] & cand0).bit_count(), reverse=True)
    idx = {v: i for i, v in enumerate(order)}
    m = len(order)
    order_adj = [adj[v] for v in order]

    # compact adjacency in order-index space
    cmap = [0] * m
    for i, v in enumerate(order):
        bits = 0
        for j, u in enumerate(order):
            if adj[v] & (1 << u):
                bits |= (1 << j)
        cmap[i] = bits

    def dfs(cand_bits, cur_size, cur_sel):
        if time.time() > deadline:
            return
        if cand_bits == 0:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
                best[0] = cur_sel
            return
        # upper bound via greedy coloring
        ub = cur_size + _color_bound(cand_bits, order_adj)
        if ub <= best_size[0]:
            return
        # pick a vertex to branch on: max degree within cand
        tmp = cand_bits
        bestv = -1
        bestd = -1
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            d = (order_adj[v] & cand_bits).bit_count()
            if d > bestd:
                bestd = d
                bestv = v
            tmp ^= lsb
        v = bestv
        bit = 1 << v
        # branch 1: include v
        dfs(cand_bits & ~bit & ~order_adj[v], cur_size + 1, cur_sel | (1 << order[v]), )
        # branch 2: exclude v
        dfs(cand_bits & ~bit, cur_size, cur_sel)

    dfs(cand0, best_size[0], init_selected)
    return best[0]


def construct():
    t0 = time.time()
    adj = _build_graph()
    init = _greedy_lower_bound(adj)
    if time.time() - t0 > _DEADLINE - 5:
        return [i + 1 for i in range(_N) if init & (1 << i)]
    result = _branch_and_bound(adj, init)
    # verify
    sel = [i + 1 for i in range(_N) if result & (1 << i)]
    for x in range(len(sel)):
        for y in range(x + 1, len(sel)):
            a, b = sel[x], sel[y]
            assert (a * b) % (a + b) != 0
    return sel
# EVOLVE-BLOCK-END
