"""Rewritten program: beam search on the (n-1)-permutation De Bruijn graph."""
import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _all_perms(n):
    """Yield all permutations of 1..n as tuples."""
    def backtrack(path, remaining):
        if not remaining:
            yield tuple(path)
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            yield from backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    yield from backtrack([], list(range(1, n + 1)))


def _classic_construction(n):
    """Classic recursive superpermutation: length 1!+2!+...+n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def construct(n):
    deadline = time.time() + 130  # safety margin under 150s
    total_perms = math.factorial(n)

    # Trivial / small cases: exact
    if n <= 3:
        # exhaustive is fine
        return "".join(map(str, _classic_construction(n)))

    # For n up to 7, attempt beam search on the De Bruijn graph.
    # Node = (n-1)-permutation (tuple). Edge: drop first, append any symbol not in tail.
    # A full word of length L has L-(n-1) windows; we need to cover all n! n-windows.
    # Equivalent: a walk in the (n-1)-graph visiting all n! "n-states"? 
    # Simpler: build word symbol by symbol; state = (last n-1 symbols, set of visited n-perms).

    # Precompute index for each (n-1)-permutation node and its successors.
    nodes = []
    node_index = {}
    for p in _all_perms(n - 1):
        node_index[p] = len(nodes)
        nodes.append(p)
    num_nodes = len(nodes)

    # Successors: for node p (length n-1), tail = p[1:], for each x not in tail:
    # new node = tail + (x,). But x must make a valid (n-1)-perm, which it does if x not in tail.
    succ = []
    for p in nodes:
        tail = p[1:]
        tail_set = set(tail)
        opts = []
        for x in range(1, n + 1):
            if x not in tail_set:
                opts.append((tail + (x,), x))
        succ.append(opts)

    # Map n-permutation -> bitmask index for visited tracking.
    # We track which n! full permutations have appeared as windows.
    # A window appears when we append a symbol to an (n-1)-node: window = node + (x,)
    # which is an n-permutation.
    nperm_index = {}
    for p in _all_perms(n):
        nperm_index[p] = len(nperm_index)
    num_nperms = len(nperm_index)

    # Beam search: state = (node_id, visited_mask, word_len, last_symbol)
    # We want to minimize word_len when visited_mask == all.
    # word_len = number of symbols so far. Start: we need an initial (n-1)-prefix.
    # Represent visited as a Python int bitmask (num_nperms up to 5040 -> big int, ok).

    all_mask = (1 << num_nperms) - 1

    # Initial: pick a starting node (any (n-1)-perm) and a starting symbol.
    # Word begins with the node's symbols (n-1 symbols), then we extend.
    # Actually let's start the word with the first n-1 symbols = some node p0,
    # then each step appends one symbol and moves to a successor node.
    # The windows seen: after building word of length L, windows are positions 0..L-n.
    # We'll track visited n-perms incrementally.

    best_len = None
    best_word = None

    # Use a priority queue / beam. Given time, use iterative deepening beam.
    # State: (node_id, mask, word) -- storing word is expensive; store length and reconstruct.
    # Instead store the full word as a tuple in the beam (memory heavy for 5040).
    # Compromise: beam width modest, store word as bytes.

    import heapq
    # We'll do a BFS-like beam with pruning: keep top-K states by (len, -popcount).
    # Use a dict keyed by (node_id, mask) -> best (len, word_bytes).

    beam = {}
    # Seed: for each starting node, the word is the node itself (n-1 symbols).
    # No n-window yet (need n symbols). 
    for nid, p in enumerate(nodes):
        w = bytes(p)
        mask = 0
        beam[(nid, mask)] = (n - 1, w)

    width = 200000  # beam width
    steps = 0
    max_steps = num_nperms + n  # enough steps to potentially cover all

    while beam and time.time() < deadline:
        # Expand one layer: for each state, append each successor symbol.
        new_beam = {}
        for (nid, mask), (ln, w) in beam.items():
            for (nnid, x) in succ[nid]:
                # new window = node(nid) + (x,) is an n-permutation? 
                # node = p (n-1 symbols), x not in p[1:], so p+(x,) has n distinct symbols -> valid n-perm.
                nperm = nodes[nid] + (x,)
                bit = 1 << nperm_index[nperm]
                new_mask = mask | bit
                new_ln = ln + 1
                new_w = w + bytes((x,))
                key = (nnid, new_mask)
                if new_mask == all_mask:
                    # complete
                    if best_len is None or new_ln < best_len:
                        best_len = new_ln
                        best_word = new_w
                    # don't keep in beam (already complete)
                    continue
                if key not in new_beam or new_ln < new_beam[key][0]:
                    new_beam[key] = (new_ln, new_w)
        if not new_beam:
            break
        # Prune to width: keep states with smallest ln, tie-break by larger popcount.
        if len(new_beam) > width:
            items = list(new_beam.items())
            # sort by (ln, -popcount)
            items.sort(key=lambda kv: (kv[1][0], -bin(kv[1][1] & 0).count('0') if False else -bin(kv[0][1]).count('1')))
            new_beam = dict(items[:width])
        beam = new_beam
        steps += 1
        if steps > max_steps:
            break

    if best_word is not None:
        return "".join(str(b) for b in best_word)

    # Fallback: classic construction (valid, length 13700 for n=7)
    return "".join(map(str, _classic_construction(n)))

# EVOLVE-BLOCK-END
