"""Clique partition of three fixed split (chordal) graphs.
Keeps the exact instance() generator from the seed program.

Structural change: instead of the previous sequential DSATUR-on-claimed-core
strategy, we model the problem as a maximum-weight clique partition of a
weighted graph.  Each potential clique C (a set of vertices that is a clique
in G) has weight = number of edges in C that are NOT yet covered.  We then
use a greedy weighted-max-clique removal loop, but with a crucial improvement:
we pre-compute all maximal cliques of the split graph (which are of the form
{v} ∪ S where S ⊆ N(v) ∩ K and S is a clique in the core, i.e. S ⊆ K since
K is a clique).  For an independent vertex v, any subset of N(v) forms a valid
clique {v} ∪ S.  The core K itself is a clique of size k.

Key insight for split graphs: every maximal clique is either K itself, or
{v} ∪ (N(v) ∩ K) for some independent vertex v.  But we don't need maximal
cliques – we need a partition.  The problem is equivalent to: cover all edges
with minimum number of cliques, each edge in exactly one clique.

New approach: treat it as a set cover / partition problem solved by
iterative greedy with look-ahead.  At each step, pick the clique that covers
the most uncovered edges per "cost" (1 clique).  Use a priority-queue-based
greedy with dynamic re-evaluation.

For efficiency on n=90, k=30: there are 60 independent vertices, each with
~21 neighbors on average (p=0.7).  The number of possible cliques involving
one independent vertex v is 2^|N(v)| which is too many.  But we can restrict
to "promising" cliques: for each v, consider cliques {v} ∪ S where S is a
maximal clique in the induced subgraph on N(v) under the "uncovered core
edges" constraint.  Since K is a clique, any subset of N(v) works structurally;
the constraint is only about which core edges are still uncovered.

Simpler effective heuristic: 
1. First, cover as many cross edges as possible using large cliques.
   For each independent vertex v, form the largest possible clique {v} ∪ S
   where S ⊆ N(v) and all core edges within S are still uncovered (so we
   claim them).  This is equivalent to finding a clique in the graph H_v
   where H_v has vertex set N(v) and edge (u,w) if core edge {u,w} is uncovered.
   Since K is a clique, H_v is a subgraph of K induced by N(v) with some
   edges removed (those already covered).  We want a maximum clique in H_v.
   
   But maximum clique is NP-hard.  However, H_v is a subgraph of a clique K
   with edges removed, so H_v is a "co-clique" complement... Actually H_v is
   just an arbitrary graph on |N(v)| vertices.  For |N(v)| ≈ 21, we can use
   a branch-and-bound or simply a greedy approximation.

2. After handling all independent vertices, cover remaining core edges.

Let me implement a cleaner version:
- For each independent vertex v (in some order), find a large subset S ⊆ N(v)
  such that all core edges within S are uncovered.  Add clique {v} ∪ S, mark
  those core edges as covered, and also mark all cross edges (v,u) for u∈S as
  covered.
- The "find large S" step: build graph on N(v) where two vertices are adjacent
  iff their core edge is uncovered.  Find a large clique in this graph.
  Use a simple greedy: repeatedly pick the vertex with highest degree among
  remaining, add to clique, remove non-neighbors.

- Repeat until no more progress, then cover leftover core edges greedily.

Ordering: process independent vertices by descending |N(v)| to pack big ones first.
"""
import time
import sys

_DEADLINE = 130.0  # wall-clock safety margin (evaluator kills at 150s)
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _greedy_clique_in_graph(vertices, adj):
    """Find a large clique in graph (vertices, adj) using greedy.
    adj: dict vertex -> set of neighbors within vertices.
    Returns a list (the clique)."""
    remaining = set(vertices)
    clique = []
    while remaining:
        # pick vertex with max degree in remaining
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        clique.append(v)
        remaining.discard(v)
        # keep only neighbors of v in remaining
        remaining = remaining & adj[v]
        if not remaining:
            break
    return clique


def _cover_leftover_core(F, K):
    """Cover the set F of core edges (on vertex set K) with few cliques.
    Greedy: repeatedly find a large clique in the graph (K, F-edges), remove
    its edges from F, add clique to result."""
    F = set(F)
    cliques = []
    while F:
        # build adjacency from F
        adj = {v: set() for v in K}
        for (a, b) in F:
            adj[a].add(b)
            adj[b].add(a)
        # find large clique
        clique = _greedy_clique_in_graph(K, adj)
        if len(clique) < 2:
            # fallback: just take one edge
            edge = next(iter(F))
            clique = [min(edge), max(edge)]
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                F.discard(frozenset((clique[a], clique[b])))
        cliques.append(sorted(clique))
    return cliques


def _solve_instance(n, k, p, seed):
    from random import Random
    rng = Random(seed)

    # cross adjacency: for each independent vertex i in [k, n), its K-neighbors
    S = {}
    for i in range(k, n):
        nbrs = [u for u in range(k) if rng.random() < p]
        S[i] = set(nbrs)

    # all core edges
    core_edges = set()
    for u in range(k):
        for v in range(u + 1, k):
            core_edges.add(frozenset((u, v)))

    uncovered_core = set(core_edges)
    uncovered_cross = set()
    for i in range(k, n):
        for u in S[i]:
            uncovered_cross.add(frozenset((i, u)))

    all_uncovered = uncovered_core | uncovered_cross
    cliques = []

    # Process independent vertices by descending neighbor set size
    order = sorted(range(k, n), key=lambda i: -len(S[i]))

    for i in order:
        if _deadline_hit():
            break
        nbrs = S[i]
        if not nbrs:
            continue

        # Build graph on nbrs: edge (u,w) if core edge {u,w} is still uncovered
        adj = {u: set() for u in nbrs}
        for a in nbrs:
            for b in nbrs:
                if a != b and frozenset((a, b)) in uncovered_core:
                    adj[a].add(b)

        # Find a large clique in this graph
        part = _greedy_clique_in_graph(nbrs, adj)

        if len(part) < 1:
            continue

        # The clique is {i} ∪ part
        clique = [i] + sorted(part)
        cliques.append(clique)

        # Mark cross edges (i, u) for u in part as covered
        for u in part:
            uncovered_cross.discard(frozenset((i, u)))

        # Mark core edges within part as covered
        for a in range(len(part)):
            for b in range(a + 1, len(part)):
                uncovered_core.discard(frozenset((part[a], part[b])))

    # Second pass: try to pick up remaining cross edges with smaller cliques
    # For each independent vertex, see if there are uncovered cross edges
    for i in order:
        if _deadline_hit():
            break
        # Find which neighbors still have uncovered cross edge to i
        remaining_nbrs = set()
        for u in S[i]:
            if frozenset((i, u)) in uncovered_cross:
                remaining_nbrs.add(u)
        if not remaining_nbrs:
            continue

        # Build graph on remaining_nbrs with uncovered core edges
        adj = {u: set() for u in remaining_nbrs}
        for a in remaining_nbrs:
            for b in remaining_nbrs:
                if a != b and frozenset((a, b)) in uncovered_core:
                    adj[a].add(b)

        part = _greedy_clique_in_graph(remaining_nbrs, adj)
        if len(part) < 1:
            # Just take one neighbor
            part = [next(iter(remaining_nbrs))]

        clique = [i] + sorted(part)
        cliques.append(clique)

        for u in part:
            uncovered_cross.discard(frozenset((i, u)))
        for a in range(len(part)):
            for b in range(a + 1, len(part)):
                uncovered_core.discard(frozenset((part[a], part[b])))

    # Cover leftover core edges
    if uncovered_core:
        core_cliques = _cover_leftover_core(uncovered_core, list(range(k)))
        cliques.extend(core_cliques)

    return cliques


def construct():
    global _t0
    _t0 = time.time()
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for spec in specs:
        if _deadline_hit():
            # safety: return what we have (will not normally trigger)
            break
        cliques = _solve_instance(*spec)
        # enforce exact types: list of lists of Python int
        cliques = [[int(v) for v in c] for c in cliques]
        result.append(cliques)
    return result
