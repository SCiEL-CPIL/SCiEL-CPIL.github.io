"""Minimize edges of a non-2-colorable 5-uniform hypergraph.

Strategy: the best known construction is K_9^5 (126 edges) with edges removed.
Removing a 5-subset S leaves a non-2-colorable hypergraph iff S contains no
5-subset that lies entirely on one side of a 4-5 split of {0..8}. We enumerate
candidate removals and verify each with a fast bitset backtracking + unit
propagation 2-coloring solver, returning the densest valid hypergraph.
"""
import time
from itertools import combinations

# Precompute the 56 5-subsets of {0..8} as 9-bit masks.
_FIVES = [0] * 56
for _i, _c in enumerate(combinations(range(9), 5)):
    _m = 0
    for _v in _c:
        _m |= 1 << _v
    _FIVES[_i] = _m
_FIVES = tuple(_FIVES)

# Precompute every 5-subset that is contained in one side of a 4-5 split.
# A 5-subset S is such a witness iff there exists a 4-subset T of S with
# |S \ T| = 1, i.e. S = T ∪ {x} for some 4-subset T. We precompute the set of
# all such S (as bitmasks) so the removal test is O(1).
_WITNESSES = set()
for _c in combinations(range(9), 5):
    # S is a witness iff it has a 4-subset T such that S\T is a singleton,
    # which is always true: pick any 4 of the 5 elements. So EVERY 5-subset
    # is a witness? No: we need the OTHER side of the split (the 4 elements)
    # to be exactly T, i.e. S ⊆ complement of T is impossible since S∩T=4.
    # Correct criterion: S is contained in one side of a 4-5 split iff there
    # exists a 4-subset T of the 9 vertices with S ⊆ (V \ T). That means
    # T ⊆ V \ S, i.e. T is a 4-subset of the 4-element complement of S.
    # The complement of S has exactly 4 elements, so T = V\S is the unique
    # candidate. Hence EVERY 5-subset S is a witness (T = complement of S).
    # So removing any single 5-edge makes it 2-colorable? That contradicts
    # the 51-edge construction. Re-examine below in _is_witness.
    pass

def _is_witness(mask):
    """True iff this 5-subset is contained in one side of some 4-5 split.

    S ⊆ (V \\ T) for a 4-subset T  <=>  T ⊆ V \\ S. Since V\\S has exactly
    4 elements, T = V\\S works, so every 5-subset is a witness. This means
    removing a single edge from K_9^5 already makes it 2-colorable.
    """
    return True  # every 5-subset is a witness (see docstring)


def _is_non_2colorable(n, edges):
    """Exact 2-colorability check via backtracking + unit propagation.

    Returns True if the hypergraph has NO proper 2-coloring.
    """
    m = len(edges)
    if m == 0:
        return False
    # Precompute for each edge the list of vertex indices.
    # Use bitmasks for vertex sets to make unit propagation fast.
    # Represent a partial assignment as:
    #   fixed0: bitmask of vertices forced to 0
    #   fixed1: bitmask of vertices forced to 1
    # An edge is satisfied (monochromatic) if all its vertices are in fixed0
    # or all in fixed1. It is "unit" (all but one fixed to same color) if
    # it has exactly one free vertex and the rest fixed to one color.

    # Build edge masks.
    emasks = []
    for e in edges:
        mm = 0
        for v in e:
            mm |= 1 << v
        emasks.append(mm)

    # We'll do DFS over vertex assignments. For efficiency, pick an unassigned
    # vertex, try both colors, propagate.
    # Use a stack-based DFS with explicit state to allow early termination.

    # For propagation we need, for each color, the set of edges that are
    # "almost monochromatic" in that color. Recomputing each step is O(m*5),
    # which is fine for m <= 126.

    # Precompute per-edge vertex bitmasks once (done above as emasks).

    def propagate(fixed0, fixed1, edges_list):
        """Given fixed0/fixed1, do unit propagation. Return (ok, new0, new1)."""
        # Iterate to fixpoint.
        changed = True
        while changed:
            changed = False
            for idx, em in enumerate(emasks):
                f0 = em & fixed0
                f1 = em & fixed1
                # Edge already monochromatic in 0?
                if f0 == em:
                    continue
                if f1 == em:
                    continue
                # Check if all-but-one are in fixed0 (the free one forced to 0)
                free = em & ~(fixed0 | fixed1)
                if free and (free & (free - 1)) == 0:  # exactly one free bit
                    # if rest are all 0 -> force free to 0
                    if (em & ~free) == (em & ~free & fixed0):
                        fixed0 |= free
                        changed = True
                    # if rest are all 1 -> force free to 1
                    if (em & ~free) == (em & ~free & fixed1):
                        fixed1 |= free
                        changed = True
                # If both fixed0 and fixed1 hit the edge, contradiction.
                if (f0 and f1):
                    return (False, 0, 0)
        return (True, fixed0, fixed1)

    # DFS
    # Order vertices 0..n-1.
    def dfs(fixed0, fixed1, v):
        # v is the next vertex to assign.
        if v == n:
            return True  # found a valid coloring -> hypergraph IS 2-colorable
        # If vertex already forced, skip.
        bit = 1 << v
        if fixed0 & bit:
            return dfs(fixed0, fixed1, v + 1)
        if fixed1 & bit:
            return dfs(fixed0, fixed1, v + 1)
        # Try color 0.
        ok, n0, n1 = propagate(fixed0 | bit, fixed1, None)
        if ok:
            if dfs(n0, n1, v + 1):
                return True
        # Try color 1.
        ok, n0, n1 = propagate(fixed0, fixed1 | bit, None)
        if ok:
            if dfs(n0, n1, v + 1):
                return True
        return False

    # If dfs returns True, a coloring exists -> it IS 2-colorable.
    return not dfs(0, 0, 0)


def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    n = 9
    all_edges = [list(c) for c in combinations(range(n), 5)]
    m_total = len(all_edges)  # 126

    # We want to REMOVE as many edges as possible while staying non-2-colorable.
    # Removing a single edge S makes it 2-colorable (S is a witness), so we must
    # keep a set of edges whose removal set is NOT "witness-covering".
    #
    # Concretely: a hypergraph H on 9 vertices (subset of K_9^5) is non-2-
    # colorable iff the set of REMOVED edges R is NOT a "witness set", i.e.
    # there is no 4-5 split (A,B) with A∪B=V, |A|=4,|B|=5, such that every
    # edge of H is monochromatic under the split. Equivalently: for the split
    # (A,B) to make H 2-colorable, H must contain no edge that crosses (A,B)
    # in a non-monochromatic way, i.e. every edge of H is ⊆ A or ⊆ B.
    # Edges ⊆ A: none (|A|=4 < 5). Edges ⊆ B: exactly the 5-subsets of B,
    # i.e. {B}. So the split (A,B) makes H 2-colorable iff B ⊆ H, i.e. iff
    # the edge B (the unique 5-subset of B) is in H.
    #
    # Therefore H is non-2-colorable iff for EVERY 4-5 split (A,B), the edge
    # B is NOT in H. There are C(9,4)=126 such splits, each with a unique
    # 5-subset B (and B ranges over all 5-subsets as A ranges over all
    # 4-subsets). So H is non-2-colorable iff H contains NO 5-subset that is
    # the "5-side" of some 4-5 split. But every 5-subset is the 5-side of
    # exactly one 4-5 split (its complement). So H is non-2-colorable iff H
    # is EMPTY. Contradiction again — I'm miscounting.
    #
    # Let me just search empirically: try removing k edges and test.

    # Fallback: return K_9^5 if search fails (guaranteed valid).
    best = {'n': n, 'edges': all_edges}

    # Try to find a sparser valid hypergraph by removing edges.
    # We'll greedily try removing random edges and check validity, keeping
    # the best (fewest edges) found.
    import random
    rng = random.Random(12345)

    # Start from full and try to remove edges one at a time.
    current = list(all_edges)
    current_set = set(tuple(e) for e in current)

    # Verify full is non-2-colorable.
    if _is_non_2colorable(n, current):
        best = {'n': n, 'edges': current}
        best_count = m_total
    else:
        return {'n': n, 'edges': all_edges}

    # Greedy removal: repeatedly try to remove a random edge and keep it
    # removed if the result is still non-2-colorable.
    order = list(range(m_total))
    rng.shuffle(order)
    for idx in order:
        if time.time() > deadline:
            break
        # Try removing edge idx.
        candidate = [e for i, e in enumerate(current) if i != idx]
        if _is_non_2colorable(n, candidate):
            current = candidate
            best = {'n': n, 'edges': current}
            best_count = len(current)

    return best
