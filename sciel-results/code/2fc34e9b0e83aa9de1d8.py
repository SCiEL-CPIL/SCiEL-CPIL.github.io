"""Rewritten program: Maximum Independent Set on square-sum conflict graph.

Strategy:
  1. Build the conflict graph on [1..256]: edge (a,b) exists iff a+b is a perfect square.
     Note: the graph is the union of perfect matchings (one per square), so max degree is small.
  2. Use a greedy heuristic to get a good initial independent set.
  3. Improve via local search with 'remove 1, add 2' moves (and 'remove 1, add 1', 'remove 2, add 3' etc.)
     to escape local maxima.
  4. Use a deadline guard to return the best found before the 90 s wall-clock limit.
"""

import math
import time
import random

# Precompute perfect squares up to 2*256 = 512
_SQUARES = set()
_i = 1
while _i * _i <= 512:
    _SQUARES.add(_i * _i)
    _i += 1

def _is_square(n):
    return n in _SQUARES

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # safety margin before 90s kill

    N = 256
    # conflict[a] = set of b such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            if _is_square(a + b):
                conflict[a].add(b)
                if a != b:
                    conflict[b].add(a)

    # Check if a candidate set is valid
    def is_valid(s):
        for a in s:
            for b in s:
                if a <= b and _is_square(a + b):
                    return False
        return True

    # Greedy construction: add elements in order of fewest conflicts
    def greedy_init():
        remaining = set(range(1, N + 1))
        chosen = []
        while remaining:
            # Pick element with minimum degree in the induced subgraph
            best = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(conflict[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best = v
            chosen.append(best)
            remaining.discard(best)
            # Remove all neighbors of best
            for nb in conflict[best]:
                remaining.discard(nb)
        return set(chosen)

    # Local search: try moves that increase set size
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try 'add 1' moves
            for v in range(1, N + 1):
                if v not in s and not (conflict[v] & s):
                    s.add(v)
                    improved = True
                    break
            if improved:
                continue
            # Try 'remove 1, add 2' moves
            removed = []
            for v in list(s):
                # If we remove v, which new elements can we add?
                candidates = [u for u in range(1, N + 1)
                              if u not in s and u != v
                              and not (conflict[u] & (s - {v}))]
                if len(candidates) >= 2:
                    # Try adding the best 2
                    # Prefer candidates with fewest conflicts
                    candidates.sort(key=lambda u: len(conflict[u] & s))
                    a, b = candidates[0], candidates[1]
                    s.discard(v)
                    s.add(a)
                    s.add(b)
                    improved = True
                    break
            if improved:
                continue
            # Try 'remove 2, add 3' moves
            for v1 in list(s):
                for v2 in list(s):
                    if v1 >= v2:
                        continue
                    remaining = s - {v1, v2}
                    candidates = [u for u in range(1, N + 1)
                                  if u not in s and u != v1 and u != v2
                                  and not (conflict[u] & remaining)]
                    if len(candidates) >= 3:
                        candidates.sort(key=lambda u: len(conflict[u] & s))
                        a, b, c = candidates[0], candidates[1], candidates[2]
                        s.discard(v1)
                        s.discard(v2)
                        s.add(a)
                        s.add(b)
                        s.add(c)
                        improved = True
                        break
                if improved:
                    break
        return s

    # Branch and bound for maximum independent set on the conflict graph
    # Using a simple recursive approach with pruning
    def branch_and_bound(nodes, current_set, best_set, deadline):
        if time.time() > deadline:
            return best_set
        if not nodes:
            if len(current_set) > len(best_set):
                best_set = set(current_set)
            return best_set

        # Lower bound: current set size
        # Upper bound: current set size + number of remaining nodes
        if len(current_set) + len(nodes) <= len(best_set):
            return best_set

        # Pick a node with maximum degree in the induced subgraph
        v = max(nodes, key=lambda x: len(conflict[x] & nodes))

        # Branch 1: include v
        new_nodes = nodes - {v} - conflict[v]
        current_set.add(v)
        best_set = branch_and_bound(new_nodes, current_set, best_set, deadline)
        current_set.discard(v)

        # Branch 2: exclude v
        new_nodes = nodes - {v}
        best_set = branch_and_bound(new_nodes, current_set, best_set, deadline)

        return best_set

    # Start with greedy
    best = greedy_init()
    best = local_search(best)
    best_size = len(best)

    # Try multiple random restarts with perturbation
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > DEADLINE:
            break
        # Perturb: remove a random fraction and re-greedy
        s = set(best)
        if s:
            to_remove = rng.sample(list(s), max(1, len(s) // 5))
            s -= set(to_remove)
            # Greedy fill
            remaining = set(range(1, N + 1)) - s
            while remaining:
                best_v = min(remaining, key=lambda v: len(conflict[v] & remaining))
                s.add(best_v)
                remaining.discard(best_v)
                for nb in conflict[best_v]:
                    remaining.discard(nb)
            s = local_search(s)
            if len(s) > best_size:
                best_size = len(s)
                best = s

    # Try B&B on a reduced problem if time permits
    if time.time() < DEADLINE - 10.0:
        # Use the current best as a lower bound
        # Try B&B on the full graph but with a time limit
        deadline_bb = min(DEADLINE - 2.0, time.time() + 30.0)
        # B&B on the complement: we want max independent set in conflict graph
        # This is equivalent to min vertex cover in bipartite... but graph isn't bipartite necessarily
        # Just do simple B&B with the current best as lower bound
        # For 256 nodes this may be too slow, so skip if too large
        pass  # B&B on 256 nodes is likely too slow; rely on local search

    # Final verification
    for a in best:
        for b in best:
            if a <= b and _is_square(a + b):
                # This shouldn't happen, but remove b to fix
                best.discard(b)
                break

    if not best:
        # Fallback: just return {1}
        return [1]

    return list(best)
