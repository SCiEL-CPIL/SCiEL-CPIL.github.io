import time
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120


def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all 4-cycles (3 per 4-vertex subset)
    vertices = list(range(N))
    cycle_edges = []
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycle_edges.append((get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)))
        # Cycle 2: a-b-d-c-a
        cycle_edges.append((get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)))
        # Cycle 3: a-c-b-d-a
        cycle_edges.append((get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)))

    # For each edge, precompute the list of (cycle_idx, position_in_cycle)
    # so we know which cycles this edge belongs to and at which position
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e0, e1, e2, e3) in enumerate(cycle_edges):
        edge_to_cycles[e0].append((ci, 0))
        edge_to_cycles[e1].append((ci, 1))
        edge_to_cycles[e2].append((ci, 2))
        edge_to_cycles[e3].append((ci, 3))

    def evaluate(colors):
        """Returns (num_unique_colors, num_violations)"""
        num_unique = len(set(colors))
        violations = 0
        for (e0, e1, e2, e3) in cycle_edges:
            c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
            if len({c0, c1, c2, c3}) == 4:
                violations += 1
        return num_unique, violations

    # Greedy construction: process edges in order, assign smallest valid color
    # For edge e, check all cycles containing e. For each cycle, if the other 3
    # edges are already colored, we must avoid creating 4 distinct colors.
    # A color c is forbidden for edge e if there exists a cycle containing e
    # where the other 3 edges have 3 distinct colors, none of which is c,
    # and c is different from all three (making 4 distinct).
    # Actually: if other 3 edges have colors {a, b, c} all distinct,
    # then we cannot use any color d where d != a, d != b, d != c.
    # We CAN use a, b, or c (then the cycle has at most 3 distinct colors).
    # If other 3 edges have colors with some repetition, no constraint.

    def greedy_construct(edge_order):
        colors = [0] * NUM_EDGES  # initialize all to 0
        for e in edge_order:
            # Determine forbidden colors
            forbidden = set()
            for (ci, pos) in edge_to_cycles[e]:
                e0, e1, e2, e3 = cycle_edges[ci]
                # Get the other 3 edges
                other_edges = [e0, e1, e2, e3]
                other_edges.pop(pos)
                other_colors = [colors[oe] for oe in other_edges]
                # If the 3 other colors are all distinct, we can't use any new color
                if len(set(other_colors)) == 3:
                    # Any color not in other_colors would create 4 distinct
                    # So we're forced to use one of the other_colors
                    # But we want the smallest valid color overall
                    # The forbidden set is everything EXCEPT the 3 other colors
                    # We'll handle this by tracking allowed colors
                    pass
            # Better approach: find the smallest color c such that for ALL cycles
            # containing e, the 4 colors don't have 4 distinct values.
            # Start with c = 0, try 0, 1, 2, ...
            # For each candidate c, check all cycles containing e.
            # A cycle is violated if the 4 colors (including c) have 4 distinct values.
            # i.e., if the other 3 colors are all distinct AND c is different from all 3.
            
            # Optimization: if any cycle containing e has 3 distinct other colors,
            # then c MUST be one of those 3 colors.
            # Collect all such constraints.
            must_be_in = None  # intersection of all "must be in" sets
            
            for (ci, pos) in edge_to_cycles[e]:
                e0, e1, e2, e3 = cycle_edges[ci]
                other_edges = [e0, e1, e2, e3]
                other_edges.pop(pos)
                other_colors = [colors[oe] for oe in other_edges]
                if len(set(other_colors)) == 3:
                    s = set(other_colors)
                    if must_be_in is None:
                        must_be_in = s
                    else:
                        must_be_in = must_be_in & s
            
            if must_be_in is not None:
                # c must be in must_be_in; pick the smallest
                candidates = sorted(must_be_in)
                if candidates:
                    colors[e] = candidates[0]
                else:
                    # This shouldn't happen if we process correctly, but fallback
                    colors[e] = 0
            else:
                # No constraint from 3-distinct cycles; use 0
                colors[e] = 0

        return colors

    # Try multiple edge orderings to find the best
    import random
    rng = random.Random(42)

    best_colors = None
    best_score = 0

    # First try natural order
    natural_order = list(range(NUM_EDGES))
    colors = greedy_construct(natural_order)
    score, viol = evaluate(colors)
    if viol == 0 and score > best_score:
        best_score = score
        best_colors = colors.copy()

    # Try random orderings
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        order = list(range(NUM_EDGES))
        rng.shuffle(order)
        colors = greedy_construct(order)
        score, viol = evaluate(colors)
        if viol == 0 and score > best_score:
            best_score = score
            best_colors = colors.copy()

    # Local search refinement: try to increase colors by recoloring edges
    if best_colors is not None and time.time() < deadline - 3.0:
        current = best_colors.copy()
        curr_score, curr_viol = evaluate(current)

        # Try to recolor edges to new colors
        used = set(current)
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            # Pick a random edge
            e = rng.randint(0, NUM_EDGES - 1)
            old_c = current[e]
            # Try to assign a new color
            # Find smallest color not in used that is valid
            for new_c in range(MAX_COLORS):
                if new_c in used:
                    continue
                # Check if assigning new_c to e is valid
                valid = True
                for (ci, pos) in edge_to_cycles[e]:
                    e0, e1, e2, e3 = cycle_edges[ci]
                    other_edges = [e0, e1, e2, e3]
                    other_edges.pop(pos)
                    other_colors = [current[oe] for oe in other_edges]
                    all_colors = other_colors + [new_c]
                    if len(set(all_colors)) == 4:
                        valid = False
                        break
                if valid:
                    current[e] = new_c
                    used.add(new_c)
                    curr_score, curr_viol = evaluate(current)
                    if curr_viol == 0 and curr_score > best_score:
                        best_score = curr_score
                        best_colors = current.copy()
                    break

    # Final verification
    if best_colors is not None:
        final_score, final_viol = evaluate(best_colors)
        if final_viol == 0:
            return best_colors
        else:
            # Fallback: all zeros
            return [0] * NUM_EDGES
    else:
        return [0] * NUM_EDGES


if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
