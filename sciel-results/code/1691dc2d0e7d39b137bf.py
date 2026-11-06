import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n = 20
    num_colors = 20

    # Precompute all 4-cycles and edge-to-cycle mappings
    all_cycles = []
    edge_to_cycles = defaultdict(list)

    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = [(min(e), max(e)) for e in cycle]
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Track forbidden colors per edge for fast lookups
    # edge_forbidden[e] = set of colors that would violate a 4-cycle constraint
    def compute_forbidden(e, edge_colors):
        """Compute the set of colors that are forbidden for edge e given current coloring."""
        forbidden = set()
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if len(set(other_colors)) < 3:
                # Cycle already broken; adding e with any color won't fix it,
                # but it also doesn't create a NEW violation from a valid state.
                # However, the cycle will remain non-rainbow, so e cannot be added
                # in a way that makes this cycle rainbow. Mark all colors forbidden
                # to prevent adding e into a broken cycle.
                forbidden.update(range(num_colors))
            else:
                forbidden.update(other_colors)
        return forbidden

    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None. Uses forbidden-set for speed."""
        forbidden = compute_forbidden(e, edge_colors)
        if len(forbidden) >= num_colors:
            return None
        # Build list of available colors, shuffled for diversity
        available = [c for c in range(num_colors) if c not in forbidden]
        random.shuffle(available)
        return available[0] if available else None

    def get_conflicts(edge_colors):
        """Return dict edge -> number of conflicting cycles it participates in."""
        conflict_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_count[e] += 1
        return conflict_count

    def repair(edge_colors, max_steps=500):
        """Remove edges to eliminate all conflicts, prioritizing high-conflict edges."""
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            conflict_count = get_conflicts(edge_colors)
            if not conflict_count:
                break
            max_count = max(conflict_count.values())
            candidates = [e for e, c in conflict_count.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            del edge_colors[e_to_remove]

    def polish(edge_colors):
        """High-precision polish pass: greedily re-add every missing edge, then a
        targeted recolor-then-add sweep."""
        # Pass A: greedy add of all missing edges (multiple sweeps for cascading)
        for _ in range(5):
            if time.time() > deadline:
                break
            progress = False
            for e in all_edges:
                if e in edge_colors:
                    continue
                c = find_valid_color(e, edge_colors)
                if c is not None:
                    edge_colors[e] = c
                    progress = True
            if not progress:
                break

        # Pass B: recolor one edge to free a color, then re-sweep additions
        for e_rec in list(edge_colors.keys()):
            if time.time() > deadline:
                break
            old_c = edge_colors[e_rec]
            for new_c in random.sample(range(num_colors), min(8, num_colors - 1)):
                if new_c == old_c:
                    continue
                del edge_colors[e_rec]
                if can_add_edge(e_rec, new_c, edge_colors):
                    edge_colors[e_rec] = new_c
                    # quick re-sweep
                    added = False
                    for e in all_edges:
                        if e not in edge_colors:
                            c = find_valid_color(e, edge_colors)
                            if c is not None:
                                edge_colors[e] = c
                                added = True
                    if added:
                        break  # keep this recolor
                else:
                    edge_colors[e_rec] = old_c
                    break
        return edge_colors

    def local_search(edge_colors, max_no_improve=800):
        """Local search with add / recolor / swap / double-swap moves and
        incremental missing-edge tracking."""
        best_count = len(edge_colors)
        best_state = dict(edge_colors)
        no_improve = 0
        temp = 1.0
        cooling = 0.999

        # Incremental missing set
        missing_set = set(all_edges) - set(edge_colors.keys())

        def save_if_better():
            nonlocal best_count, best_state
            if len(edge_colors) > best_count:
                best_count = len(edge_colors)
                best_state = dict(edge_colors)

        while time.time() < deadline:
            improved = False

            # Move 1: add a missing edge
            if missing_set:
                e_add = random.choice(tuple(missing_set))
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    missing_set.discard(e_add)
                    save_if_better()
                    no_improve = 0
                    continue

            # Move 2: recolor an existing edge
            if edge_colors:
                e_rec = random.choice(tuple(edge_colors.keys()))
                old_c = edge_colors[e_rec]
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c:
                    del edge_colors[e_rec]
                    if can_add_edge(e_rec, new_c, edge_colors):
                        edge_colors[e_rec] = new_c
                    else:
                        edge_colors[e_rec] = old_c

            # Move 3: single swap (remove one, add one, try to re-add removed)
            if missing_set and edge_colors and random.random() < 0.5:
                e_remove = random.choice(tuple(edge_colors.keys()))
                e_add = random.choice(tuple(missing_set))
                del edge_colors[e_remove]
                missing_set.add(e_remove)
                c_add = find_valid_color(e_add, edge_colors)
                if c_add is not None:
                    edge_colors[e_add] = c_add
                    missing_set.discard(e_add)
                    save_if_better()
                    c_rem = find_valid_color(e_remove, edge_colors)
                    if c_rem is not None:
                        edge_colors[e_remove] = c_rem
                        missing_set.discard(e_remove)
                        save_if_better()
                        no_improve = 0
                        improved = True
                    # else: swapped in a new edge, count stays same but state changed
                else:
                    c_rem = find_valid_color(e_remove, edge_colors)
                    if c_rem is not None:
                        edge_colors[e_remove] = c_rem
                        missing_set.discard(e_remove)

            # Move 4: double swap (remove 2, add 2, try to re-add)
            if len(missing_set) >= 2 and len(edge_colors) >= 2 and random.random() < 0.2:
                existing_list = list(edge_colors.keys())
                missing_list = list(missing_set)
                e1, e2 = random.sample(existing_list, 2)
                e3, e4 = random.sample(missing_list, 2)
                del edge_colors[e1]
                del edge_colors[e2]
                missing_set.add(e1)
                missing_set.add(e2)
                c3 = find_valid_color(e3, edge_colors)
                c4 = find_valid_color(e4, edge_colors)
                if c3 is not None and c4 is not None:
                    edge_colors[e3] = c3
                    edge_colors[e4] = c4
                    missing_set.discard(e3)
                    missing_set.discard(e4)
                    save_if_better()
                    c1 = find_valid_color(e1, edge_colors)
                    c2 = find_valid_color(e2, edge_colors)
                    if c1 is not None:
                        edge_colors[e1] = c1
                        missing_set.discard(e1)
                    if c2 is not None:
                        edge_colors[e2] = c2
                        missing_set.discard(e2)
                    save_if_better()
                    no_improve = 0
                    improved = True
                else:
                    # revert
                    c1 = find_valid_color(e1, edge_colors)
                    c2 = find_valid_color(e2, edge_colors)
                    if c1 is not None:
                        edge_colors[e1] = c1
                        missing_set.discard(e1)
                    if c2 is not None:
                        edge_colors[e2] = c2
                        missing_set.discard(e2)

            if not improved:
                no_improve += 1
                temp *= cooling
            if no_improve > max_no_improve:
                break

        return best_state

    # Multiple restarts
    best_result = None
    best_count = 0

    elapsed = time.time() - start_time
    remaining = deadline - start_time
    num_restarts = max(2, int(remaining / 18))
    num_restarts = min(num_restarts, 8)

    for restart in range(num_restarts):
        if time.time() > deadline - 20:  # reserve 20s for final polish
            break

        edge_colors = {}
        # Initialize with a moderately dense random graph
        p_init = 0.7 + 0.05 * random.random()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p_init:
                    edge_colors[(u, v)] = random.randint(0, num_colors - 1)

        # Phase 1: repair
        repair(edge_colors, max_steps=800)

        # Phase 2: local search
        edge_colors = local_search(edge_colors, max_no_improve=600)

        # Phase 3: polish
        if time.time() < deadline - 15:
            edge_colors = polish(edge_colors)
            # ensure still valid
            repair(edge_colors, max_steps=200)

        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_result = dict(edge_colors)

    # Final global polish on the best state
    if best_result is not None:
        if time.time() < deadline - 5:
            best_result = polish(best_result)
            repair(best_result, max_steps=300)
            # re-polish after repair
            if time.time() < deadline - 2:
                best_result = polish(best_result)

    if best_result is None:
        return []

    return [[u, v, c] for (u, v), c in best_result.items()]
