import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def _elapsed():
        return time.time() - start_time

    def is_2_colorable_fast(n, edges):
        if not edges:
            return True

        edge_adj = [[] for _ in range(n)]
        for idx, e in enumerate(edges):
            for v in e:
                edge_adj[v].append(idx)

        assignments = [-1] * n
        count_0 = [0] * len(edges)
        count_1 = [0] * len(edges)

        def backtrack():
            # Find most constrained unassigned vertex
            v = -1
            best_constraints = -1
            for i in range(n):
                if assignments[i] == -1:
                    # Count how many edges have 4+ assigned vertices of one color
                    constraints = 0
                    for e_idx in edge_adj[i]:
                        c0 = count_0[e_idx]
                        c1 = count_1[e_idx]
                        if c0 >= 4 or c1 >= 4:
                            constraints += 1
                    if constraints > best_constraints:
                        best_constraints = constraints
                        v = i
                    if best_constraints > 0:
                        # Early exit if we found a highly constrained vertex
                        pass

            if v == -1:
                return True

            # Try both colors
            for color in (0, 1):
                other = 1 - color
                # Check if valid
                valid = True
                affected = []
                for e_idx in edge_adj[v]:
                    if color == 0:
                        if count_0[e_idx] == 4:
                            valid = False
                            break
                        affected.append((e_idx, 0))
                    else:
                        if count_1[e_idx] == 4:
                            valid = False
                            break
                        affected.append((e_idx, 1))

                if not valid:
                    continue

                # Apply
                assignments[v] = color
                for e_idx, _ in affected:
                    if color == 0:
                        count_0[e_idx] += 1
                    else:
                        count_1[e_idx] += 1

                if backtrack():
                    # Undo before return
                    assignments[v] = -1
                    for e_idx, _ in affected:
                        if color == 0:
                            count_0[e_idx] -= 1
                        else:
                            count_1[e_idx] -= 1
                    return True

                # Undo
                assignments[v] = -1
                for e_idx, _ in affected:
                    if color == 0:
                        count_0[e_idx] -= 1
                    else:
                        count_1[e_idx] -= 1

            return False

        return backtrack()

    def greedy_set_cover(n):
        total_colorings = 1 << n
        all_edges = list(combinations(range(n), 5))

        # Precompute which colorings each edge kills
        # Edge kills coloring if edge is monochromatic
        kill_masks = []
        for e in all_edges:
            e_mask = 0
            for v in e:
                e_mask |= (1 << v)
            km = 0
            for c in range(total_colorings):
                if (c & e_mask) == 0 or (c & e_mask) == e_mask:
                    km |= (1 << c)
            kill_masks.append(km)

        remaining = (1 << total_colorings) - 1
        selected_edges = []
        selected_indices = set()

        while remaining:
            if _elapsed() > deadline - 10:
                return None

            best_idx = -1
            best_count = 0
            best_new_remaining = remaining

            for i, km in enumerate(kill_masks):
                if i in selected_indices:
                    continue
                new_killed = remaining & km
                if new_killed:
                    cnt = new_killed.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
                        best_new_remaining = remaining & ~km

            if best_idx == -1:
                return None

            selected_edges.append(all_edges[best_idx])
            selected_indices.add(best_idx)
            remaining = best_new_remaining

        return selected_edges

    def local_search_delete(n, edges):
        current_edges = edges[:]
        changed = True
        while changed and _elapsed() < deadline - 5:
            changed = False
            # Try deleting each edge, in random order for diversity
            import random
            indices = list(range(len(current_edges)))
            random.shuffle(indices)
            for i in indices:
                test_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable_fast(n, test_edges):
                    current_edges = test_edges
                    changed = True
                    break

        return current_edges

    best_edges = None
    best_n = 0
    best_count = float('inf')

    # Try n from 9 to 14
    for n in range(9, 15):
        if _elapsed() > deadline - 15:
            break

        cand_edges = greedy_set_cover(n)
        if cand_edges is None:
            continue

        improved_edges = local_search_delete(n, cand_edges)

        if len(improved_edges) < best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
            best_n = n

    if best_edges is None:
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    if is_2_colorable_fast(best_n, best_edges):
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    final_edges = [list(e) for e in best_edges]
    return {'n': best_n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
