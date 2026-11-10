import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Fast 2-colorability check. Returns True if colorable, False otherwise.
    Uses backtracking with unit propagation and proper state save/restore.
    """
    if not edges:
        return True

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]

    # v_to_e_mask[v] = bitmask of edges containing v
    v_to_e_mask = [0] * n
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_e_mask[v] |= (1 << i)

    def solve():
        colors = [-1] * n
        e0 = [0] * m
        e1 = [0] * m

        def backtrack():
            best_v = -1
            best_score = -1
            for v in range(n):
                if colors[v] != -1:
                    continue
                score = 0
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    if e0[ei] + e1[ei] >= 3:
                        score += 1
                    mm ^= lb
                if score > best_score:
                    best_score = score
                    best_v = v
                if best_score >= 8:
                    break

            if best_v == -1:
                return True

            v = best_v

            for try_color in [0, 1]:
                undo_log = []
                colors[v] = try_color
                undo_log.append(('v', v, -1))

                conflict = False
                queue = []
                mm = v_to_e_mask[v]
                while mm:
                    lb = mm & (-mm)
                    ei = lb.bit_length() - 1
                    queue.append(ei)
                    mm ^= lb

                visited_edges = set(queue)
                qi = 0
                while qi < len(queue):
                    ei = queue[qi]
                    qi += 1
                    verts = edge_verts[ei]
                    c0 = 0
                    c1 = 0
                    uncolored = []
                    for uv in verts:
                        cv = colors[uv]
                        if cv == 0:
                            c0 += 1
                        elif cv == 1:
                            c1 += 1
                        else:
                            uncolored.append(uv)

                    old_e0 = e0[ei]
                    old_e1 = e1[ei]
                    e0[ei] = c0
                    e1[ei] = c1
                    undo_log.append(('e0', ei, old_e0))
                    undo_log.append(('e1', ei, old_e1))

                    if c0 == 5 or c1 == 5:
                        conflict = True
                        break

                    if c0 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 1
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 0:
                                conflict = True
                                break
                        if conflict:
                            break
                    elif c1 == 4:
                        for uv in uncolored:
                            if colors[uv] == -1:
                                colors[uv] = 0
                                undo_log.append(('v', uv, -1))
                                mm2 = v_to_e_mask[uv]
                                while mm2:
                                    lb2 = mm2 & (-mm2)
                                    ej = lb2.bit_length() - 1
                                    if ej not in visited_edges:
                                        queue.append(ej)
                                        visited_edges.add(ej)
                                    mm2 ^= lb2
                            elif colors[uv] == 1:
                                conflict = True
                                break
                        if conflict:
                            break

                if not conflict:
                    if backtrack():
                        return True

                for item in reversed(undo_log):
                    if item[0] == 'v':
                        colors[item[1]] = item[2]
                    elif item[0] == 'e0':
                        e0[item[1]] = item[2]
                    elif item[0] == 'e1':
                        e1[item[1]] = item[2]

            return False

        return backtrack()

    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    best_edges_count = float('inf')
    best_result = None

    for n in [9, 10, 11, 12, 13]:
        if time.time() > deadline:
            break

        all_possible = list(combinations(range(n), 5))

        for trial in range(15):
            if time.time() > deadline:
                break

            random.shuffle(all_possible)

            for m_start in [35, 45, 55, 70]:
                if time.time() > deadline:
                    break

                edges = [list(e) for e in all_possible[:m_start]]

                if _is_2_colorable(n, edges):
                    continue

                current_edges = edges[:]
                random.shuffle(current_edges)

                # Iterative peeling to fixed point
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    i = 0
                    while i < len(current_edges) and time.time() < deadline:
                        temp_edges = current_edges[:i] + current_edges[i + 1:]
                        if not _is_2_colorable(n, temp_edges):
                            current_edges = temp_edges
                            improved = True
                        else:
                            i += 1

                # Second pass with different order
                if time.time() < deadline and len(current_edges) > 25:
                    random.shuffle(current_edges)
                    i = 0
                    while i < len(current_edges) and time.time() < deadline:
                        temp_edges = current_edges[:i] + current_edges[i + 1:]
                        if not _is_2_colorable(n, temp_edges):
                            current_edges = temp_edges
                        else:
                            i += 1

                if len(current_edges) < best_edges_count:
                    best_edges_count = len(current_edges)
                    best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                    if best_edges_count <= 30:
                        break

            if best_edges_count <= 30:
                break

    if best_result is None:
        edges = [list(e) for e in combinations(range(9), 5)]
        return {'n': 9, 'edges': edges}

    return best_result
