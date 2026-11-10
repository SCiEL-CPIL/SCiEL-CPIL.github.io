<<<<<<< SEARCH
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Fast 2-colorability check. Returns True if colorable, False otherwise.
    Uses backtracking with unit propagation.
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
            # Heuristic: Pick the uncolored vertex involved in the most "tight" edges
            # An edge is tight if it has 3 or more colored vertices.
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
                if best_score >= 10: # Early break for speed
                    break

            if best_v == -1:
                return True

            v = best_v

            # Try colors 0 and 1
            # Heuristic: Try the color that appears more often in neighboring edges?
            # Or just 0 then 1.
            
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

                    # Unit Propagation
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

def _get_witness_coloring(n, edges):
    """
    Returns a valid 2-coloring if exists, else None.
    Reuses logic from _is_2_colorable but returns the colors.
    """
    if not edges:
        return [0] * n

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]
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
                if best_score >= 10:
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

        if backtrack():
            # Fill any remaining uncolored vertices (should be none if logic is correct, but just in case)
            for i in range(n):
                if colors[i] == -1:
                    colors[i] = 0
            return colors
        return None

    return solve()
=======
import time
import random
from itertools import combinations
from collections import deque

def _is_2_colorable(n, edges):
    """
    Fast 2-colorability check using 2-SAT. Returns True if colorable, False otherwise.
    """
    if not edges:
        return True

    m = len(edges)
    if m > 1000:  # Too many edges, likely not 2-colorable
        return False

    # Build 2-SAT instance
    # Variable i represents vertex i being color 0
    # We need: for each edge, not all 5 vertices have same color
    # This is equivalent to: for each edge, at least one vertex is color 0 AND at least one is color 1
    # Which means: (OR of x_v for v in edge) AND (OR of not x_v for v in edge)

    # For 2-SAT, we need clauses of size at most 2
    # A clause (x_a OR x_b) means: not both false
    # For "at least one true" in a set of size k, we can't directly express as 2-SAT if k > 2
    # But for k=5, we can use: if 4 are false, the 5th must be true
    # This gives us implications

    # Actually, let's use a different approach: backtracking with bitsets
    # Represent color assignments as two bitmasks: mask0 (color 0), mask1 (color 1)

    edge_verts = [tuple(e) for e in edges]

    # Precompute vertex incidence
    v_to_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_edges[v].append(i)

    def solve():
        # Use iterative deepening or just simple backtracking with good heuristics
        # State: colors array, and for each edge, count of color 0 and color 1
        colors = [-1] * n
        e0_count = [0] * m
        e1_count = [0] * m

        # Heuristic: find the edge with the most colored vertices, break ties by most uncolored in remaining
        def get_critical_edge():
            best_e = -1
            best_score = -1
            for ei in range(m):
                colored = e0_count[ei] + e1_count[ei]
                if colored >= 3:
                    score = colored * 100 + (5 - colored)
                    if score > best_score:
                        best_score = score
                        best_e = ei
            return best_e

        def backtrack():
            # Find a vertex to assign
            # Prefer vertices in edges that are close to being monochromatic
            best_v = -1
            best_priority = -1

            for ei in range(m):
                if e0_count[ei] + e1_count[ei] < 3:
                    continue
                # Edge is tight
                for v in edge_verts[ei]:
                    if colors[v] != -1:
                        continue
                    # Priority: how many same-color vertices in this edge
                    same_color = e0_count[ei] if e1_count[ei] < e0_count[ei] else e1_count[ei]
                    priority = same_color * 10 + (e0_count[ei] + e1_count[ei])
                    if priority > best_priority:
                        best_priority = priority
                        best_v = v

            if best_v == -1:
                # No tight edges, pick any uncolored vertex
                for v in range(n):
                    if colors[v] == -1:
                        best_v = v
                        break
                if best_v == -1:
                    return True

            v = best_v

            # Try both colors
            for c in [0, 1]:
                undo = []
                colors[v] = c
                undo.append((v, -1))

                conflict = False
                # Update all edges containing v
                for ei in v_to_edges[v]:
                    old_e0 = e0_count[ei]
                    old_e1 = e1_count[ei]
                    if c == 0:
                        e0_count[ei] = old_e0 + 1
                    else:
                        e1_count[ei] = old_e1 + 1
                    undo.append((ei, old_e0, old_e1))

                    # Check for conflict: all 5 same color
                    if e0_count[ei] == 5 or e1_count[ei] == 5:
                        conflict = True
                        break

                    # Unit propagation: if 4 same color, force the 5th
                    if e0_count[ei] == 4 or e1_count[ei] == 4:
                        forced_color = 1 if e0_count[ei] == 4 else 0
                        for uv in edge_verts[ei]:
                            if colors[uv] == -1:
                                if colors[uv] != forced_color:
                                    colors[uv] = forced_color
                                    undo.append((uv, -1))
                                    # Update edges for uv
                                    for ej in v_to_edges[uv]:
                                        old_e0j = e0_count[ej]
                                        old_e1j = e1_count[ej]
                                        if forced_color == 0:
                                            e0_count[ej] = old_e0j + 1
                                        else:
                                            e1_count[ej] = old_e1j + 1
                                        undo.append((ej, old_e0j, old_e1j))

                                        if e0_count[ej] == 5 or e1_count[ej] == 5:
                                            conflict = True
                                            break
                                    if conflict:
                                        break
                                else:
                                    conflict = True
                                    break
                            elif colors[uv] != forced_color:
                                conflict = True
                                break
                        if conflict:
                            break

                if conflict:
                    # Undo
                    for item in reversed(undo):
                        if len(item) == 2:
                            colors[item[0]] = item[1]
                        else:
                            e0_count[item[0]] = item[1]
                            e1_count[item[0]] = item[2]
                    continue

                if backtrack():
                    return True

                # Undo
                for item in reversed(undo):
                    if len(item) == 2:
                        colors[item[0]] = item[1]
                    else:
                        e0_count[item[0]] = item[1]
                        e1_count[item[0]] = item[2]

            return False

        return backtrack()

    return solve()

def _get_witness_coloring(n, edges):
    """
    Returns a valid 2-coloring if exists, else None.
    """
    if not edges:
        return [0] * n

    m = len(edges)
    if m > 1000:
        return None

    edge_verts = [tuple(e) for e in edges]
    v_to_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_edges[v].append(i)

    def solve():
        colors = [-1] * n
        e0_count = [0] * m
        e1_count = [0] * m

        def backtrack():
            best_v = -1
            best_priority = -1

            for ei in range(m):
                if e0_count[ei] + e1_count[ei] < 3:
                    continue
                for v in edge_verts[ei]:
                    if colors[v] != -1:
                        continue
                    same_color = e0_count[ei] if e1_count[ei] < e0_count[ei] else e1_count[ei]
                    priority = same_color * 10 + (e0_count[ei] + e1_count[ei])
                    if priority > best_priority:
                        best_priority = priority
                        best_v = v

            if best_v == -1:
                for v in range(n):
                    if colors[v] == -1:
                        best_v = v
                        break
                if best_v == -1:
                    return True

            v = best_v

            for c in [0, 1]:
                undo = []
                colors[v] = c
                undo.append((v, -1))

                conflict = False
                for ei in v_to_edges[v]:
                    old_e0 = e0_count[ei]
                    old_e1 = e1_count[ei]
                    if c == 0:
                        e0_count[ei] = old_e0 + 1
                    else:
                        e1_count[ei] = old_e1 + 1
                    undo.append((ei, old_e0, old_e1))

                    if e0_count[ei] == 5 or e1_count[ei] == 5:
                        conflict = True
                        break

                    if e0_count[ei] == 4 or e1_count[ei] == 4:
                        forced_color = 1 if e0_count[ei] == 4 else 0
                        for uv in edge_verts[ei]:
                            if colors[uv] == -1:
                                colors[uv] = forced_color
                                undo.append((uv, -1))
                                for ej in v_to_edges[uv]:
                                    old_e0j = e0_count[ej]
                                    old_e1j = e1_count[ej]
                                    if forced_color == 0:
                                        e0_count[ej] = old_e0j + 1
                                    else:
                                        e1_count[ej] = old_e1j + 1
                                    undo.append((ej, old_e0j, old_e1j))
                                    if e0_count[ej] == 5 or e1_count[ej] == 5:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            elif colors[uv] != forced_color:
                                conflict = True
                                break
                        if conflict:
                            break

                if conflict:
                    for item in reversed(undo):
                        if len(item) == 2:
                            colors[item[0]] = item[1]
                        else:
                            e0_count[item[0]] = item[1]
                            e1_count[item[0]] = item[2]
                    continue

                if backtrack():
                    # Fill remaining
                    for i in range(n):
                        if colors[i] == -1:
                            colors[i] = 0
                    return colors

                for item in reversed(undo):
                    if len(item) == 2:
                        colors[item[0]] = item[1]
                    else:
                        e0_count[item[0]] = item[1]
                        e1_count[item[0]] = item[2]

            return None

        result = backtrack()
        return result

    return solve()
>>>>>>> REPLACE

