"""Cochromatic partition: precompute valid part masks, branch on smallest vertex."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 80.0  # safety margin under 90s wall-clock


def construct():
    start = time.time()
    n = N
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    all_mask = (1 << n) - 1

    # Precompute: for each mask, is it a clique or independent set?
    # valid[mask] = True if mask induces a clique or independent set
    valid = bytearray(1 << n)
    valid[0] = 0
    for mask in range(1, 1 << n):
        verts = []
        tmp = mask
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            verts.append(v)
            tmp ^= lsb
        if len(verts) <= 1:
            valid[mask] = 1
            continue
        is_clique = True
        is_indep = True
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if adj[verts[i]] & (1 << verts[j]):
                    is_indep = False
                else:
                    is_clique = False
            if not is_clique and not is_indep:
                break
        if is_clique or is_indep:
            valid[mask] = 1

    # For each vertex v, precompute list of valid masks containing v, sorted by size desc
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if valid[mask]:
            tmp = mask
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                masks_by_vertex[v].append(mask)
                tmp ^= lsb
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)

    # Compute max clique size and max independent set size for lower bound
    max_clique = 0
    max_indep = 0
    for mask in range(1, 1 << n):
        if valid[mask]:
            bc = mask.bit_count()
            # Check if it's a clique
            verts = []
            tmp = mask
            while tmp:
                lsb = tmp & (-tmp)
                verts.append(lsb.bit_length() - 1)
                tmp ^= lsb
            is_clique = True
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    if not (adj[verts[i]] & (1 << verts[j])):
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                if bc > max_clique:
                    max_clique = bc
            else:
                if bc > max_indep:
                    max_indep = bc

    max_part_size = max(max_clique, max_indep, 1)

    # Greedy initial upper bound
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))

    def _greedy():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                clique = True
                indep = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        if (trial[i], trial[j]) in es:
                            indep = False
                        else:
                            clique = False
                        if not clique and not indep:
                            break
                    if not clique and not indep:
                        break
                if clique or indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    greedy_parts = _greedy()
    best_k = len(greedy_parts)
    best_solution = [list(p) for p in greedy_parts]

    node_count = [0]
    found_solution = [False]

    def search(rem_mask, chosen, count, max_count):
        nonlocal best_k, best_solution, found_solution
        if time.time() - start > DEADLINE:
            return False
        node_count[0] += 1

        if rem_mask == 0:
            if count < best_k:
                best_k = count
                parts = []
                for m in chosen:
                    parts.append([i for i in range(n) if (m >> i) & 1])
                best_solution = parts
                found_solution = True
            return True

        if count >= max_count:
            return False

        # Lower bound
        rem_size = rem_mask.bit_count()
        lb = (rem_size + max_part_size - 1) // max_part_size
        if count + lb >= max_count:
            return False

        # Pick smallest vertex in remaining
        lsb = rem_mask & (-rem_mask)
        v = lsb.bit_length() - 1

        # Try valid parts containing v that are subset of rem_mask
        for part_mask in masks_by_vertex[v]:
            if part_mask & ~rem_mask:
                continue
            chosen.append(part_mask)
            if search(rem_mask ^ part_mask, chosen, count + 1, max_count):
                chosen.pop()
                return True
            chosen.pop()
        return False

    # Try k from 1 up to best_k - 1
    for k in range(1, best_k):
        if time.time() - start > DEADLINE:
            break
        found_solution[0] = False
        if search(all_mask, [], 0, k):
            if found_solution[0]:
                break

    # Build final result
    result = []
    for part in best_solution:
        if part:
            result.append([int(v) for v in part])

    # Validate
    flat = []
    for part in result:
        flat.extend(part)
    if sorted(flat) != list(range(n)) or len(result) == 0:
        result = [[int(v) for v in p] for p in _greedy()]

    return result
