"""Cochromatic partition solver: bitmask lookup + iterative deepening over set partitions."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def _build_graph():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj


def construct():
    n, adj = _build_graph()
    deadline = time.time() + 80

    # Precompute neighbor bitmasks
    adj_mask = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        adj_mask[i] = m

    full_mask = (1 << n) - 1

    # Precompute: for every subset, is it a clique? is it independent?
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True

    for mask in range(1, 1 << n):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True

    def valid_part(mask):
        return is_clique[mask] or is_indep[mask]

    # Greedy upper bound (returns list of bitmask parts)
    def greedy_partition():
        parts = []
        remaining = full_mask
        while remaining:
            placed = False
            for i in range(len(parts)):
                for v in range(n):
                    if (remaining >> v) & 1:
                        new_mask = parts[i] | (1 << v)
                        if valid_part(new_mask):
                            parts[i] = new_mask
                            remaining ^= (1 << v)
                            placed = True
                            break
                if placed:
                    break
            if not placed:
                v = max((v for v in range(n) if (remaining >> v) & 1),
                        key=lambda x: bin(adj_mask[x]).count('1'))
                parts.append(1 << v)
                remaining ^= (1 << v)
        return parts

    best_parts_masks = greedy_partition()
    best_count = len(best_parts_masks)

    # Iterative deepening search
    def search_k_parts(k):
        """Returns list of k non-zero masks if found, else None."""
        result = [0] * k

        def backtrack(unassigned):
            if time.time() > deadline:
                return False
            if unassigned == 0:
                return True
            lb = unassigned & (-unassigned)
            v = lb.bit_length() - 1
            v_mask = 1 << v
            new_unassigned = unassigned ^ v_mask

            for i in range(k):
                if result[i] == 0:
                    # Only start new part if no gaps
                    if i > 0 and result[i - 1] == 0:
                        break
                    if valid_part(v_mask):
                        result[i] = v_mask
                        if backtrack(new_unassigned):
                            return True
                        result[i] = 0
                else:
                    new_mask = result[i] | v_mask
                    if valid_part(new_mask):
                        result[i] = new_mask
                        if backtrack(new_unassigned):
                            return True
                        result[i] ^= v_mask
            return False

        if backtrack(full_mask):
            return [m for m in result if m != 0]
        return None

    for k in range(best_count - 1, 0, -1):
        if time.time() > deadline:
            break
        found = search_k_parts(k)
        if found is not None:
            best_parts_masks = found
            best_count = k
            break

    # Convert bitmask parts to lists of ints
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    # Validate
    def validate(parts):
        seen = 0
        for part in parts:
            mask = 0
            for v in part:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
                if (seen >> v) & 1:
                    return False
                seen |= (1 << v)
                mask |= (1 << v)
            if not valid_part(mask):
                return False
        return seen == full_mask

    result = [mask_to_list(m) for m in best_parts_masks]

    if not validate(result):
        # Fallback to greedy
        result = [mask_to_list(m) for m in greedy_partition()]

    # Ensure all elements are plain Python ints
    final = []
    for part in result:
        final.append([int(v) for v in part])
    return final

# EVOLVE-BLOCK-END
