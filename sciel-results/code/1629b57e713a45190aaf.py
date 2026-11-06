import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    best_edges = None
    best_n = 0
    best_count = float('inf')

    def get_vertex_masks(n):
        """Precompute M0[v] and M1[v] bitmasks for all vertices."""
        total_colorings = 1 << n
        M0 = [0] * n
        M1 = [0] * n
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
        return M0, M1

    def compute_edge_masks(n, all_edges, M0, M1):
        """Compute monochromatic mask for each edge."""
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
        return edge_masks

    def is_cover_complete(covered, all_ones):
        """Check if covered equals all_ones."""
        return covered == all_ones

    def greedy_set_cover(all_edges, edge_masks, all_ones, deadline, shuffle_seed=None):
        """Greedy set cover with optional shuffling. Returns list of edge indices or None."""
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            remaining = list(range(len(all_edges)))
            rng.shuffle(remaining)
        else:
            remaining = list(range(len(all_edges)))

        covered = 0
        selected = []
        removed = [False] * len(all_edges)

        while covered != all_ones:
            if time.time() > deadline - 2:
                return None

            best_gain = -1
            best_idx = -1

            for idx in remaining:
                if removed[idx]:
                    continue
                gain = (edge_masks[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    if gain == 0:
                        break

            if best_gain <= 0:
                return None  # Failed to cover all

            selected.append(best_idx)
            covered |= edge_masks[best_idx]
            removed[best_idx] = True
            # Remove from remaining for efficiency
            remaining = [i for i in remaining if i != best_idx]

            if len(selected) > 200:
                return None

        return selected

    def minimize_edges(selected_indices, all_edges, edge_masks, all_ones, deadline):
        """Try to remove redundant edges from the selected set using coverage counts."""
        if not selected_indices:
            return []

        # num_colorings is 2^n, which is the bit length of all_ones (since all_ones is 2^(2^n)-1? No)
        # all_ones = (1 << (1 << n)) - 1.
        # bit_length of (1 << K) - 1 is K.
        # Here K = 1 << n.
        num_colorings = all_ones.bit_length()
        
        # Initialize coverage counts
        coverage_count = [0] * num_colorings
        
        # Build initial counts
        for idx in selected_indices:
            mask = edge_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                coverage_count[c] += 1
                m ^= lsb

        final_indices = list(selected_indices)
        changed = True

        while changed and len(final_indices) > 1:
            if time.time() > deadline - 2:
                break
            changed = False
            # Try multiple removal orders
            orders = [
                list(range(len(final_indices) - 1, -1, -1)),  # reverse
                list(range(len(final_indices))),  # forward
            ]
            # Add a random order
            rng = random.Random()
            random_order = list(range(len(final_indices)))
            rng.shuffle(random_order)
            orders.append(random_order)

            for order in orders:
                if time.time() > deadline - 2:
                    break

                for i in order:
                    if time.time() > deadline - 2:
                        break

                    if i >= len(final_indices):
                        continue

                    idx_to_remove = final_indices[i]
                    mask_to_remove = edge_masks[idx_to_remove]
                    
                    # Check if removable: all colorings covered by this edge must be covered by >1 edge
                    removable = True
                    m = mask_to_remove
                    while m:
                        lsb = m & -m
                        c = lsb.bit_length() - 1
                        if coverage_count[c] <= 1:
                            removable = False
                            break
                        m ^= lsb
                    
                    if removable:
                        # Remove it
                        final_indices.pop(i)
                        # Update counts
                        m = mask_to_remove
                        while m:
                            lsb = m & -m
                            c = lsb.bit_length() - 1
                            coverage_count[c] -= 1
                            m ^= lsb
                        changed = True
                        break  # Restart order loop

        return final_indices

    # Try n from 9 to 14
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break

        all_edges = list(combinations(range(n), 5))
        m_total = len(all_edges)

        if m_total < 10:
            continue

        # Precompute vertex masks
        M0, M1 = get_vertex_masks(n)
        all_ones = (1 << (1 << n)) - 1

        # Compute edge masks
        edge_masks = compute_edge_masks(n, all_edges, M0, M1)

        # Try multiple greedy restarts
        best_local = None
        best_local_count = float('inf')

        # Number of restarts: more for smaller n
        if n <= 10:
            num_restarts = 100
        elif n <= 12:
            num_restarts = 60
        else:
            num_restarts = 30

        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break

            # Deterministic first, then random
            if r == 0:
                seed = None
            else:
                seed = r * 12345 + n

            selected = greedy_set_cover(all_edges, edge_masks, all_ones, deadline, shuffle_seed=seed)

            if selected is not None:
                # Minimize
                minimized = minimize_edges(selected, all_edges, edge_masks, all_ones, deadline)

                if len(minimized) < best_local_count:
                    best_local_count = len(minimized)
                    best_local = minimized

                    # Early exit if we hit the known best
                    if best_local_count <= 51:
                        break

        if best_local is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(all_edges[i]) for i in best_local]

                if best_count <= 51:
                    # Found optimal or near-optimal, stop searching
                    break

    # Fallback: if nothing found, use complete hypergraph on small n
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}

if __name__ == '__main__':
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
