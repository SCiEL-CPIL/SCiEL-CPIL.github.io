"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = 14
    m = 35
    maximum = 10000

    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def collision_density(values):
        """Score a value set by how many edges share the same sum or product."""
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        sum_counts = {}
        prod_counts = {}
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        # Higher density = more collisions = better
        sum_density = sum(c * c for c in sum_counts.values())
        prod_density = sum(c * c for c in prod_counts.values())
        return sum_density + prod_density

    def hungarian_edge_select(values, K):
        """
        Use Hungarian algorithm to select m edges that maximize the number
        of edges whose sum and product are in the top-K most frequent groups.
        Returns the selected edges and the raw value.
        """
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        num_pairs = len(all_pairs)

        # Precompute sums and products
        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[(i, j)] = values[i] + values[j]
            pair_prods[(i, j)] = values[i] * values[j]

        # Find top-K sum groups and top-K product groups
        sum_counts = {}
        prod_counts = {}
        for i, j in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

        # Sort by frequency, take top K
        top_sums = set(sorted(sum_counts.keys(), key=lambda x: sum_counts[x], reverse=True)[:K])
        top_prods = set(sorted(prod_counts.keys(), key=lambda x: prod_counts[x], reverse=True)[:K])

        # Build cost matrix: we want to maximize the number of edges that hit
        # both top sums and top prods. Use Hungarian on a square matrix.
        # Cost for edge e: 0 if both sum and prod are in top sets, 1 otherwise
        # We want to minimize total cost = maximize hits.

        # Hungarian requires square matrix. We have num_pairs edges, need to pick m.
        # Pad to square: m x m assignment where we assign m "slots" to m edges.
        # Actually, we need to select m edges from num_pairs. This is an assignment
        # problem: assign each of m slots to a distinct edge.
        # Cost matrix is m x num_pairs, but Hungarian needs square.
        # Pad to num_pairs x num_pairs with infinite cost for unused slots.

        # Simpler: use m x m by padding. But num_pairs=91, m=35.
        # Let's do it properly: create a num_pairs x num_pairs cost matrix where
        # the first m rows are "slots" and remaining rows have infinite cost.
        # Actually, let me use a different approach.

        # Alternative: just use greedy with Hungarian for the final selection.
        # Or: formulate as min-cost max-flow. But let's keep it simple.

        # Let me use a simpler formulation:
        # Create cost matrix of size num_pairs x num_pairs
        # Row i, col j: cost of assigning edge i to slot j
        # For slot j < m: cost = 0 if edge i is "good", 1 otherwise
        # For slot j >= m: cost = large (won't be used)
        # But this doesn't work directly with Hungarian.

        # Let me use a different approach: iterative Hungarian
        # Or just use a priority-based selection with collision awareness.

        # Actually, let me reformulate:
        # We want to select m edges. The objective is max(distinct_sums, distinct_prods).
        # If we restrict to top-K sums and top-K prods, then distinct_sums <= K and distinct_prods <= K.
        # So raw_value <= K. We want the smallest K such that we can select m edges
        # all having sums in top-K and prods in top-K.

        # For a given K, check if we can select m edges from the "good" set
        # (edges where sum in top_sums AND prod in top_prods).
        # If |good_edges| >= m, we can select m of them.
        # But we also want to minimize the actual distinct counts, not just bound by K.

        # Let me use a different strategy:
        # 1. For a given K, find all edges with sum in top-K sums AND prod in top-K prods.
        # 2. If there are at least m such edges, select m of them (prefer those with
        #    higher collision frequency to minimize actual distinct counts).
        # 3. The raw value is at most K.

        # But this might be too restrictive. Let me try a relaxed version:
        # Allow edges where sum in top-K OR prod in top-K, and minimize the max.

        # Let me just implement a direct Hungarian-based selection:
        # Cost for selecting edge e = (1 if sum not in current_sums else 0) + (1 if prod not in current_prods else 0)
        # But this is sequential, not global.

        # For a global approach, let me use the following:
        # Assign each edge a "quality" score based on how many other edges share its sum and product.
        # Then select the m edges with highest quality scores, breaking ties by collision frequency.

        # Compute quality for each edge
        edge_quality = {}
        for i, j in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            q = sum_counts[s] + prod_counts[p]
            edge_quality[(i, j)] = q

        # Sort edges by quality (descending)
        sorted_edges = sorted(all_pairs, key=lambda e: edge_quality[e], reverse=True)

        # Select top m edges
        selected = sorted_edges[:m]

        # Compute actual raw value
        sums = set()
        prods = set()
        for i, j in selected:
            sums.add(pair_sums[(i, j)])
            prods.add(pair_prods[(i, j)])

        return selected, max(len(sums), len(prods))

    def hungarian_assignment(values, m):
        """
        Use Hungarian algorithm to assign m edges to m slots, minimizing the
        number of distinct sums and products.
        """
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        num_pairs = len(all_pairs)

        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[(i, j)] = values[i] + values[j]
            pair_prods[(i, j)] = values[i] * values[j]

        sum_counts = {}
        prod_counts = {}
        for i, j in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1

        # For Hungarian, we need a square cost matrix.
        # We have num_pairs edges and need to select m.
        # Create a num_pairs x num_pairs matrix where:
        # - Row i represents edge i
        # - Column j represents "slot" j (j < m) or "unused" (j >= m)
        # - Cost[i][j] for j < m: penalty if edge i introduces new sum/prod
        # - Cost[i][j] for j >= m: large penalty (discourage use)
        #
        # But the cost depends on which other edges are selected, so it's not static.
        #
        # Alternative: use a two-phase approach.
        # Phase 1: Select a candidate pool of "good" edges.
        # Phase 2: Use Hungarian to select m from the pool.

        # Let me try a simpler but effective approach:
        # Score each edge by (sum_frequency + prod_frequency) / (sum_frequency * prod_frequency)
        # This rewards edges that are in large collision groups.
        # Then select top m.

        edge_scores = {}
        for i, j in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            sf = sum_counts[s]
            pf = prod_counts[p]
            # Higher score = more collision-friendly
            score = sf * pf  # Product of frequencies
            edge_scores[(i, j)] = score

        sorted_edges = sorted(all_pairs, key=lambda e: edge_scores[e], reverse=True)
        selected = sorted_edges[:m]

        sums = set()
        prods = set()
        for i, j in selected:
            sums.add(pair_sums[(i, j)])
            prods.add(pair_prods[(i, j)])

        return selected, max(len(sums), len(prods))

    def local_search_edges(values, edges, max_time):
        """Local search to improve edge selection."""
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        edge_set = set(tuple(e) for e in edges)

        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[(i, j)] = values[i] + values[j]
            pair_prods[(i, j)] = values[i] * values[j]

        def compute_score(edgs):
            sums = set()
            prods = set()
            for i, j in edgs:
                sums.add(pair_sums[(i, j)])
                prods.add(pair_prods[(i, j)])
            return max(len(sums), len(prods))

        current_score = compute_score(edges)
        best_edges = list(edges)
        best_score = current_score
        deadline_ls = time.time() + max_time

        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            for idx in range(len(edges)):
                if time.time() > deadline_ls:
                    break
                old_edge = edges[idx]
                # Compute sums/prods without old_edge
                sums_wo = set()
                prods_wo = set()
                for e in edges:
                    if e != old_edge:
                        sums_wo.add(pair_sums[(tuple(e) if isinstance(e, list) else e)])
                        prods_wo.add(pair_prods[(tuple(e) if isinstance(e, list) else e)])

                for new_edge in all_pairs:
                    if new_edge in edge_set:
                        continue
                    new_s = pair_sums[new_edge]
                    new_p = pair_prods[new_edge]
                    new_sums = sums_wo | {new_s}
                    new_prods = prods_wo | {new_p}
                    new_score = max(len(new_sums), len(new_prods))

                    if new_score < best_score:
                        edges[idx] = list(new_edge)
                        edge_set.discard(tuple(old_edge))
                        edge_set.add(new_edge)
                        best_score = new_score
                        best_edges = list(edges)
                        improved = True
                        break

        return best_edges, best_score

    # Generate candidate value sets
    candidate_value_sets = []

    # APs
    for start in range(1, 5):
        candidate_value_sets.append(list(range(start, start + 14)))

    # Divisor-rich sets
    base_divisor = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidate_value_sets.append(base_divisor)
    for extra in [24, 30, 36, 40, 48, 60, 72, 90, 120]:
        s = list(base_divisor[:13]) + [extra]
        if len(set(s)) == 14 and all(1 <= x <= maximum for x in s):
            candidate_value_sets.append(sorted(s))

    # Random sets biased toward small values
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        # Sample from 1..30 with bias toward smaller
        vals = []
        for _ in range(14):
            r = random.random()
            if r < 0.5:
                v = random.randint(1, 10)
            elif r < 0.8:
                v = random.randint(11, 20)
            else:
                v = random.randint(21, 30)
            if v not in vals:
                vals.append(v)
        if len(vals) == 14:
            candidate_value_sets.append(sorted(vals))

    # Deduplicate
    seen = set()
    unique_sets = []
    for vs in candidate_value_sets:
        key = tuple(vs)
        if key not in seen:
            seen.add(key)
            unique_sets.append(vs)
    candidate_value_sets = unique_sets

    # Rank by collision density
    candidate_value_sets.sort(key=lambda vs: collision_density(vs), reverse=True)

    best_result = None
    best_score = float('inf')

    for vals in candidate_value_sets:
        if time.time() > deadline:
            break

        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 1.0:
            break

        # Use Hungarian-based selection
        edges, score = hungarian_assignment(vals, m)

        # Local search to improve
        ls_time = min(remaining * 0.2, 2.0)
        if ls_time > 0.1:
            edges, score = local_search_edges(vals, edges, ls_time)

        if score < best_score:
            best_score = score
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}

            if best_score <= 25:
                break

    # If no result, fallback
    if best_result is None:
        vals = list(range(1, 15))
        edges, score = hungarian_assignment(vals, m)
        best_result = {'values': vals, 'edges': [list(e) for e in edges]}

    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]

    return best_result
# EVOLVE-BLOCK-END
