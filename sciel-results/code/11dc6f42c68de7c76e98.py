import time
import random
from itertools import combinations

# Precompute all possible triples on 0..15
ALL_TRIPLES = list(combinations(range(16), 3))
TRIPLE_MASKS = []
for t in ALL_TRIPLES:
    mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
TRIPLE_MASKS.append(mask)

# Precompute pair mask for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    pairs = set()
    for i in range(3):
        for j in range(i + 1, 3):
            pairs.add((t[i], t[j]))
    TRIPLE_PAIRS.append(pairs)

# Precompute subset masks for independence checking
SUBSET_MASKS = {}
for k in range(5, 9):
    for subset in combinations(range(16), k):
        smask = 0
        for v in subset:
            smask |= (1 << v)
        SUBSET_MASKS[(k, subset)] = smask

def calculate_alpha_fast(edge_masks):
    """
    Find the largest independent set size using precomputed masks.
    edge_masks is a list of integer masks for the edges.
    """
    if not edge_masks:
        return 16

    # Check from k=8 down to 5
    for k in range(8, 4, -1):
        for subset in combinations(range(16), k):
            smask = SUBSET_MASKS[(k, subset)]
            is_independent = True
            for emask in edge_masks:
                if (emask & smask) == emask:
                    is_independent = False
                    break
            if is_independent:
                return k
    return 4

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_edges = []
    best_alpha = 16

    # Try multiple random restarts
    for restart in range(50):
        if time.time() > deadline - 3:
            break

        # Initial Construction: Random Greedy
        shuffled_triples = ALL_TRIPLES[:]
        random.shuffle(shuffled_triples)
        edges = []
        edges_set = set()
        used_pairs = set()

        for t in shuffled_triples:
            if len(edges) >= 40:
                break
            valid = True
            for p in TRIPLE_PAIRS[ALL_TRIPLES.index(t)]:
                if p in used_pairs or (p[1], p[0]) in used_pairs:
                    valid = False
                    break
            if valid:
                edges.append(t)
                edges_set.add(t)
                for p in TRIPLE_PAIRS[ALL_TRIPLES.index(t)]:
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))

        # Convert to masks for fast alpha calculation
        edge_masks = []
        for e in edges:
            mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
            edge_masks.append(mask)

        current_alpha = calculate_alpha_fast(edge_masks)

        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_edges = edges[:]

        if best_alpha <= 5:
            break

        # Local Search
        if time.time() > deadline - 10:
            continue

        # Simple local search: try to swap edges to reduce alpha
        for step in range(100):
            if time.time() > deadline - 3:
                break

            if len(edges) == 0:
                break

            idx = random.randint(0, len(edges) - 1)
            edge_to_remove = edges[idx]

            # Remove edge
            edges.pop(idx)
            edges_set.remove(edge_to_remove)
            for p in TRIPLE_PAIRS[ALL_TRIPLES.index(edge_to_remove)]:
                used_pairs.discard(p)
                used_pairs.discard((p[1], p[0]))

            # Find candidate
            candidates = []
            for t in ALL_TRIPLES:
                if t in edges_set:
                    continue
                valid = True
                for p in TRIPLE_PAIRS[ALL_TRIPLES.index(t)]:
                    if p in used_pairs or (p[1], p[0]) in used_pairs:
                        valid = False
                        break
                if valid:
                    candidates.append(t)

            if not candidates:
                # Restore
                edges.append(edge_to_remove)
                edges_set.add(edge_to_remove)
                for p in TRIPLE_PAIRS[ALL_TRIPLES.index(edge_to_remove)]:
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))
                continue

            new_edge = random.choice(candidates)

            # Add new edge
            edges.append(new_edge)
            edges_set.add(new_edge)
            for p in TRIPLE_PAIRS[ALL_TRIPLES.index(new_edge)]:
                used_pairs.add(p)
                used_pairs.add((p[1], p[0]))

            # Recalculate alpha
            edge_masks = []
            for e in edges:
                mask = (1 << e[0]) | (1 << e[1]) | (1 << e[2])
                edge_masks.append(mask)
            new_alpha = calculate_alpha_fast(edge_masks)

            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = edges[:]
                    if best_alpha <= 5:
                        break
            elif new_alpha == current_alpha:
                if random.random() < 0.3:
                    current_alpha = new_alpha
                else:
                    # Reject
                    edges.pop()
                    edges_set.remove(new_edge)
                    for p in TRIPLE_PAIRS[ALL_TRIPLES.index(new_edge)]:
                        used_pairs.discard(p)
                        used_pairs.discard((p[1], p[0]))
                    edges.append(edge_to_remove)
                    edges_set.add(edge_to_remove)
                    for p in TRIPLE_PAIRS[ALL_TRIPLES.index(edge_to_remove)]:
                        used_pairs.add(p)
                        used_pairs.add((p[1], p[0]))
            else:
                # Reject
                edges.pop()
                edges_set.remove(new_edge)
                for p in TRIPLE_PAIRS[ALL_TRIPLES.index(new_edge)]:
                    used_pairs.discard(p)
                    used_pairs.discard((p[1], p[0]))
                edges.append(edge_to_remove)
                edges_set.add(edge_to_remove)
                for p in TRIPLE_PAIRS[ALL_TRIPLES.index(edge_to_remove)]:
                    used_pairs.add(p)
                    used_pairs.add((p[1], p[0]))

        if best_alpha <= 5:
            break

    return [list(e) for e in best_edges]
