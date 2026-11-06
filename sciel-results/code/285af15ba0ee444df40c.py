"""Rewritten program for triple systems on 16 points without K(2,2,2)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Generate all ways to partition 6 points into 3 pairs
    def get_partitions_into_pairs(six_points):
        p0, p1, p2, p3, p4, p5 = six_points
        yield ((p0,p1), (p2,p3), (p4,p5))
        yield ((p0,p1), (p2,p4), (p3,p5))
        yield ((p0,p1), (p2,p5), (p3,p4))
        yield ((p0,p2), (p1,p3), (p4,p5))
        yield ((p0,p2), (p1,p4), (p3,p5))
        yield ((p0,p2), (p1,p5), (p3,p4))
        yield ((p0,p3), (p1,p2), (p4,p5))
        yield ((p0,p3), (p1,p4), (p2,p5))
        yield ((p0,p3), (p1,p5), (p2,p4))
        yield ((p0,p4), (p1,p2), (p3,p5))
        yield ((p0,p4), (p1,p3), (p2,p5))
        yield ((p0,p4), (p1,p5), (p2,p3))
        yield ((p0,p5), (p1,p2), (p3,p4))
        yield ((p0,p5), (p1,p3), (p2,p4))
        yield ((p0,p5), (p1,p4), (p2,p3))

    # Precompute all K(2,2,2) patterns
    # For each 6-point subset, for each partition into 3 pairs,
    # compute the 8 transversal triples
    triple_to_k222s = defaultdict(list)  # triple_idx -> list of k222_indices
    k222_triple_sets = []  # each element is a frozenset of 8 triple indices

    for six_points in combinations(range(n), 6):
        for pairs in get_partitions_into_pairs(six_points):
            (a, a_p), (b, b_p), (c, c_p) = pairs
            transversal_triples = [
                (a, b, c), (a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p)
            ]
            transversal_triples = [tuple(sorted(t)) for t in transversal_triples]
            triple_indices = []
            valid = True
            for t in transversal_triples:
                if t in triple_to_idx:
                    triple_indices.append(triple_to_idx[t])
                else:
                    valid = False
                    break
            if valid:
                k222_triple_sets.append(frozenset(triple_indices))
                for idx in triple_indices:
                    triple_to_k222s[idx].append(len(k222_triple_sets) - 1)

    num_k222s = len(k222_triple_sets)

    # Track for each K(2,2,2) how many of its 8 triples are present
    k222_counts = [0] * num_k222s

    def is_valid_add(triple_idx):
        """Check if adding triple_idx creates a K(2,2,2) violation"""
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            if k222_counts[k222_idx] == 7:
                return False
        return True

    def add_triple(triple_idx, current_set):
        current_set[triple_idx] = True
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            k222_counts[k222_idx] += 1

    def remove_triple(triple_idx, current_set):
        current_set[triple_idx] = False
        for k222_idx in triple_to_k222s.get(triple_idx, []):
            k222_counts[k222_idx] -= 1

    # Weighted greedy initialization with dynamic weights
    # Start with all triples having weight 1.0
    # Iteratively select the triple with highest weight that doesn't create a violation
    # After selection, decrease weights of conflicting triples

    weights = [1.0] * num_triples
    current_set = [False] * num_triples
    current_count = 0

    # Multiple greedy passes with different weight initializations
    best_set = [False] * num_triples
    best_count = 0

    rng = random.Random(12345)

    for pass_num in range(5):
        # Reset
        for i in range(num_triples):
            weights[i] = 1.0
            current_set[i] = False
        for i in range(num_k222s):
            k222_counts[i] = 0
        current_count = 0

        # Add some randomness to weights
        if pass_num > 0:
            for i in range(num_triples):
                weights[i] = 0.5 + rng.random()

        available = set(range(num_triples))
        while available and time.time() < deadline - 5:
            # Find the triple with highest weight among available
            best_idx = -1
            best_w = -1
            for idx in available:
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_idx = idx

            if best_idx == -1:
                break

            if is_valid_add(best_idx):
                add_triple(best_idx, current_set)
                current_count += 1
                available.remove(best_idx)

                # Decrease weights of triples that conflict with best_idx
                for k222_idx in triple_to_k222s.get(best_idx, []):
                    for other_idx in k222_triple_sets[k222_idx]:
                        if other_idx != best_idx and other_idx in available:
                            weights[other_idx] *= 0.7
            else:
                # Remove from available since it can't be added
                available.remove(best_idx)

        if current_count > best_count:
            best_count = current_count
            best_set = current_set[:]

    # Local search with simulated annealing
    current_set = best_set[:]
    current_count = best_count

    # Reset k222_counts
    for i in range(num_k222s):
        k222_counts[i] = 0
    for i in range(num_triples):
        if current_set[i]:
            for k222_idx in triple_to_k222s.get(i, []):
                k222_counts[k222_idx] += 1

    temperature = 2.0
    cooling_rate = 0.999
    min_temperature = 0.005

    rng = random.Random(99999)

    while time.time() < deadline - 2 and temperature > min_temperature:
        # Try a random move
        move_type = rng.random()

        if move_type < 0.3:
            # Remove a random triple
            if current_count == 0:
                temperature *= cooling_rate
                continue
            idx = rng.randrange(num_triples)
            if not current_set[idx]:
                continue
            remove_triple(idx, current_set)
            new_count = current_count - 1
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_set = current_set[:]
        elif move_type < 0.6:
            # Add a random triple
            idx = rng.randrange(num_triples)
            if current_set[idx]:
                continue
            if not is_valid_add(idx):
                continue
            add_triple(idx, current_set)
            new_count = current_count + 1
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_set = current_set[:]
        else:
            # Swap: remove one, add another
            if current_count == 0:
                temperature *= cooling_rate
                continue
            idx_remove = rng.randrange(num_triples)
            if not current_set[idx_remove]:
                continue
            idx_add = rng.randrange(num_triples)
            if current_set[idx_add]:
                continue
            remove_triple(idx_remove, current_set)
            if is_valid_add(idx_add):
                add_triple(idx_add, current_set)
                # Count stays the same
                if current_count > best_count:
                    best_count = current_count
                    best_set = current_set[:]
            else:
                add_triple(idx_remove, current_set)

        temperature *= cooling_rate

    # Final polish: try to add any remaining valid triples
    for i in range(num_triples):
        if not best_set[i] and is_valid_add(i):
            add_triple(i, best_set)
            best_count += 1

    # Convert best_set back to list of triples
    result = []
    for i, included in enumerate(best_set):
        if included:
            result.append(list(all_triples[i]))

    return result

# EVOLVE-BLOCK-END
