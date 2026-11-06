import time
import random
from itertools import combinations

# Precomputation
N = 16
ALL_TRIPLES = list(combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES)  # 560

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Generate all K(2,2,2) configurations
def generate_k222s():
    k222s = []
    for verts in combinations(range(N), 6):
        v = list(verts)
        for i in range(1, 6):
            pair0 = (v[0], v[i])
            remaining = [v[j] for j in range(6) if j != 0 and j != i]
            
            # Option 1
            pair1 = (remaining[0], remaining[1])
            pair2 = (remaining[2], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 2
            pair1 = (remaining[0], remaining[2])
            pair2 = (remaining[1], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 3
            pair1 = (remaining[0], remaining[3])
            pair2 = (remaining[1], remaining[2])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
    return k222s

K222S = generate_k222s()
NUM_K222 = len(K222S)

# Map each edge to the list of K222 indices it belongs to
edge_to_k222s = [[] for _ in range(NUM_TRIPLES)]
for k_idx, edges in enumerate(K222S):
    for e_idx in edges:
        edge_to_k222s[e_idx].append(k_idx)


def _greedy_repair(present, counts, absent_set, present_set):
    """Try to add any absent triple that fits. Returns (added_count, new_absent, new_present)."""
    added = 0
    # Build list of absent indices for iteration
    absent_list = list(absent_set)
    for idx in absent_list:
        valid = True
        for k_idx in edge_to_k222s[idx]:
            if counts[k_idx] == 7:
                valid = False
                break
        if valid:
            present[idx] = True
            absent_set.remove(idx)
            present_set.add(idx)
            for k_idx in edge_to_k222s[idx]:
                counts[k_idx] += 1
            added += 1
    return added


def _targeted_swap(present, counts, absent_set, present_set, deadline):
    """Perform a targeted swap: find a critical K222 (count=7), remove one of its edges,
    then try to add a blocked triple that was blocked by that K222."""
    # Find all critical K222s (count == 7)
    critical = [k for k in range(NUM_K222) if counts[k] == 7]
    if not critical or not absent_set:
        return False
    
    k_idx = random.choice(critical)
    edges_in_k222 = K222S[k_idx]
    
    # Find present edges in this critical K222
    present_edges_in_k222 = [e for e in edges_in_k222 if present[e]]
    if not present_edges_in_k222:
        return False
    
    rem_idx = random.choice(present_edges_in_k222)
    
    # Remove rem_idx
    present[rem_idx] = False
    absent_set.add(rem_idx)
    present_set.remove(rem_idx)
    for k_idx_2 in edge_to_k222s[rem_idx]:
        counts[k_idx_2] -= 1
    
    # Now try to add an absent triple that was blocked by k_idx
    # Find absent triples that include k_idx in their edge_to_k222s
    added_any = False
    # Sample some absent triples
    absent_list = list(absent_set)
    random.shuffle(absent_list)
    for new_idx in absent_list:
        if k_idx not in edge_to_k222s[new_idx]:
            continue
        # Check if new_idx is now addable
        valid = True
        for k_idx_3 in edge_to_k222s[new_idx]:
            if counts[k_idx_3] == 7:
                valid = False
                break
        if valid:
            present[new_idx] = True
            absent_set.remove(new_idx)
            present_set.add(new_idx)
            for k_idx_4 in edge_to_k222s[new_idx]:
                counts[k_idx_4] += 1
            added_any = True
            break
    
    if not added_any:
        # Revert
        present[rem_idx] = True
        absent_set.remove(rem_idx)
        present_set.add(rem_idx)
        for k_idx_5 in edge_to_k222s[rem_idx]:
            counts[k_idx_5] += 1
    
    return added_any


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_count = 0
    best_system = []

    triple_list = ALL_TRIPLES

    while time.time() < deadline:
        # 1. Greedy Initialization
        shuffled_indices = list(range(NUM_TRIPLES))
        random.shuffle(shuffled_indices)

        present = [False] * NUM_TRIPLES
        counts = [0] * NUM_K222
        current_count = 0

        for idx in shuffled_indices:
            valid = True
            for k_idx in edge_to_k222s[idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                present[idx] = True
                current_count += 1
                for k_idx in edge_to_k222s[idx]:
                    counts[k_idx] += 1

        # Greedy repair after initial greedy
        absent_set = set(i for i in range(NUM_TRIPLES) if not present[i])
        present_set = set(i for i in range(NUM_TRIPLES) if present[i])
        added = _greedy_repair(present, counts, absent_set, present_set)
        current_count += added

        if current_count > best_count:
            best_count = current_count
            best_system = [triple_list[i] for i in present_set]

        # 2. Local Search
        max_steps_per_restart = 8000
        step = 0
        no_improve = 0
        max_no_improve = 2000

        while time.time() < deadline and step < max_steps_per_restart and no_improve < max_no_improve:
            step += 1

            move_type = random.random()
            improved = False

            if move_type < 0.5:
                # Targeted swap using critical K222s
                if _targeted_swap(present, counts, absent_set, present_set, deadline):
                    # Count changed: removed 1, added 1 => net 0, but structure changed
                    # Actually we need to track count properly
                    pass
                # After targeted swap, try greedy repair
                added = _greedy_repair(present, counts, absent_set, present_set)
                if added > 0:
                    current_count += added
                    improved = True
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [triple_list[i] for i in present_set]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            elif move_type < 0.7:
                # Add Move: Pick an absent triple, add if possible
                if absent_set:
                    new_idx = random.choice(list(absent_set))
                    valid = True
                    for k_idx in edge_to_k222s[new_idx]:
                        if counts[k_idx] == 7:
                            valid = False
                            break
                    if valid:
                        present[new_idx] = True
                        absent_set.remove(new_idx)
                        present_set.add(new_idx)
                        current_count += 1
                        for k_idx in edge_to_k222s[new_idx]:
                            counts[k_idx] += 1
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_system = [triple_list[i] for i in present_set]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            elif move_type < 0.85:
                # Remove Move: Remove a random present triple
                if present_set:
                    rem_idx = random.choice(list(present_set))
                    present[rem_idx] = False
                    absent_set.add(rem_idx)
                    present_set.remove(rem_idx)
                    current_count -= 1
                    for k_idx in edge_to_k222s[rem_idx]:
                        counts[k_idx] -= 1
                    no_improve += 1
                else:
                    no_improve += 1

            else:
                # Random swap: remove one present, try to add one absent
                if present_set and absent_set:
                    rem_idx = random.choice(list(present_set))
                    present[rem_idx] = False
                    absent_set.add(rem_idx)
                    present_set.remove(rem_idx)
                    current_count -= 1
                    for k_idx in edge_to_k222s[rem_idx]:
                        counts[k_idx] -= 1
                    
                    added = _greedy_repair(present, counts, absent_set, present_set)
                    current_count += added
                    if added > 0:
                        improved = True
                        if current_count > best_count:
                            best_count = current_count
                            best_system = [triple_list[i] for i in present_set]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

        # Final greedy repair before next restart
        added = _greedy_repair(present, counts, absent_set, present_set)
        current_count += added
        if current_count > best_count:
            best_count = current_count
            best_system = [triple_list[i] for i in present_set]

    return best_system
