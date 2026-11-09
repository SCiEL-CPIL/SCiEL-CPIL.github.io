import time
import random
import sys

N = 256
DEADLINE = 75.0  # safety margin before 90s kill

def _build_4ap_structures():
    """Return list of 4-APs (each a 4-tuple) and for each x, list of 4-AP indices containing x."""
    four_aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    # For each element, which 4-APs contain it
    elem_4aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_4aps[x].append(idx)
    return four_aps, elem_4aps

def _build_3ap_structures():
    """Return list of 3-APs (each a 3-tuple (a,b,c)) and for each x, list of 3-AP indices containing x."""
    three_aps = []
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2):  # a+c even => b integer
            b = (a + c) // 2
            three_aps.append((a, b, c))
    elem_3aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_3aps[x].append(idx)
    return three_aps, elem_3aps

def _count_3aps_in_set(selected_set, three_aps):
    """Count 3-APs fully in selected_set."""
    cnt = 0
    for (a, b, c) in three_aps:
        if a in selected_set and b in selected_set and c in selected_set:
            cnt += 1
    return cnt

def _is_4ap_free(selected_set, four_aps, four_ap_flags):
    """Check if any 4-AP is fully in selected_set. four_ap_flags is a list of booleans."""
    for idx, ap in enumerate(four_aps):
        if all(x in selected_set for x in ap):
            return False
    return True

def construct():
    start_time = time.time()
    random.seed(42)

    four_aps, elem_4aps = _build_4ap_structures()
    three_aps, elem_3aps = _build_3ap_structures()
    num_4aps = len(four_aps)
    num_3aps = len(three_aps)

    # Precompute for each 3-AP, the set of 4-APs that would be violated if we add an element
    # Actually, for feasibility: adding x is OK iff no 4-AP containing x has all other 3 elements already in set.

    def can_add(x, selected_set):
        """Check if adding x keeps 4-AP-free."""
        for ap_idx in elem_4aps[x]:
            ap = four_aps[ap_idx]
            # Check if the other 3 elements are all in selected_set
            all_others = True
            for y in ap:
                if y != x and y not in selected_set:
                    all_others = False
                    break
            if all_others:
                return False
        return True

    # Greedy seed: build a 4-AP-free set greedily, preferring elements that create many 3-APs
    # We'll do multiple random-order greedy constructions and keep the best.

    best_set = set()
    best_score = 0

    # Precompute for each element, the 3-APs it participates in (for scoring)
    # Greedy construction with random ordering
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        score = _count_3aps_in_set(s, three_aps)
        if score > best_score:
            best_score = score
            best_set = set(s)

    # Local search: try adding, removing, swapping elements
    # Maintain a set and its 3-AP count incrementally

    def score_set(s):
        return _count_3aps_in_set(s, three_aps)

    current_set = set(best_set)
    current_score = best_score

    # For incremental updates, track which 3-APs are satisfied
    def get_3ap_flags(s):
        flags = [False] * num_3aps
        for idx, (a, b, c) in enumerate(three_aps):
            if a in s and b in s and c in s:
                flags[idx] = True
        return flags

    # Local search phase
    no_improve_count = 0
    max_no_improve = 200

    while time.time() - start_time < DEADLINE - 5:
        improved = False

        # Try adding elements
        candidates = [x for x in range(1, N + 1) if x not in current_set]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > DEADLINE - 5:
                break
            if can_add(x, current_set):
                # Count how many 3-APs would be added
                gain = 0
                for ap_idx in elem_3aps[x]:
                    ap = three_aps[ap_idx]
                    others = [y for y in ap if y != x]
                    if all(y in current_set for y in others):
                        gain += 1
                if gain > 0:
                    current_set.add(x)
                    current_score += gain
                    improved = True
                    no_improve_count = 0
                    break  # restart search after improvement

        if not improved:
            # Try removing elements (to free up space for better additions)
            # Only remove if it doesn't hurt too much, or as part of a swap
            pass

        # Try swaps: remove one, add another
        if not improved:
            removed_candidates = list(current_set)
            random.shuffle(removed_candidates)
            added_candidates = [x for x in range(1, N + 1) if x not in current_set]
            random.shuffle(added_candidates)

            for x_out in removed_candidates:
                if time.time() - start_time > DEADLINE - 5:
                    break
                for x_in in added_candidates:
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    # Check if removing x_out and adding x_in is valid
                    temp_set = set(current_set)
                    temp_set.discard(x_out)
                    temp_set.add(x_in)
                    # Check 4-AP free
                    valid = True
                    for ap_idx in elem_4aps[x_in]:
                        ap = four_aps[ap_idx]
                        all_others = True
                        for y in ap:
                            if y != x_in and y not in temp_set:
                                all_others = False
                                break
                        if all_others:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Compute score change
                    # 3-APs lost by removing x_out
                    loss = 0
                    for ap_idx in elem_3aps[x_out]:
                        ap = three_aps[ap_idx]
                        others = [y for y in ap if y != x_out]
                        if all(y in current_set for y in others):
                            loss += 1
                    # 3-APs gained by adding x_in (in temp_set)
                    gain = 0
                    for ap_idx in elem_3aps[x_in]:
                        ap = three_aps[ap_idx]
                        others = [y for y in ap if y != x_in]
                        if all(y in temp_set for y in others):
                            gain += 1
                    # Net change
                    net = gain - loss
                    if net > 0:
                        current_set = temp_set
                        current_score += net
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a different random restart
                break

    return sorted(current_set)


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Elements: {result}")
