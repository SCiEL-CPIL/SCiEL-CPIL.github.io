import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    # Precompute all 4-APs and index by element
    # AP: (a, a+d, a+2d, a+3d) with d>=1, a>=1, a+3d <= N
    element_to_aps = {i: [] for i in range(1, N + 1)}
    ap_list = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            ap_id = len(ap_list)
            ap_list.append(ap)
            for elem in ap:
                element_to_aps[elem].append(ap_id)

    # Use a boolean list for O(1) membership
    inS = [False] * (N + 1)

    def can_add(x):
        """Check if x can be added to current set without creating a 4-AP."""
        if inS[x]:
            return True  # already in set, no conflict
        for ap_id in element_to_aps[x]:
            ap = ap_list[ap_id]
            # Count how many of the other 3 elements are in S
            cnt = 0
            for y in ap:
                if y != x and inS[y]:
                    cnt += 1
            if cnt >= 3:
                return False
        return True

    def add_to_set(x):
        inS[x] = True

    def remove_from_set(x):
        inS[x] = False

    def set_size():
        return sum(inS)

    # Phase 1: Greedy construction in natural order
    S = set()
    for x in range(1, N + 1):
        if can_add(x):
            add_to_set(x)
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # Phase 2: Local search
    random.seed(42)
    rejected = [x for x in range(1, N + 1) if x not in S]

    # Try to add rejected elements in various orders
    while time.time() < deadline:
        # Try adding all possible rejected elements
        added_any = False
        for x in rejected[:]:
            if time.time() > deadline:
                break
            if can_add(x):
                add_to_set(x)
                S.add(x)
                rejected.remove(x)
                added_any = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)

        if not rejected:
            break

        if not added_any:
            # Stuck: try a swap. Remove a random element from S, 
            # then try to add multiple rejected elements.
            if not S:
                break

            # Try several random removals to find a productive one
            found_swap = False
            for attempt in range(20):
                if time.time() > deadline:
                    break

                y = random.choice(list(S))
                remove_from_set(y)
                S.discard(y)
                rejected.append(y)

                # Try to add as many rejected elements as possible
                added_count = 0
                for z in rejected[:]:
                    if z == y:
                        continue
                    if can_add(z):
                        add_to_set(z)
                        S.add(z)
                        rejected.remove(z)
                        added_count += 1

                if added_count > 0:
                    # Good swap: we removed 1 and added >= 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    found_swap = True
                    break
                else:
                    # Revert: put y back
                    add_to_set(y)
                    S.add(y)
                    rejected.remove(y)

            if not found_swap:
                # No productive swap found. Try a different strategy:
                # Remove 2 elements and try to add 3.
                if len(S) >= 2:
                    y1 = random.choice(list(S))
                    S.discard(y1)
                    remove_from_set(y1)
                    rejected.append(y1)

                    remaining = [x for x in S if x != y1]
                    if remaining:
                        y2 = random.choice(remaining)
                        S.discard(y2)
                        remove_from_set(y2)
                        rejected.append(y2)

                        added_count = 0
                        for z in rejected[:]:
                            if z == y1 or z == y2:
                                continue
                            if can_add(z):
                                add_to_set(z)
                                S.add(z)
                                rejected.remove(z)
                                added_count += 1

                        if added_count >= 2:
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                        else:
                            # Revert
                            add_to_set(y1)
                            S.add(y1)
                            rejected.remove(y1)
                            if y2 in rejected:
                                add_to_set(y2)
                                S.add(y2)
                                rejected.remove(y2)

    # Final greedy pass
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if not inS[x] and can_add(x):
            add_to_set(x)
            S.add(x)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    return sorted(best_S)

# EVOLVE-BLOCK-END
