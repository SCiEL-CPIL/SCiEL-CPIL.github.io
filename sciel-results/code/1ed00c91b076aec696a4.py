import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit

    n = 256

    # Precompute 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    # 4-AP: (a, a+d, a+2d, a+3d)

    aps3 = []
    aps4 = []

    # Map element to list of AP indices
    elem_to_aps3 = [[] for _ in range(n + 1)]
    elem_to_aps4 = [[] for _ in range(n + 1)]

    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps3.append((a, b, c))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1

    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps4.append((a, b, c, e))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    total_aps3 = len(aps3)
    total_aps4 = len(aps4)

    # Helper to count 3-APs in a set A
    def count_3aps(A):
        count = 0
        for a, b, c in aps3:
            if a in A and b in A and c in A:
                count += 1
        return count

    # Helper to check if adding x to A creates a 4-AP
    def is_valid_add(A, x):
        for ap_idx in elem_to_aps4[x]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(y in A for y in others):
                return False
        return True

    # Helper to count 4-APs in A
    def count_4aps(A):
        count = 0
        for a, b, c, d in aps4:
            if a in A and b in A and c in A and d in A:
                count += 1
        return count

    # Greedy construction
    def run_greedy(order=None):
        A = set()
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if is_valid_add(A, x):
                A.add(x)
        return A

    # Try multiple greedy orders
    best_set = set()
    best_score = 0

    # Standard ascending
    A = run_greedy()
    score = count_3aps(A)
    if score > best_score:
        best_score = score
        best_set = set(A)

    # Descending
    A = run_greedy(list(range(n, 0, -1)))
    score = count_3aps(A)
    if score > best_score:
        best_score = score
        best_set = set(A)

    # Several random shuffles
    rng = random.Random(42)
    for trial in range(8):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        A = run_greedy(order)
        # Ensure maximality
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x not in A and is_valid_add(A, x):
                    A.add(x)
                    changed = True
        score = count_3aps(A)
        if score > best_score:
            best_score = score
            best_set = set(A)

    # Local search: try improving via swaps
    A = set(best_set)
    current_score = best_score

    # Precompute for faster delta
    # For each element, store which 3-APs it's in
    # We'll use the elem_to_aps3 structure

    def get_delta_3aps_remove_add(A, u, v):
        """Compute delta in 3-AP count when removing u and adding v to A."""
        # 3-APs lost: those containing u with both other elements in A
        lost = 0
        for ap_idx in elem_to_aps3[u]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != u]
            if all(y in A for y in others):
                lost += 1

        # 3-APs gained: those containing v with both other elements in A, 
        # but NOT containing u (since u is removed)
        gained = 0
        for ap_idx in elem_to_aps3[v]:
            a, b, c = aps3[ap_idx]
            others = [y for y in (a, b, c) if y != v]
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1

        return gained - lost

    def get_delta_4aps_remove_add(A, u, v):
        """Compute delta in 4-AP count when removing u and adding v to A."""
        lost = 0
        for ap_idx in elem_to_aps4[u]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != u]
            if all(y in A for y in others):
                lost += 1

        gained = 0
        for ap_idx in elem_to_aps4[v]:
            a, b, c, d = aps4[ap_idx]
            others = [y for y in (a, b, c, d) if y != v]
            valid = True
            for y in others:
                if y not in A or y == u:
                    valid = False
                    break
            if valid:
                gained += 1

        return gained - lost

    # Simulated annealing
    T = 5.0
    cooling = 0.9995
    min_T = 0.1

    while time.time() < deadline and T > min_T:
        if random.random() < 0.3:
            # Try insertion
            v = random.randint(1, n)
            if v not in A and is_valid_add(A, v):
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    a, b, c = aps3[ap_idx]
                    others = [y for y in (a, b, c) if y != v]
                    if all(y in A for y in others):
                        gained += 1
                if gained > 0:
                    A.add(v)
                    current_score += gained
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(A)
                elif random.random() < math.exp(gained / T):
                    A.add(v)
                    current_score += gained
        else:
            # Try swap
            if not A:
                continue
            u = random.choice(list(A))
            v = random.randint(1, n)
            if v in A or v == u:
                continue

            delta_4 = get_delta_4aps_remove_add(A, u, v)
            if delta_4 > 0:
                continue  # Would create 4-APs

            delta_3 = get_delta_3aps_remove_add(A, u, v)

            if delta_3 > 0:
                A.remove(u)
                A.add(v)
                current_score += delta_3
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(A)
            else:
                if random.random() < math.exp(delta_3 / T):
                    A.remove(u)
                    A.add(v)
                    current_score += delta_3

        T *= cooling

    # Final verification
    if count_4aps(best_set) != 0:
        # Fallback
        best_set = run_greedy()

    return sorted(best_set)
