import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Precompute LCM conflict graph
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def greedy_color_bound(vertices, color_map):
        if not vertices:
            return 0
        vset = set(vertices)
        ordered = sorted(vertices, key=lambda v: -sum(1 for u in vertices if u != v and (adj[v] >> u) & 1))
        color_map.clear()
        next_color = 0
        for v in ordered:
            used_colors = set()
            for u in vertices:
                if u != v and u in color_map and (adj[v] >> u) & 1:
                    used_colors.add(color_map[u])
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
            next_color = max(next_color, c + 1)
        return next_color

    def branch_and_bound(candidates_mask, current_size, best_size, best_set, deadline):
        if (time.time() - start_time) > deadline:
            return best_size, best_set
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
            return best_size, best_set

        candidates_list = [i for i in range(1, N + 1) if (candidates_mask >> i) & 1]
        if not candidates_list:
            if current_size > best_size:
                best_size = current_size
            return best_size, best_set

        color_map = {}
        ub = greedy_color_bound(candidates_list, color_map)
        if current_size + ub <= best_size:
            return best_size, best_set

        max_color = -1
        pivot = None
        for v in candidates_list:
            if color_map.get(v, 0) > max_color:
                max_color = color_map[v]
                pivot = v

        if pivot is None:
            pivot = candidates_list[0]

        new_mask = candidates_mask & comp_adj[pivot]
        new_mask &= ~(1 << pivot)
        best_size, best_set = branch_and_bound(
            new_mask, current_size + 1, best_size, best_set, deadline
        )

        new_mask2 = candidates_mask & ~(1 << pivot)
        best_size, best_set = branch_and_bound(
            new_mask2, current_size, best_size, best_set, deadline
        )

        return best_size, best_set

    # Multi-start with local search
    def local_search(initial_set):
        current = set(initial_set)
        improved = True
        while improved:
            improved = False
            # Try to add elements
            for v in range(1, N + 1):
                if v not in current:
                    compatible = all((comp_adj[v] >> u) & 1 for u in current)
                    if compatible:
                        current.add(v)
                        improved = True
            # Try swaps: remove one, add two
            for v in list(current):
                temp = current - {v}
                added = False
                for u in range(1, N + 1):
                    if u not in temp:
                        compatible = all((comp_adj[u] >> w) & 1 for w in temp)
                        if compatible:
                            temp.add(u)
                            added = True
                            break
                if added and len(temp) > len(current):
                    current = temp
                    improved = True
        return current

    # Generate initial solutions from divisor hubs
    best_initial = []
    for m in [360, 480, 420, 336, 300, 240, 210, 180, 168, 144, 120, 96, 84, 72, 60, 48, 42, 36, 30, 24]:
        divs = get_divisors(m)
        if len(divs) > len(best_initial):
            best_initial = divs

    # Verify initial set
    init_set = set(best_initial)
    valid = True
    for i in range(len(best_initial)):
        for j in range(i+1, len(best_initial)):
            a, b = best_initial[i], best_initial[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                valid = False
                break
        if not valid:
            break

    if not valid:
        best_initial = list(range(1, 21))

    # Apply local search to improve initial solution
    improved_set = local_search(best_initial)
    best_size = len(improved_set)
    best_set = list(improved_set)

    # Run B&B from full universe with improved initial bound
    full_candidates = 0
    for i in range(1, N + 1):
        full_candidates |= (1 << i)

    best_size, best_set = branch_and_bound(
        full_candidates, 0, best_size, best_set, time_limit
    )

    # Final validation
    result = best_set
    result_set = set(result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                return best_initial

    return sorted(result)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
